#!/usr/bin/env python3
"""Scoring for the Gene Completion benchmark: % amino-acid (AA) recovery.

Two panels, two alignment strategies (because eukaryotic genes have introns):

* Prokaryote/archaea panel -- the generated coding sequence is translated and
  globally aligned to the reference protein; AA recovery is the percent identity
  over the *non-prompt* region of the protein.

* Eukaryote panel -- the generated genomic sequence is spliced against the
  reference protein with exonerate (protein2genome), translated, and compared to
  the spliced reference protein over the non-prompt region.

In both cases "recovery" is measured only over the portion of the protein the
model had to generate (the non-prompt region), never the part it was given.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq


# ── Translation ─────────────────────────────────────────────────────────────

def translate_dna(dna: str, to_stop: bool = True) -> str:
    """Translate a DNA string in frame 0, trimming to a multiple of three."""
    dna = "".join(dna.split()).upper()
    usable = len(dna) - (len(dna) % 3)
    if usable <= 0:
        return ""
    return str(Seq(dna[:usable]).translate(to_stop=to_stop))


def positional_identity(a: str, b: str) -> float:
    """Ungapped positional percent identity over the shorter of the two strings.

    Matches the metric used in the published eukaryotic pipeline.
    """
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return 100.0 * matches / n


# ── Prokaryote panel: global protein alignment ──────────────────────────────

_ALIGNER: Optional[PairwiseAligner] = None


def _aligner() -> PairwiseAligner:
    global _ALIGNER
    if _ALIGNER is None:
        a = PairwiseAligner()
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score = -11
        a.extend_gap_score = -1
        a.mode = "global"
        _ALIGNER = a
    return _ALIGNER


def aligned_identity_after(query_aa: str, ref_aa: str, ref_start: int) -> float:
    """Percent identity between query and reference proteins, counted only over
    reference residues at index >= ``ref_start`` (the non-prompt region).

    A global alignment maps query residues onto the reference so insertions and
    deletions are handled correctly.
    """
    if not query_aa or not ref_aa:
        return float("nan")
    aln = _aligner().align(ref_aa, query_aa)[0]
    matches = aligned = 0
    # aln.indices is a 2 x L array of residue indices per column (-1 for a gap).
    # Identity is counted over columns where BOTH sequences are aligned (no gap)
    # and the reference residue lies in the non-prompt region (index >= ref_start).
    ref_row, qry_row = aln.indices
    for r, q in zip(ref_row, qry_row):
        if r >= ref_start and q >= 0:
            aligned += 1
            if ref_aa[r] == query_aa[q]:
                matches += 1
    if aligned == 0:
        return float("nan")
    return 100.0 * matches / aligned


def score_prokaryote(
    generated_full: str,
    reference_protein: str,
    upstream_len: int,
    prompt_cds_aa: int,
) -> float:
    """AA recovery for a prokaryotic completion.

    ``generated_full`` is prompt + generation; the coding sequence begins at
    ``upstream_len`` within it. ``prompt_cds_aa`` is the number of coding residues
    that were part of the prompt (the non-prompt region starts there).
    """
    coding = generated_full[upstream_len:]
    gen_protein = translate_dna(coding, to_stop=True)
    return aligned_identity_after(gen_protein, reference_protein, prompt_cds_aa)


# ── Eukaryote panel: exonerate spliced alignment ────────────────────────────

def _resolve(tool: str) -> str:
    if Path(tool).exists():
        return tool
    found = shutil.which(tool)
    if found is None:
        raise FileNotFoundError(
            f"'{tool}' not found. Install with: conda install -c bioconda exonerate"
        )
    return found


def exonerate_spliced_protein(
    dna_seq: str,
    ref_protein: str,
    exonerate_path: str = "exonerate",
) -> Tuple[str, str]:
    """Align ``ref_protein`` to ``dna_seq`` with exonerate protein2genome, splice
    the predicted exons, and return ``(spliced_cds, spliced_protein)``.

    Returns ("", "") if exonerate finds no exons.
    """
    exe = _resolve(exonerate_path)
    cleaned = "".join(c if c in "ACGT" else "N" for c in "".join(dna_seq.split()).upper())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "q.faa").write_text(f">query\n{ref_protein}\n")
        (tmp / "t.fna").write_text(f">target\n{cleaned}\n")
        cmd = [
            exe, "--model", "protein2genome",
            "--showalignment", "no", "--showvulgar", "no",
            "--showquerygff", "no", "--showtargetgff", "yes",
            str(tmp / "q.faa"), str(tmp / "t.fna"),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError:
            return "", ""

    exons: List[Tuple[int, int]] = []
    strand = "+"
    for line in out.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9 or parts[2] != "exon":
            continue
        exons.append((int(parts[3]), int(parts[4])))
        strand = parts[6]
    if not exons:
        return "", ""

    exons.sort(key=lambda x: x[0])
    cds = "".join(cleaned[s - 1:e] for s, e in exons if 1 <= s <= e <= len(cleaned))
    if strand == "-":
        cds = str(Seq(cds).reverse_complement())
    return cds, translate_dna(cds, to_stop=False)


def score_eukaryote(
    generated_full: str,
    reference_spliced_protein: str,
    prompt_len: int,
    reference_protein: str,
    exonerate_path: str = "exonerate",
) -> float:
    """AA recovery for a eukaryotic completion.

    The generated genomic sequence is spliced against ``reference_protein`` to
    recover its protein, then compared to ``reference_spliced_protein`` (the
    exonerate-spliced protein of the *true* genomic) over the non-prompt region.
    """
    _, gen_protein = exonerate_spliced_protein(generated_full, reference_protein, exonerate_path)
    if not gen_protein or not reference_spliced_protein:
        return float("nan")
    prompt_aa = prompt_len // 3
    return positional_identity(
        reference_spliced_protein[prompt_aa:], gen_protein[prompt_aa:]
    )
