#!/usr/bin/env python3
"""Prompt construction for the Gene Completion benchmark.

Each panel uses a fixed, published prompt scheme:

* Prokaryote/archaea: the prompt is the ``upstream_len`` nucleotides immediately
  upstream of the coding sequence plus the first ``fraction`` of the coding
  region (rounded to whole codons). The model must complete the remaining CDS.

* Eukaryote: the prompt is the first ``fraction`` of the genomic window (upstream
  + gene, including introns). The model must complete the remaining genomic
  sequence, which is then spliced for scoring.
"""
from __future__ import annotations

from typing import Tuple

PROMPT_FRACTION = 0.30
PROK_UPSTREAM_LEN = 1000


def prokaryote_prompt(
    genomic: str,
    cds_start: int = 5000,
    upstream_len: int = PROK_UPSTREAM_LEN,
    fraction: float = PROMPT_FRACTION,
) -> Tuple[str, int]:
    """Return ``(prompt, prompt_cds_aa)`` for a prokaryotic gene.

    ``prompt_cds_aa`` is the number of coding residues contained in the prompt;
    AA recovery is measured over the reference protein beyond this index.
    """
    genomic = "".join(genomic.split()).upper()
    coding_nt = len(genomic) - cds_start
    coding_aa_take = round(coding_nt / 3.0 * fraction)
    take_nt = coding_aa_take * 3
    start = max(0, cds_start - upstream_len)
    prompt = genomic[start:cds_start] + genomic[cds_start:cds_start + take_nt]
    return prompt, coding_aa_take


def eukaryote_prompt(genomic: str, fraction: float = PROMPT_FRACTION) -> Tuple[str, int]:
    """Return ``(prompt, prompt_len)`` for a eukaryotic gene."""
    genomic = "".join(genomic.split()).upper()
    prompt_len = int(len(genomic) * fraction)
    return genomic[:prompt_len], prompt_len
