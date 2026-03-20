#!/usr/bin/env python3
"""Mutation type analysis: BLASTn hits vs query genomes.

For each of 4 PhiX174-like phage datasets, BLASTn each genome against NCBI nt,
take the top 10 hits, fetch annotated GenBank records for those hits, classify
all mutations by type (synonymous, nonsynonymous, indel, intergenic) per gene,
and plot distributions comparing datasets.

Usage:
    conda run -n sk_evo2_exp python mutation_type_analysis.py
    conda run -n sk_evo2_exp python mutation_type_analysis.py --test
"""

import argparse
import csv
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Blast import NCBIXML, NCBIWWW
from Bio.Seq import Seq

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "cache"
BLAST_CACHE = CACHE_DIR / "blast_results"
GBK_CACHE = CACHE_DIR / "genbank_records"

DATASETS = {
    "AI-generated": BASE_DIR / "data/viable_generated_phage_genomes.fasta",
    "Lab-evolved": BASE_DIR / "data/wichman2005_lt180_genomes.fasta",
    "Wild isolates": BASE_DIR / "data/rokyta2006_phix174like_genomes.fasta",
    "PhiX174 variants": BASE_DIR / "data/phage_sft_genomes_phix174_variants.fna",
}

# Entrez
Entrez.email = ""
Entrez.api_key = ""

# Gene ordering for plots
GENE_ORDER = ["A", "A*", "B", "K", "C", "D", "E", "J", "F", "G", "H", "intergenic"]

GENE_NAME_MAP = {
    "A protein": "A",
    "A* protein": "A*",
    "Protein A": "A",
    "Protein A*": "A*",
    "protein A": "A",
    "protein A*": "A*",
    "Protein B": "B",
    "protein B": "B",
    "Protein K": "K",
    "protein K": "K",
    "Protein C": "C",
    "protein C": "C",
    "Protein D": "D",
    "protein D": "D",
    "Protein E": "E",
    "protein E": "E",
    "Protein J": "J",
    "protein J": "J",
    "Protein F": "F",
    "protein F": "F",
    "Protein G": "G",
    "protein G": "G",
    "Protein H": "H",
    "protein H": "H",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Utility ────────────────────────────────────────────────────────────────────

def sanitize_filename(name):
    """Sanitize a string for use as a filename."""
    return re.sub(r'[^\w\-.]', '_', name)


def clean_fasta_for_mafft(input_fasta):
    """Remove gap characters and replace IUPAC ambiguity codes with N."""
    records = list(SeqIO.parse(input_fasta, "fasta"))
    needs_clean = any('.' in str(r.seq) or re.search(r'[^ACGTacgt\n]', str(r.seq))
                       for r in records)
    if not needs_clean:
        return input_fasta

    cleaned = str(Path(input_fasta).with_suffix('.cleaned.fasta'))
    with open(cleaned, 'w') as f:
        for rec in records:
            seq = str(rec.seq).replace('.', '')
            seq = re.sub(r'[^ACGTacgt]', 'N', seq)
            f.write(f">{rec.description}\n{seq}\n")
    return cleaned


# ── Step 2: BLASTn via NCBI remote API ─────────────────────────────────────────

BATCH_SIZE = 15  # NCBI API limit: ~16 genomes per multi-FASTA submission


def run_blast_for_dataset(dataset_name, fasta_path):
    """Run BLASTn via NCBI API in batches, caching XML per batch."""
    dataset_cache = BLAST_CACHE / sanitize_filename(dataset_name)
    dataset_cache.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    log.info("Dataset '%s': %d genomes", dataset_name, len(records))

    # Split into batches
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    log.info("  Split into %d batches of up to %d genomes", len(batches), BATCH_SIZE)

    all_hits = {}

    for batch_idx, batch_records in enumerate(batches):
        xml_path = dataset_cache / f"blast_batch_{batch_idx}.xml"

        if xml_path.exists() and xml_path.stat().st_size > 0:
            # Verify cached result doesn't contain NCBI server errors
            if _blast_xml_has_errors(xml_path):
                log.warning("  Batch %d cached XML has errors, re-submitting", batch_idx)
                xml_path.unlink()
            else:
                log.info("  Batch %d/%d: using cached results (%d genomes)",
                         batch_idx + 1, len(batches), len(batch_records))

        if not (xml_path.exists() and xml_path.stat().st_size > 0):
            # Build cleaned multi-FASTA for this batch
            fasta_lines = []
            for rec in batch_records:
                clean_seq = str(rec.seq).replace('.', '').replace('-', '')
                clean_seq = re.sub(r'[^ACGTacgt]', 'N', clean_seq)
                fasta_lines.append(f">{rec.description}\n{clean_seq}")
            fasta_str = "\n".join(fasta_lines) + "\n"

            log.info("  Batch %d/%d: submitting %d queries to NCBI BLAST API...",
                     batch_idx + 1, len(batches), len(batch_records))
            for attempt in range(3):
                try:
                    result_handle = NCBIWWW.qblast(
                        program="blastn",
                        database="nt",
                        sequence=fasta_str,
                        megablast=True,
                        hitlist_size=50,
                        format_type="XML",
                    )
                    with open(xml_path, 'w') as f:
                        f.write(result_handle.read())
                    result_handle.close()

                    # Verify the result
                    if _blast_xml_has_errors(xml_path):
                        log.warning("  Batch %d attempt %d: NCBI returned errors, retrying...",
                                    batch_idx, attempt + 1)
                        xml_path.unlink()
                        time.sleep(30)
                        continue

                    log.info("  Batch %d/%d: BLAST complete", batch_idx + 1, len(batches))
                    break
                except Exception as e:
                    log.warning("  Batch %d attempt %d failed: %s", batch_idx, attempt + 1, e)
                    if attempt < 2:
                        time.sleep(30)
                    else:
                        log.error("  Batch %d failed after 3 attempts", batch_idx)

            # Brief pause between batches
            time.sleep(2)

        # Parse this batch
        if xml_path.exists() and xml_path.stat().st_size > 0:
            batch_hits = parse_multi_blast_xml(xml_path, batch_records)
            all_hits.update(batch_hits)

    log.info("  Total: parsed hits for %d/%d queries", len(all_hits), len(records))
    return all_hits


def _blast_xml_has_errors(xml_path):
    """Check if a BLAST XML file contains NCBI server error messages."""
    with open(xml_path) as f:
        content = f.read(50000)  # Check first 50KB
    return "Process size limit exceeded" in content or "Error:" in content


def parse_multi_blast_xml(xml_path, query_records):
    """Parse multi-query BLAST XML and return {query_id: [hits]} for each query."""
    with open(xml_path) as f:
        try:
            blast_records = list(NCBIXML.parse(f))
        except Exception as e:
            log.error("Failed to parse %s: %s", xml_path, e)
            return {}

    all_hits = {}

    # Match BLAST records to query records by order
    for idx, blast_record in enumerate(blast_records):
        if idx < len(query_records):
            query_id = query_records[idx].description
        else:
            query_id = blast_record.query or f"query_{idx}"

        query_acc = query_id.split()[0].split('.')[0]

        seen_accessions = set()
        hits = []

        for alignment in blast_record.alignments:
            acc = extract_accession(alignment)
            if acc is None:
                continue

            acc_base = acc.split('.')[0]

            # Self-hit filtering
            if acc_base == query_acc:
                continue

            if acc in seen_accessions:
                continue
            seen_accessions.add(acc)

            best_hsp = max(alignment.hsps, key=lambda h: h.score)

            # Percent identity from best HSP (matches NCBI website display)
            pct_id = 100.0 * best_hsp.identities / best_hsp.align_length

            # Query coverage: union of all HSP ranges on the query
            query_len = blast_record.query_length or blast_record.query_letters
            covered_positions = set()
            for hsp in alignment.hsps:
                covered_positions.update(range(hsp.query_start, hsp.query_end + 1))
            query_cover = 100.0 * len(covered_positions) / query_len if query_len else 0

            # Aggregate stats across all HSPs
            total_align_length = sum(h.align_length for h in alignment.hsps)
            total_identities = sum(h.identities for h in alignment.hsps)
            total_gaps = sum(h.gaps or 0 for h in alignment.hsps)

            hits.append({
                "accession": acc,
                "max_score": best_hsp.bits,
                "total_score": sum(h.bits for h in alignment.hsps),
                "evalue": best_hsp.expect,
                "pct_identity": pct_id,
                "query_cover": query_cover,
                "align_length": total_align_length,
                "subject_length": alignment.length,
                "gaps": total_gaps,
                "mismatches": total_align_length - total_identities - total_gaps,
                "n_hsps": len(alignment.hsps),
                "hit_def": alignment.hit_def[:80],
            })

            if len(hits) >= 10:
                break

        all_hits[query_id] = hits

    log.info("  Parsed hits for %d/%d queries", len(all_hits), len(query_records))
    return all_hits


def extract_accession(alignment):
    """Extract a clean accession from a BLAST alignment object."""
    if hasattr(alignment, 'accession') and alignment.accession:
        acc = alignment.accession
        if acc and acc != "Unknown":
            return acc

    hit_id = alignment.hit_id
    parts = hit_id.split('|')
    for i, p in enumerate(parts):
        if p in ('ref', 'gb', 'emb', 'dbj') and i + 1 < len(parts):
            return parts[i + 1].rstrip('|')

    hit_def = alignment.hit_def
    match = re.match(r'(\w+\.\d+)', hit_def)
    if match:
        return match.group(1)

    first = hit_id.split('|')[0] if '|' in hit_id else hit_id.split()[0]
    if re.match(r'^[A-Z]{1,2}_?\d+', first):
        return first

    return None


# ── Step 3: Fetch GenBank records ──────────────────────────────────────────────

def fetch_genbank_records(all_hits):
    """Fetch GenBank records for all unique accessions across all datasets."""
    GBK_CACHE.mkdir(parents=True, exist_ok=True)

    # Collect all unique accessions
    all_accessions = set()
    for dataset_hits in all_hits.values():
        for query_hits in dataset_hits.values():
            for hit in query_hits:
                all_accessions.add(hit["accession"])

    log.info("Total unique accessions to fetch: %d", len(all_accessions))

    # Fetch missing records
    to_fetch = [acc for acc in all_accessions if not (GBK_CACHE / f"{acc}.gbk").exists()]
    log.info("Already cached: %d, need to fetch: %d", len(all_accessions) - len(to_fetch), len(to_fetch))

    for i, acc in enumerate(to_fetch):
        gbk_path = GBK_CACHE / f"{acc}.gbk"
        try:
            log.info("  Fetching [%d/%d]: %s", i + 1, len(to_fetch), acc)
            handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
            with open(gbk_path, 'w') as f:
                f.write(handle.read())
            handle.close()
            time.sleep(0.1)
        except Exception as e:
            log.error("  Failed to fetch %s: %s", acc, e)

    return all_accessions


# ── Step 4: QC GenBank annotations ────────────────────────────────────────────

def qc_genbank_records(accessions):
    """QC GenBank records: check gene count, duplicates, completeness."""
    qc_results = []
    valid_accessions = set()

    expected_genes = {"A", "B", "C", "D", "E", "F", "G", "H", "J"}

    for acc in sorted(accessions):
        gbk_path = GBK_CACHE / f"{acc}.gbk"
        if not gbk_path.exists():
            qc_results.append({"accession": acc, "status": "MISSING", "reason": "GBK file not found"})
            continue

        try:
            record = SeqIO.read(gbk_path, "genbank")
        except Exception as e:
            qc_results.append({"accession": acc, "status": "PARSE_ERROR", "reason": str(e)})
            continue

        cds_features = [f for f in record.features if f.type == "CDS"]
        n_cds = len(cds_features)

        # Check gene count — PhiX174-like genomes have 10-11 CDS
        if n_cds < 10:
            qc_results.append({"accession": acc, "status": "FAIL", "n_cds": n_cds,
                                "reason": f"Too few CDS ({n_cds}), expected 10-11"})
            continue

        # Extract gene names
        gene_names = set()
        for feat in cds_features:
            gene = get_gene_name(feat)
            if gene:
                gene_names.add(gene)

        # Check for duplicate coordinates
        coords = [(int(f.location.start), int(f.location.end), f.location.strand) for f in cds_features]
        has_dups = len(coords) != len(set(coords))

        # Check completeness
        missing = expected_genes - gene_names
        found_count = len(expected_genes & gene_names)

        status = "PASS"
        reason = ""
        if found_count < 8:
            status = "FAIL"
            reason = f"Missing core genes: {', '.join(sorted(missing))}"
            qc_results.append({
                "accession": acc, "status": status, "n_cds": n_cds,
                "genes_found": ", ".join(sorted(gene_names)),
                "missing_genes": ", ".join(sorted(missing)) if missing else "",
                "has_duplicates": has_dups,
                "seq_len": len(record.seq),
                "reason": reason,
            })
            continue
        if has_dups:
            reason += "; duplicate CDS coordinates"

        qc_results.append({
            "accession": acc, "status": status, "n_cds": n_cds,
            "genes_found": ", ".join(sorted(gene_names)),
            "missing_genes": ", ".join(sorted(missing)) if missing else "",
            "has_duplicates": has_dups,
            "seq_len": len(record.seq),
            "reason": reason,
        })

        if status in ("PASS", "WARN"):
            valid_accessions.add(acc)

    # Save QC report
    qc_path = OUT_DIR / "qc_report.csv"
    if qc_results:
        keys = list(qc_results[0].keys())
        all_keys = set()
        for r in qc_results:
            all_keys.update(r.keys())
        keys = sorted(all_keys)

        with open(qc_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(qc_results)
        log.info("QC report saved: %s", qc_path)

    n_pass = sum(1 for r in qc_results if r["status"] == "PASS")
    n_warn = sum(1 for r in qc_results if r["status"] == "WARN")
    n_fail = sum(1 for r in qc_results if r["status"] not in ("PASS", "WARN"))
    log.info("QC: %d PASS, %d WARN, %d FAIL out of %d", n_pass, n_warn, n_fail, len(qc_results))

    return valid_accessions


def get_gene_name(feature):
    """Extract gene letter name from a CDS feature."""
    # Check gene qualifier first — most reliable when present (e.g., /gene="A*")
    # Then fall back to standard_name, product
    for qual in ["gene", "standard_name", "product"]:
        val = feature.qualifiers.get(qual, [None])[0]
        if val:
            # Direct match in GENE_NAME_MAP
            if val in GENE_NAME_MAP:
                return GENE_NAME_MAP[val]
            # Try to match single-letter gene names (with optional asterisk)
            val_upper = val.strip().upper()
            if val_upper in {"A", "A*", "B", "C", "D", "E", "F", "G", "H", "J", "K"}:
                return val_upper
            # Match patterns like "gene A protein", "protein A*"
            # Use $ or \s instead of \b after the optional * since \b won't match
            # between * (non-word) and end-of-string/space correctly
            m = re.search(r'\b(gene\s+)?([A-K]\*?)(?:\s|$)', val, re.IGNORECASE)
            if m:
                g = m.group(2).upper()
                if g in {"A", "A*", "B", "C", "D", "E", "F", "G", "H", "J", "K"}:
                    return g
            # Match "gpX" or "gpX*" pattern
            m = re.match(r'gp([A-K]\*?)(?:\b|\s|$)', val, re.IGNORECASE)
            if m:
                g = m.group(1).upper()
                if g in {"A", "A*", "B", "C", "D", "E", "F", "G", "H", "J", "K"}:
                    return g
    return None


# ── Step 5: Pairwise alignment and mutation classification ─────────────────────

def build_cds_map(record):
    """Build a position-to-CDS mapping for a GenBank record.

    Returns:
        pos_to_cds: dict mapping 0-based position -> list of
                    {"gene": str, "cds_seq": str, "cds_positions": list[int],
                     "strand": int}
    """
    cds_entries = []

    for feat in record.features:
        if feat.type != "CDS":
            continue
        gene = get_gene_name(feat)
        if gene is None:
            continue

        strand = feat.location.strand if feat.location.strand else 1

        # Get all positions covered by this CDS (handles join locations)
        positions = []
        parts = feat.location.parts if hasattr(feat.location, "parts") else [feat.location]
        for part in parts:
            positions.extend(range(int(part.start), int(part.end)))

        # Extract CDS sequence
        cds_seq = str(feat.extract(record.seq))

        cds_entries.append({
            "gene": gene,
            "cds_seq": cds_seq,
            "cds_positions": positions,
            "strand": strand,
        })

    # Build position -> list of CDS entries
    pos_to_cds = defaultdict(list)
    for entry in cds_entries:
        for pos in entry["cds_positions"]:
            pos_to_cds[pos].append(entry)

    return dict(pos_to_cds)


def classify_mutations(query_seq, hit_seq, hit_record, pos_to_cds):
    """Classify mutations between aligned query and hit sequences.

    Returns dict: {gene_or_intergenic: {"synonymous": n, "nonsynonymous": n, "indel": n}}
    """
    counts = defaultdict(lambda: {"synonymous": 0, "nonsynonymous": 0, "indel": 0})

    # Map alignment columns to hit genome positions
    hit_pos = -1  # 0-based position in hit genome
    query_pos = -1

    aln_len = len(query_seq)

    # First pass: build column-to-position mappings
    col_to_hit_pos = []
    col_to_query_pos = []
    for col in range(aln_len):
        if hit_seq[col] != '-':
            hit_pos += 1
        if query_seq[col] != '-':
            query_pos += 1
        col_to_hit_pos.append(hit_pos if hit_seq[col] != '-' else None)
        col_to_query_pos.append(query_pos if query_seq[col] != '-' else None)

    # Track indels as contiguous gap runs
    in_gap = False
    gap_type = None  # 'query_gap' or 'hit_gap'
    gap_genes = set()

    for col in range(aln_len):
        q_base = query_seq[col]
        h_base = hit_seq[col]

        is_gap = (q_base == '-' or h_base == '-')

        if is_gap:
            if not in_gap:
                # Start of new gap run
                in_gap = True
                gap_type = 'query_gap' if q_base == '-' else 'hit_gap'
                gap_genes = set()

            # Determine which gene(s) this gap position falls in
            hp = col_to_hit_pos[col]
            if hp is not None and hp in pos_to_cds:
                for entry in pos_to_cds[hp]:
                    gap_genes.add(entry["gene"])
            elif hp is not None:
                gap_genes.add("intergenic")
            else:
                # Gap in hit — use surrounding context
                # Look at nearby hit positions to assign gene
                for offset in range(1, 20):
                    if col - offset >= 0 and col_to_hit_pos[col - offset] is not None:
                        hp_near = col_to_hit_pos[col - offset]
                        if hp_near in pos_to_cds:
                            for entry in pos_to_cds[hp_near]:
                                gap_genes.add(entry["gene"])
                        else:
                            gap_genes.add("intergenic")
                        break
                else:
                    gap_genes.add("intergenic")
        else:
            # End of gap run — record the indel
            if in_gap:
                if not gap_genes:
                    gap_genes.add("intergenic")
                for gene in gap_genes:
                    counts[gene]["indel"] += 1
                in_gap = False
                gap_genes = set()

            # Check for substitution
            if q_base != h_base:
                hp = col_to_hit_pos[col]
                if hp is not None and hp in pos_to_cds:
                    # Position is in one or more CDS
                    for entry in pos_to_cds[hp]:
                        mut_type = classify_substitution(
                            hp, q_base, h_base, entry, query_seq, hit_seq,
                            col, col_to_hit_pos, col_to_query_pos
                        )
                        counts[entry["gene"]][mut_type] += 1
                else:
                    counts["intergenic"]["nonsynonymous"] += 0  # don't count as nonsyn
                    counts["intergenic"]["indel"] += 0
                    # Intergenic substitutions tracked separately
                    counts["intergenic"]["synonymous"] += 0
                    # Use a dedicated counter for intergenic substitutions
                    counts["intergenic"].setdefault("substitution", 0)
                    counts["intergenic"]["substitution"] = counts["intergenic"].get("substitution", 0) + 1

    # Handle trailing gap
    if in_gap:
        if not gap_genes:
            gap_genes.add("intergenic")
        for gene in gap_genes:
            counts[gene]["indel"] += 1

    return dict(counts)


def classify_substitution(hit_pos, query_base, hit_base, cds_entry, query_aln, hit_aln,
                           col, col_to_hit_pos, col_to_query_pos):
    """Classify a single substitution as synonymous or nonsynonymous.

    Uses the CDS positions list to determine the codon containing hit_pos,
    then compares translated codons.
    """
    positions = cds_entry["cds_positions"]
    strand = cds_entry["strand"]

    # Find which position in the CDS this hit_pos corresponds to
    try:
        cds_idx = positions.index(hit_pos)
    except ValueError:
        return "nonsynonymous"  # shouldn't happen, but be conservative

    # Determine codon boundaries (0-based within CDS)
    codon_start_idx = (cds_idx // 3) * 3
    codon_end_idx = codon_start_idx + 3

    if codon_end_idx > len(positions):
        return "nonsynonymous"  # partial codon at end

    codon_positions = positions[codon_start_idx:codon_end_idx]

    # Extract the codon bases from both query and hit using alignment
    hit_codon = []
    query_codon = []

    for cpos in codon_positions:
        # Find alignment column for this hit position
        acol = None
        for c in range(len(col_to_hit_pos)):
            if col_to_hit_pos[c] == cpos:
                acol = c
                break

        if acol is None:
            return "nonsynonymous"  # can't resolve

        h = hit_aln[acol]
        q = query_aln[acol]

        if h == '-' or q == '-':
            return "nonsynonymous"  # gap within codon

        hit_codon.append(h)
        query_codon.append(q)

    if len(hit_codon) != 3 or len(query_codon) != 3:
        return "nonsynonymous"

    hit_codon_str = ''.join(hit_codon)
    query_codon_str = ''.join(query_codon)

    # Handle negative strand
    if strand == -1:
        hit_codon_str = str(Seq(hit_codon_str).reverse_complement())
        query_codon_str = str(Seq(query_codon_str).reverse_complement())

    # Translate
    try:
        hit_aa = str(Seq(hit_codon_str).translate())
        query_aa = str(Seq(query_codon_str).translate())
    except Exception:
        return "nonsynonymous"

    if hit_aa == query_aa:
        return "synonymous"
    else:
        return "nonsynonymous"


MAFFT_CACHE = CACHE_DIR / "mafft_alignments"


def run_pairwise_alignment(query_seq, hit_seq, query_id, hit_acc):
    """Run MAFFT pairwise alignment between query and hit sequences, with caching."""
    MAFFT_CACHE.mkdir(parents=True, exist_ok=True)

    # Cache key based on query_id and hit_acc
    safe_query = sanitize_filename(query_id)
    cache_path = MAFFT_CACHE / f"{safe_query}_vs_{hit_acc}.fasta"

    # Check cache
    if cache_path.exists():
        aligned = {}
        current_id = None
        current_seq = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        aligned[current_id] = ''.join(current_seq).upper()
                    current_id = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None:
                aligned[current_id] = ''.join(current_seq).upper()
        return aligned.get("query"), aligned.get("hit")

    # Run MAFFT
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp.write(f">query\n{query_seq}\n>hit\n{hit_seq}\n")
        tmp_path = tmp.name

    clean_path = clean_fasta_for_mafft(tmp_path)

    cmd = f"conda run -n sk_evo2_exp mafft --auto --thread 1 {clean_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if clean_path != tmp_path:
        Path(clean_path).unlink(missing_ok=True)

    if result.returncode != 0:
        log.error("MAFFT failed for %s vs %s: %s", query_id[:40], hit_acc, result.stderr[:200])
        return None, None

    # Parse aligned sequences from stdout
    aligned = {}
    current_id = None
    current_seq = []
    for line in result.stdout.strip().split('\n'):
        if line.startswith('>'):
            if current_id is not None:
                aligned[current_id] = ''.join(current_seq).upper()
            current_id = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id is not None:
        aligned[current_id] = ''.join(current_seq).upper()

    # Save to cache
    with open(cache_path, 'w') as f:
        for seq_id, seq in aligned.items():
            f.write(f">{seq_id}\n{seq}\n")

    return aligned.get("query"), aligned.get("hit")


def analyze_mutations_for_dataset(dataset_name, fasta_path, blast_hits, valid_accessions):
    """Analyze mutations for all query-hit pairs in a dataset."""
    records = {}
    for r in SeqIO.parse(fasta_path, "fasta"):
        seq = str(r.seq).replace('.', '').replace('-', '')
        seq = re.sub(r'[^ACGTacgt]', 'N', seq)
        records[r.description] = seq
    results = []

    query_ids = list(blast_hits.keys())
    total_pairs = sum(len(hits) for hits in blast_hits.values())
    log.info("Dataset '%s': %d queries, %d total pairs to analyze", dataset_name, len(query_ids), total_pairs)

    pair_count = 0
    for query_id in query_ids:
        hits = blast_hits[query_id]
        query_seq = records.get(query_id)
        if query_seq is None:
            log.warning("Query not found in FASTA: %s", query_id[:60])
            continue

        for rank, hit in enumerate(hits, 1):
            acc = hit["accession"]
            if acc not in valid_accessions:
                continue

            pair_count += 1
            gbk_path = GBK_CACHE / f"{acc}.gbk"
            try:
                hit_record = SeqIO.read(gbk_path, "genbank")
            except Exception as e:
                log.warning("Failed to parse %s: %s", acc, e)
                continue

            hit_seq = str(hit_record.seq)

            # Run pairwise alignment
            query_aln, hit_aln = run_pairwise_alignment(query_seq, hit_seq, query_id, acc)
            if query_aln is None:
                continue

            # Build CDS map for hit
            pos_to_cds = build_cds_map(hit_record)

            # Classify mutations
            mutation_counts = classify_mutations(query_aln, hit_seq=hit_aln,
                                                  hit_record=hit_record, pos_to_cds=pos_to_cds)

            # Record results per gene
            for gene in GENE_ORDER:
                if gene in mutation_counts:
                    c = mutation_counts[gene]
                    total = c.get("synonymous", 0) + c.get("nonsynonymous", 0) + c.get("indel", 0)
                    if gene == "intergenic":
                        total += c.get("substitution", 0)
                    results.append({
                        "query_id": query_id,
                        "dataset": dataset_name,
                        "hit_accession": acc,
                        "hit_rank": rank,
                        "gene": gene,
                        "synonymous": c.get("synonymous", 0),
                        "nonsynonymous": c.get("nonsynonymous", 0),
                        "indel": c.get("indel", 0),
                        "intergenic_substitution": c.get("substitution", 0) if gene == "intergenic" else 0,
                        "total_mutations": total,
                    })

            if pair_count % 50 == 0:
                log.info("  Processed %d/%d pairs", pair_count, total_pairs)

    log.info("  Done: %d result rows for dataset '%s'", len(results), dataset_name)
    return results


# ── Step 6: Aggregate results ──────────────────────────────────────────────────

def aggregate_results(all_results):
    """Create summary DataFrames from per-gene results."""
    # Per-gene results
    per_gene_path = OUT_DIR / "mutation_counts_per_gene.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(per_gene_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        log.info("Per-gene results: %s (%d rows)", per_gene_path, len(all_results))

    # Summary: one row per query-hit pair
    pair_summary = defaultdict(lambda: {
        "synonymous": 0, "nonsynonymous": 0, "indel": 0, "intergenic": 0, "total": 0
    })
    for row in all_results:
        key = (row["query_id"], row["dataset"], row["hit_accession"], row["hit_rank"])
        pair_summary[key]["synonymous"] += row["synonymous"]
        pair_summary[key]["nonsynonymous"] += row["nonsynonymous"]
        pair_summary[key]["indel"] += row["indel"]
        if row["gene"] == "intergenic":
            pair_summary[key]["intergenic"] += row.get("intergenic_substitution", 0)
        pair_summary[key]["total"] += row["total_mutations"]

    summary_path = OUT_DIR / "mutation_counts_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "dataset", "hit_accession", "hit_rank",
                          "synonymous", "nonsynonymous", "indel", "intergenic", "total_mutations"])
        for (qid, ds, acc, rank), counts in sorted(pair_summary.items()):
            writer.writerow([qid, ds, acc, rank,
                              counts["synonymous"], counts["nonsynonymous"],
                              counts["indel"], counts["intergenic"], counts["total"]])
    log.info("Summary results: %s (%d rows)", summary_path, len(pair_summary))

    return pair_summary


# ── Step 7: Test mode ──────────────────────────────────────────────────────────

def run_test():
    """Test mode: BLAST PhiX174 reference via NCBI API and print top 10 hits."""
    log.info("=== TEST MODE: BLASTing PhiX174 reference (NC_001422.1) via NCBI API ===")

    gbk_path = BASE_DIR / "experiments/20260227_phage_msa_sampling/gbks/NC_001422.1.gbk"
    if not gbk_path.exists():
        log.error("PhiX174 GBK not found at %s", gbk_path)
        return

    rec = SeqIO.read(gbk_path, "genbank")
    fasta_str = f">NC_001422.1 Enterobacteria phage phiX174\n{str(rec.seq)}\n"

    xml_path = OUT_DIR / "test_phix174_blast.xml"

    if xml_path.exists() and xml_path.stat().st_size > 0:
        log.info("Using cached test BLAST result: %s", xml_path)
    else:
        log.info("Submitting BLAST query to NCBI (this may take a few minutes)...")
        result_handle = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=fasta_str,
            megablast=True,
            hitlist_size=50,
            format_type="XML",
        )
        with open(xml_path, 'w') as f:
            f.write(result_handle.read())
        result_handle.close()
        log.info("BLAST complete, saved to %s", xml_path)

    # Create a dummy record list for the multi-query parser
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq as BSeq
    dummy_rec = SeqRecord(BSeq(""), id="NC_001422.1",
                          description="NC_001422.1 Enterobacteria phage phiX174")
    all_hits = parse_multi_blast_xml(xml_path, [dummy_rec])
    hits = all_hits.get(dummy_rec.description, [])

    print("\n" + "=" * 120)
    print("Top BLASTn hits for PhiX174 (NC_001422.1) against NCBI nt")
    print("Compare these with https://blast.ncbi.nlm.nih.gov/Blast.cgi")
    print("=" * 120)
    print(f"{'Rank':<5} {'Accession':<15} {'MaxScore':<10} {'TotScore':<10} {'E-value':<12} {'%ID':<7} "
          f"{'QCov%':<7} {'AlnLen':<8} {'SubjLen':<9} {'Gaps':<6} {'Mismatch':<9} {'Description'}")
    print("-" * 130)
    for i, hit in enumerate(hits, 1):
        print(f"{i:<5} {hit['accession']:<15} {hit['max_score']:<10.1f} "
              f"{hit['total_score']:<10.1f} "
              f"{hit['evalue']:<12.2e} {hit['pct_identity']:<7.2f} "
              f"{hit['query_cover']:<7.1f} {hit['align_length']:<8} "
              f"{hit['subject_length']:<9} {hit['gaps']:<6} "
              f"{hit['mismatches']:<9} {hit['hit_def'][:50]}")
    print("=" * 130)

    # Save as CSV
    csv_path = OUT_DIR / "test_phix174_top_hits.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Accession", "Max Score", "Total Score", "E-value",
                          "Percent Identity", "Query Cover %", "Alignment Length",
                          "Subject Length", "Gaps", "Mismatches", "Description"])
        for i, hit in enumerate(hits, 1):
            writer.writerow([i, hit["accession"], hit["max_score"], hit["total_score"],
                              hit["evalue"], f"{hit['pct_identity']:.2f}",
                              f"{hit['query_cover']:.1f}", hit["align_length"],
                              hit["subject_length"], hit["gaps"], hit["mismatches"],
                              hit["hit_def"]])
    log.info("Test hits saved to %s", csv_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mutation type analysis via BLASTn")
    parser.add_argument("--test", action="store_true", help="Run test mode with PhiX174 reference")
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    log.info("Mutation type analysis: BLASTn hits vs query genomes")
    log.info("Output directory: %s", OUT_DIR)

    # Create cache directories
    BLAST_CACHE.mkdir(parents=True, exist_ok=True)
    GBK_CACHE.mkdir(parents=True, exist_ok=True)

    # Step 2: Run BLASTn for all datasets
    log.info("=" * 60)
    log.info("Step 2: Running BLASTn against local NCBI nt")
    all_blast_hits = {}
    for ds_name, fasta_path in DATASETS.items():
        all_blast_hits[ds_name] = run_blast_for_dataset(ds_name, fasta_path)

    # Step 3: Fetch GenBank records
    log.info("=" * 60)
    log.info("Step 3: Fetching GenBank records")
    all_accessions = fetch_genbank_records(all_blast_hits)

    # Step 4: QC GenBank annotations
    log.info("=" * 60)
    log.info("Step 4: QC GenBank annotations")
    valid_accessions = qc_genbank_records(all_accessions)

    # Step 5: Pairwise alignment and mutation classification
    log.info("=" * 60)
    log.info("Step 5: Pairwise alignment and mutation classification")
    all_results = []
    for ds_name, fasta_path in DATASETS.items():
        ds_results = analyze_mutations_for_dataset(
            ds_name, fasta_path, all_blast_hits[ds_name], valid_accessions
        )
        all_results.extend(ds_results)

    # Step 6: Aggregate results
    log.info("=" * 60)
    log.info("Step 6: Aggregating results")
    pair_summary = aggregate_results(all_results)

    log.info("=" * 60)
    log.info("Analysis complete!")


if __name__ == "__main__":
    main()
