"""
Phage competition sequencing analysis.

Stages:
  1. QC + Alignment: fastp filtering -> minimap2 alignment (primary only)
  2. SNV-based Read Assignment: score reads at variable positions, windowed chimera detection
  3. Fold Change Calculations: cumulative log2 proportion FC, signed AUC, T0->Tfinal comparison
  4. Visualization: line plots, facet plots, AUC comparison, read assignment diagnostics

Supports two scoring modes:
  - "direct": all references same length, variable positions by direct column comparison (used for natural phage competition)
  - "msa": variable-length references, builds MSA via pairwise alignment to anchor (used for generated Evo phage competition)

Usage:
    conda run -n read_mapping3 python competition_analysis.py [natural|evo|both] [--stages 1234] [--skip-existing]

Examples:
    python competition_analysis.py natural             # run all stages on natural phage data
    python competition_analysis.py evo --stages 34     # only fold change + plots for evo data
    python competition_analysis.py both                # run both datasets
"""

import os
import csv
import argparse
import subprocess
import numpy as np
import pandas as pd
import pysam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from collections import defaultdict


# ═══════════════════════════════════════════════════════
# Stage 1: QC + Alignment
# ═══════════════════════════════════════════════════════

def run_fastp(input_file, output_file, report_file, min_length=250,
              quality_threshold=20, unqualified_limit=30):
    """Filter reads with fastp."""
    print(f"  [fastp] {os.path.basename(input_file)}")
    subprocess.run([
        "fastp", "-i", input_file, "-o", output_file,
        "-l", str(min_length), "-q", str(quality_threshold),
        "-u", str(unqualified_limit), "--html", report_file
    ], check=True)


def align_reads(filtered_fastq, reference_fasta, output_sam, threads=8):
    """Align reads using minimap2 with primary alignments only."""
    print(f"  [minimap2] {os.path.basename(filtered_fastq)}")
    subprocess.run([
        "minimap2", "-ax", "map-ont", "--secondary=no", "-t", str(threads),
        reference_fasta, filtered_fastq, "-o", output_sam
    ], check=True)


def run_stage1(config, skip_existing=False):
    """QC + alignment for all samples."""
    print(f"\n{'='*60}")
    print(f"  STAGE 1: QC + Alignment — {config['name']}")
    print(f"{'='*60}")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for sample_name, fastq_file in config["samples"]:
        input_fastq = os.path.join(config["raw_fastq_dir"], fastq_file)
        filtered_fastq = os.path.join(output_dir, f"{sample_name}_filtered.fastq")
        fastp_report = os.path.join(output_dir, f"{sample_name}_fastp.html")
        output_sam = os.path.join(output_dir, f"{sample_name}.sam")

        if skip_existing and os.path.exists(output_sam):
            print(f"  [skip] {sample_name} — SAM exists")
            continue

        if not os.path.exists(input_fastq):
            print(f"  [warn] {sample_name} — FASTQ not found: {input_fastq}")
            continue

        print(f"\n  === {sample_name} ===")
        run_fastp(input_fastq, filtered_fastq, fastp_report)
        align_reads(filtered_fastq, config["reference_fasta"], output_sam)

    print(f"\n  Stage 1 complete. Output: {output_dir}")


# ═══════════════════════════════════════════════════════
# Stage 2: SNV-based Read Assignment + Chimera Detection
# ═══════════════════════════════════════════════════════

N_WINDOWS = 5
MIN_VAR_PER_WINDOW = 5
MIN_WINDOW_MARGIN = 3
MIN_DISAGREEING_WINDOWS = 1


def load_references(fasta_path):
    """Load reference sequences from FASTA."""
    refs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        refs[rec.id] = str(rec.seq).upper()
    return refs


# --- Direct mode (equal-length references) ---

def find_variable_positions(refs):
    """Find positions where not all references have the same base.
    All references must be the same length (direct mode)."""
    seq_len = len(next(iter(refs.values())))
    names = list(refs.keys())
    assert all(len(refs[n]) == seq_len for n in names), \
        f"Direct mode requires equal-length references, got {set(len(refs[n]) for n in names)}"
    var_positions = []
    for pos in range(seq_len):
        bases = set(refs[name][pos] for name in names)
        if len(bases) > 1:
            var_positions.append(pos)
    return var_positions


def build_expected_bases(refs, var_positions):
    """Build lookup: {position: {accession: expected_base}}."""
    expected = {}
    for pos in var_positions:
        expected[pos] = {name: refs[name][pos] for name in refs}
    return expected


def build_varpos_window_map(var_positions, n_windows):
    """Map each variable position to a window index (equal count per window).

    Instead of splitting by genomic coordinate (which gives unequal
    statistical power when variable positions are clustered), we split
    by variable-position rank so each window has ~equal informative sites.

    Returns dict {position: window_index}.
    """
    n = len(var_positions)
    pos_to_window = {}
    for i, pos in enumerate(var_positions):
        pos_to_window[pos] = min(i * n_windows // n, n_windows - 1)
    return pos_to_window


def assign_read_by_snvs(read, var_positions, expected, ref_names,
                        varpos_window_map=None, n_windows=N_WINDOWS):
    """
    Score a read against all phages at variable positions (direct mode).
    Returns (best_phage, margin, n_informative, is_chimera, disagreeing_windows).
    """
    aligned_pairs = read.get_aligned_pairs(matches_only=False)
    ref_to_query = {}
    for qpos, rpos in aligned_pairs:
        if rpos is not None and qpos is not None:
            ref_to_query[rpos] = qpos

    query_seq = read.query_sequence
    if query_seq is None:
        return None, 0, 0, False

    scores = {name: 0 for name in ref_names}
    n_informative = 0

    if varpos_window_map is not None:
        window_scores = [{name: 0 for name in ref_names} for _ in range(n_windows)]
        window_n_info = [0] * n_windows

    for pos in var_positions:
        if pos not in ref_to_query:
            continue
        qpos = ref_to_query[pos]
        read_base = query_seq[qpos].upper()
        n_informative += 1

        for name in ref_names:
            if expected[pos][name] == read_base:
                scores[name] += 1

        if varpos_window_map is not None:
            wi = varpos_window_map[pos]
            window_n_info[wi] += 1
            for name in ref_names:
                if expected[pos][name] == read_base:
                    window_scores[wi][name] += 1

    if n_informative == 0:
        return None, 0, 0, False

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1]
    margin = best_score - second_score

    is_chimera = False
    disagreeing_windows = []
    if varpos_window_map is not None and margin >= 1:
        for wi in range(n_windows):
            if window_n_info[wi] < MIN_VAR_PER_WINDOW:
                continue
            ws = window_scores[wi]
            w_sorted = sorted(ws.items(), key=lambda x: -x[1])
            w_best = w_sorted[0][0]
            w_margin = w_sorted[0][1] - w_sorted[1][1]
            if w_best != best_name and w_margin >= MIN_WINDOW_MARGIN:
                disagreeing_windows.append(wi)
        if len(disagreeing_windows) >= MIN_DISAGREEING_WINDOWS:
            is_chimera = True

    return best_name, margin, n_informative, is_chimera, disagreeing_windows


def snv_based_counts_direct(sam_path, refs, var_positions, expected,
                            min_pid=0.95, min_align_frac=0.90):
    """Count reads using direct SNV-based assignment with windowed chimera detection."""
    ref_names = list(refs.keys())
    ref_lengths = {name: len(seq) for name, seq in refs.items()}
    varpos_window_map = build_varpos_window_map(var_positions, N_WINDOWS)

    aln = pysam.AlignmentFile(sam_path, "r")
    counts = defaultdict(int)
    stats = {"total_primary": 0, "pass_filters": 0, "assigned": 0,
             "ambiguous": 0, "no_informative": 0, "reassigned": 0, "chimera": 0}
    chimera_window_counts = [0] * N_WINDOWS
    chimera_n_disagree = defaultdict(int)

    for r in aln:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        stats["total_primary"] += 1

        ref_name = r.reference_name
        if ref_name not in ref_lengths:
            continue

        exp_len = ref_lengths[ref_name]
        qlen = r.query_alignment_length or 0
        if qlen < min_align_frac * exp_len:
            continue

        nm = r.get_tag("NM") if r.has_tag("NM") else 0
        pid = 1.0 - (nm / max(1, qlen))
        if pid < min_pid:
            continue

        stats["pass_filters"] += 1

        best_phage, margin, _, is_chimera, disagree_wins = assign_read_by_snvs(
            r, var_positions, expected, ref_names, varpos_window_map
        )

        if best_phage is None:
            stats["no_informative"] += 1
            continue
        if margin < 1:
            stats["ambiguous"] += 1
            continue
        if is_chimera:
            stats["chimera"] += 1
            chimera_n_disagree[len(disagree_wins)] += 1
            for wi in disagree_wins:
                chimera_window_counts[wi] += 1
            continue

        stats["assigned"] += 1
        if best_phage != ref_name:
            stats["reassigned"] += 1
        counts[best_phage] += 1

    aln.close()
    chimera_window_stats = {
        "window_counts": chimera_window_counts,
        "n_disagree_dist": dict(chimera_n_disagree),
    }
    return dict(counts), stats, chimera_window_stats


# --- MSA mode (variable-length references) ---

def build_msa_via_pairwise(refs):
    """
    Build a simple MSA by aligning all references to an anchor (longest) sequence.
    Returns (msa_cols, ref_to_msa).
    """
    names = list(refs.keys())
    anchor = max(names, key=lambda n: len(refs[n]))
    anchor_seq = refs[anchor]
    print(f"  Anchor reference: {anchor} ({len(anchor_seq)} bp)")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    ref_to_anchor = {anchor: {i: i for i in range(len(anchor_seq))}}

    for name in names:
        if name == anchor:
            continue
        seq = refs[name]
        alignments = aligner.align(anchor_seq, seq)
        aln = alignments[0]

        mapping = {}
        for a_indices, q_indices in zip(aln.aligned[0], aln.aligned[1]):
            a_start, a_end = a_indices
            q_start, _ = q_indices
            length = a_end - a_start
            for i in range(length):
                mapping[q_start + i] = a_start + i
        ref_to_anchor[name] = mapping

    max_anchor_pos = len(anchor_seq) - 1
    anchor_to_ref = {}
    for name in names:
        rev = {}
        for ref_pos, anc_pos in ref_to_anchor[name].items():
            rev[anc_pos] = ref_pos
        anchor_to_ref[name] = rev

    msa_cols = []
    ref_to_msa = {name: {} for name in names}

    for anc_pos in range(max_anchor_pos + 1):
        col = {}
        for name in names:
            if anc_pos in anchor_to_ref[name]:
                ref_pos = anchor_to_ref[name][anc_pos]
                col[name] = (ref_pos, refs[name][ref_pos])
            else:
                col[name] = None
        msa_cols.append(col)

        msa_idx = len(msa_cols) - 1
        for name in names:
            if col[name] is not None:
                ref_pos = col[name][0]
                ref_to_msa[name][ref_pos] = msa_idx

    return msa_cols, ref_to_msa


def find_variable_msa_columns(msa_cols, ref_names):
    """Find MSA columns where not all references have the same base (ignoring gaps)."""
    var_cols = []
    for idx, col in enumerate(msa_cols):
        bases = set()
        n_present = 0
        for name in ref_names:
            if col[name] is not None:
                bases.add(col[name][1])
                n_present += 1
        if n_present >= 2 and len(bases) > 1:
            var_cols.append(idx)
    return var_cols


def build_expected_bases_msa(msa_cols, var_cols, ref_names):
    """Build lookup: {msa_col_idx: {ref_name: expected_base or None}}."""
    expected = {}
    for col_idx in var_cols:
        col = msa_cols[col_idx]
        expected[col_idx] = {}
        for name in ref_names:
            if col[name] is not None:
                expected[col_idx][name] = col[name][1]
            else:
                expected[col_idx][name] = None
    return expected


def assign_read_by_snvs_msa(read, ref_to_msa_mapping, var_cols, expected,
                            ref_names, varcol_window_map=None, n_windows=N_WINDOWS):
    """Score a read against all phages at variable MSA positions."""
    aligned_pairs = read.get_aligned_pairs(matches_only=False)
    ref_to_query = {}
    for qpos, rpos in aligned_pairs:
        if rpos is not None and qpos is not None:
            ref_to_query[rpos] = qpos

    query_seq = read.query_sequence
    if query_seq is None:
        return None, 0, 0, False, []

    mapped_ref = read.reference_name
    pos_mapping = ref_to_msa_mapping.get(mapped_ref, {})

    scores = {name: 0 for name in ref_names}
    n_informative = 0

    if varcol_window_map is not None:
        window_scores = [{name: 0 for name in ref_names} for _ in range(n_windows)]
        window_n_info = [0] * n_windows

    for ref_pos, qpos in ref_to_query.items():
        msa_col = pos_mapping.get(ref_pos)
        if msa_col is None or msa_col not in expected:
            continue

        read_base = query_seq[qpos].upper()
        n_informative += 1

        for name in ref_names:
            exp_base = expected[msa_col][name]
            if exp_base is not None and exp_base == read_base:
                scores[name] += 1

        if varcol_window_map is not None:
            wi = varcol_window_map[msa_col]
            window_n_info[wi] += 1
            for name in ref_names:
                exp_base = expected[msa_col][name]
                if exp_base is not None and exp_base == read_base:
                    window_scores[wi][name] += 1

    if n_informative == 0:
        return None, 0, 0, False, []

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best_name, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1]
    margin = best_score - second_score

    is_chimera = False
    disagreeing_windows = []
    if varcol_window_map is not None and margin >= 1:
        for wi in range(n_windows):
            if window_n_info[wi] < MIN_VAR_PER_WINDOW:
                continue
            ws = window_scores[wi]
            w_sorted = sorted(ws.items(), key=lambda x: -x[1])
            w_best = w_sorted[0][0]
            w_margin = w_sorted[0][1] - w_sorted[1][1]
            if w_best != best_name and w_margin >= MIN_WINDOW_MARGIN:
                disagreeing_windows.append(wi)
        if len(disagreeing_windows) >= MIN_DISAGREEING_WINDOWS:
            is_chimera = True

    return best_name, margin, n_informative, is_chimera, disagreeing_windows


def snv_based_counts_msa(sam_path, refs, ref_to_msa, var_cols, expected,
                         varcol_window_map, min_pid=0.90, min_align_frac=0.90):
    """Count reads using MSA-based SNV assignment with windowed chimera detection."""
    ref_names = list(refs.keys())
    ref_lengths = {name: len(seq) for name, seq in refs.items()}

    aln = pysam.AlignmentFile(sam_path, "r")
    counts = defaultdict(int)
    stats = {"total_primary": 0, "pass_filters": 0, "assigned": 0,
             "ambiguous": 0, "no_informative": 0, "reassigned": 0, "chimera": 0}
    chimera_window_counts = [0] * N_WINDOWS
    chimera_n_disagree = defaultdict(int)

    for r in aln:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        stats["total_primary"] += 1

        ref_name = r.reference_name
        if ref_name not in ref_lengths:
            continue

        exp_len = ref_lengths[ref_name]
        qlen = r.query_alignment_length or 0
        if qlen < min_align_frac * exp_len:
            continue

        nm = r.get_tag("NM") if r.has_tag("NM") else 0
        pid = 1.0 - (nm / max(1, qlen))
        if pid < min_pid:
            continue

        stats["pass_filters"] += 1

        best_phage, margin, _, is_chimera, disagree_wins = assign_read_by_snvs_msa(
            r, ref_to_msa, var_cols, expected, ref_names, varcol_window_map
        )

        if best_phage is None:
            stats["no_informative"] += 1
            continue
        if margin < 1:
            stats["ambiguous"] += 1
            continue
        if is_chimera:
            stats["chimera"] += 1
            chimera_n_disagree[len(disagree_wins)] += 1
            for wi in disagree_wins:
                chimera_window_counts[wi] += 1
            continue

        stats["assigned"] += 1
        if best_phage != ref_name:
            stats["reassigned"] += 1
        counts[best_phage] += 1

    aln.close()
    chimera_window_stats = {
        "window_counts": chimera_window_counts,
        "n_disagree_dist": dict(chimera_n_disagree),
    }
    return dict(counts), stats, chimera_window_stats


def write_counts_csv(counts, out_csv):
    """Write counts with proportions."""
    total = sum(counts.values())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Sequence", "Read Count", "Proportion"])
        w.writeheader()
        for ref, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            proportion = (c / total) if total else 0.0
            w.writerow({"Sequence": ref, "Read Count": c, "Proportion": proportion})


def run_stage2(config, skip_existing=False):
    """SNV-based read assignment + chimera detection."""
    print(f"\n{'='*60}")
    print(f"  STAGE 2: SNV-based Read Assignment — {config['name']}")
    print(f"{'='*60}")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    fasta_path = config["reference_fasta"]
    scoring_mode = config["scoring_mode"]
    min_pid = config["min_pid"]
    min_align_frac = config["min_align_frac"]

    # Determine SAM directory: use sam_dir if provided (for pre-aligned data),
    # otherwise use output_dir (where stage 1 writes SAMs)
    sam_dir = config.get("sam_dir", output_dir)

    print(f"  Loading references from {os.path.basename(fasta_path)}...")
    refs = load_references(fasta_path)
    ref_names = list(refs.keys())

    if scoring_mode == "direct":
        var_positions = find_variable_positions(refs)
        expected = build_expected_bases(refs, var_positions)
        print(f"  {len(refs)} references, {len(var_positions)} variable positions (direct mode)")
    elif scoring_mode == "msa":
        print(f"  Building MSA via pairwise alignments...")
        msa_cols, ref_to_msa = build_msa_via_pairwise(refs)
        var_cols = find_variable_msa_columns(msa_cols, ref_names)
        expected_msa = build_expected_bases_msa(msa_cols, var_cols, ref_names)
        varcol_window_map = build_varpos_window_map(var_cols, N_WINDOWS)
        print(f"  {len(refs)} references, {len(msa_cols)} MSA columns, {len(var_cols)} variable (MSA mode)")
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    samples = [s[0] for s in config["samples"]]
    all_counts = []
    all_sample_stats = []  # per-sample stats for read fate plot
    total_stats = defaultdict(int)
    total_chimera_window_counts = [0] * N_WINDOWS
    total_chimera_n_disagree = defaultdict(int)

    for sample in samples:
        # Check for SAM in sam_dir first, then output_dir
        sam_path = os.path.join(sam_dir, f"{sample}.sam")
        if not os.path.exists(sam_path):
            sam_path = os.path.join(output_dir, f"{sample}.sam")

        # If still no SAM, try re-aligning from filtered FASTQ
        if not os.path.exists(sam_path):
            filtered_fastq = os.path.join(sam_dir, f"{sample}_filtered.fastq")
            if not os.path.exists(filtered_fastq):
                filtered_fastq = os.path.join(output_dir, f"{sample}_filtered.fastq")
            if os.path.exists(filtered_fastq):
                sam_path = os.path.join(output_dir, f"{sample}.sam")
                align_reads(filtered_fastq, fasta_path, sam_path)
            else:
                print(f"  [skip] {sample} — no SAM or filtered FASTQ found")
                continue

        counts_csv = os.path.join(output_dir, f"{sample}_counts_primary.csv")
        if skip_existing and os.path.exists(counts_csv):
            print(f"  [skip] {sample} — counts CSV exists")
            df = pd.read_csv(counts_csv)
            df["Sample"] = sample
            all_counts.append(df)
            continue

        if scoring_mode == "direct":
            counts, sample_stats, cw_stats = snv_based_counts_direct(
                sam_path, refs, var_positions, expected,
                min_pid=min_pid, min_align_frac=min_align_frac
            )
        else:
            counts, sample_stats, cw_stats = snv_based_counts_msa(
                sam_path, refs, ref_to_msa, var_cols, expected_msa,
                varcol_window_map, min_pid=min_pid, min_align_frac=min_align_frac
            )

        for wi in range(N_WINDOWS):
            total_chimera_window_counts[wi] += cw_stats["window_counts"][wi]
        for n, cnt in cw_stats["n_disagree_dist"].items():
            total_chimera_n_disagree[n] += cnt

        write_counts_csv(counts, counts_csv)

        pf = sample_stats["pass_filters"]
        print(f"  {sample}: {pf} pass filters -> "
              f"{sample_stats['assigned']} assigned, {sample_stats['ambiguous']} ambiguous, "
              f"{sample_stats['chimera']} chimera, {sample_stats['reassigned']} reassigned")

        all_sample_stats.append({"Sample": sample, **sample_stats})
        for k, v in sample_stats.items():
            total_stats[k] += v

        df = pd.read_csv(counts_csv)
        df["Sample"] = sample
        all_counts.append(df)

    # Print summary stats
    pf = total_stats.get("pass_filters", 0)
    if pf > 0:
        print(f"\n  {'─'*50}")
        print(f"  OVERALL: {total_stats['total_primary']} primary alignments, "
              f"{pf} pass filters")
        print(f"  Assigned: {total_stats['assigned']} ({total_stats['assigned']/pf*100:.1f}%), "
              f"Ambiguous: {total_stats['ambiguous']} ({total_stats['ambiguous']/pf*100:.1f}%), "
              f"Chimera: {total_stats['chimera']} ({total_stats['chimera']/pf*100:.1f}%)")
        if total_stats["assigned"] > 0:
            print(f"  Reassigned: {total_stats['reassigned']} ({total_stats['reassigned']/total_stats['assigned']*100:.1f}%)")

    # Save merged counts
    if all_counts:
        merged_df = pd.concat(all_counts, ignore_index=True)
        merged_csv = os.path.join(output_dir, "all_timepoints_read_counts_primary.csv")
        merged_df.to_csv(merged_csv, index=False)
        print(f"\n  Merged CSV: {merged_csv}")
        print(f"  Total assigned reads: {merged_df['Read Count'].sum():.0f}")
        print(f"  Unique sequences: {merged_df['Sequence'].nunique()}")

    # Save per-sample read fate stats
    if all_sample_stats:
        stats_df = pd.DataFrame(all_sample_stats)
        stats_csv = os.path.join(output_dir, "read_fate_stats.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"  Read fate stats: {stats_csv}")

    # Save chimera window stats
    if total_stats.get("chimera", 0) > 0:
        cw_rows = []
        for wi in range(N_WINDOWS):
            cw_rows.append({"window": wi, "chimera_disagree_count": total_chimera_window_counts[wi]})
        cw_df = pd.DataFrame(cw_rows)
        cw_csv = os.path.join(output_dir, "chimera_window_stats.csv")
        cw_df.to_csv(cw_csv, index=False)

        nd_rows = []
        for n in sorted(total_chimera_n_disagree.keys()):
            nd_rows.append({"n_disagreeing_windows": n, "chimera_count": total_chimera_n_disagree[n]})
        nd_df = pd.DataFrame(nd_rows)
        nd_csv = os.path.join(output_dir, "chimera_ndisagree_stats.csv")
        nd_df.to_csv(nd_csv, index=False)
        print(f"  Chimera window stats: {cw_csv}")


# ═══════════════════════════════════════════════════════
# Stage 3: Fold Change Calculations
# ═══════════════════════════════════════════════════════

def parse_sample_info(sample_name):
    """Extract timepoint and replicate from 'T{n}_rep{m}' format."""
    parts = sample_name.split("_")
    tp = int(parts[0][1:])
    rep = int(parts[1][3:])
    return tp, rep


def calculate_cumulative_fc_per_replicate(df, timepoint_to_hours):
    """Calculate cumulative log2 proportion FC from earliest detected timepoint."""
    df = df.copy()
    df["Timepoint"] = df["Sample"].apply(lambda x: parse_sample_info(x)[0])
    df["Replicate"] = df["Sample"].apply(lambda x: parse_sample_info(x)[1])
    df["Hours"] = df["Timepoint"].apply(lambda t: timepoint_to_hours.get(t, float(t)))

    out_rows = []
    for (seq, rep), sub in df.groupby(["Sequence", "Replicate"]):
        sub = sub.sort_values("Timepoint")
        baseline_rows = sub[sub["Proportion"] > 0]
        baseline_prop = baseline_rows["Proportion"].iloc[0] if len(baseline_rows) > 0 else 0.0
        baseline_tp = baseline_rows["Timepoint"].iloc[0] if len(baseline_rows) > 0 else 0

        prev_prop = None
        for _, row in sub.iterrows():
            tp = row["Timepoint"]
            hours = row["Hours"]
            curr_prop = row["Proportion"]

            if prev_prop is not None and prev_prop > 0 and curr_prop > 0:
                fold_change = np.log2(curr_prop / prev_prop)
            else:
                fold_change = 0.0

            if tp <= baseline_tp:
                cumulative_fc = 0.0
            elif baseline_prop > 0 and curr_prop > 0:
                cumulative_fc = np.log2(curr_prop / baseline_prop)
            else:
                cumulative_fc = 0.0

            prev_prop = curr_prop
            out_rows.append({
                "Sequence": seq, "Replicate": rep, "Timepoint": tp,
                "Hours": hours, "Read_Count": row["Read Count"],
                "Proportion": curr_prop, "Fold_Change": fold_change,
                "Cumulative_Fold_Change": cumulative_fc,
            })
    return pd.DataFrame(out_rows)


def compute_signed_auc(cum_df):
    """Compute signed AUC of cumulative log2 FC for each sequence-replicate."""
    auc_rows = []
    for (seq, rep), sub in cum_df.groupby(["Sequence", "Replicate"]):
        sub = sub[["Hours", "Cumulative_Fold_Change"]].dropna().sort_values("Hours")
        if len(sub) < 2:
            auc_val = np.nan
        else:
            auc_val = np.trapezoid(y=sub["Cumulative_Fold_Change"].to_numpy(),
                                   x=sub["Hours"].to_numpy())
        auc_rows.append({"Sequence": seq, "Replicate": rep, "SignedAUC": auc_val})
    return pd.DataFrame(auc_rows)


def compute_t0_to_tfinal_fold_changes(df, max_timepoint):
    """Compute T0->Tfinal fold changes for both raw counts and proportions."""
    df = df.copy()
    df["Timepoint"] = df["Sample"].apply(lambda x: int(x.split("_")[0][1:]))
    df["Replicate"] = df["Sample"].apply(lambda x: int(x.split("_")[1][3:]))
    t0 = df[df["Timepoint"] == 0]
    tf = df[df["Timepoint"] == max_timepoint]
    rows = []
    for seq in df["Sequence"].unique():
        for rep in df["Replicate"].unique():
            r0 = t0[(t0["Sequence"] == seq) & (t0["Replicate"] == rep)]
            rf = tf[(tf["Sequence"] == seq) & (tf["Replicate"] == rep)]
            if len(r0) == 0 or len(rf) == 0:
                continue
            c0, cf = r0["Read Count"].iloc[0], rf["Read Count"].iloc[0]
            p0, pf = r0["Proportion"].iloc[0], rf["Proportion"].iloc[0]
            count_fc = np.log2(cf / c0) if c0 > 0 and cf > 0 else np.nan
            prop_fc = np.log2(pf / p0) if p0 > 0 and pf > 0 else np.nan
            rows.append({"Sequence": seq, "Replicate": rep,
                         "Count_log2FC": count_fc, "Prop_log2FC": prop_fc})
    return pd.DataFrame(rows)


def run_stage3(config):
    """Fold change calculations."""
    print(f"\n{'='*60}")
    print(f"  STAGE 3: Fold Change Calculations — {config['name']}")
    print(f"{'='*60}")

    output_dir = config["output_dir"]
    merged_csv = os.path.join(output_dir, "all_timepoints_read_counts_primary.csv")
    if not os.path.exists(merged_csv):
        print(f"  [error] Merged CSV not found: {merged_csv}")
        print(f"  Run stage 2 first.")
        return

    df = pd.read_csv(merged_csv)
    name_map = config.get("accession_to_name", {})
    if name_map:
        df["Sequence"] = df["Sequence"].map(lambda x: name_map.get(x, x))
    exclude = config.get("exclude_phages", set())
    if exclude:
        df = df[~df["Sequence"].isin(exclude)]

    max_tp = config["max_timepoint"]
    timepoint_to_hours = config["timepoint_to_hours"]

    print(f"  Loaded {len(df)} rows, {df['Sequence'].nunique()} phages")

    # Cumulative proportion FC
    print(f"  [1] Cumulative proportion fold changes...")
    cum_df = calculate_cumulative_fc_per_replicate(df, timepoint_to_hours)
    cum_df = cum_df[cum_df["Timepoint"] <= max_tp].copy()
    cum_csv = os.path.join(output_dir, "cumulative_proportion_fold_changes.csv")
    cum_df.to_csv(cum_csv, index=False)
    print(f"      Saved: {cum_csv}")

    # Signed AUC
    print(f"  [2] Signed AUC...")
    auc_df = compute_signed_auc(cum_df)
    auc_csv = os.path.join(output_dir, "signed_auc_cumulative_log2FC.csv")
    auc_df.to_csv(auc_csv, index=False)
    print(f"      Saved: {auc_csv}")

    # T0->Tfinal FC comparison
    print(f"  [3] T0->T{max_tp} fold change comparison...")
    fc_df = compute_t0_to_tfinal_fold_changes(df, max_tp)
    fc_csv = os.path.join(output_dir, "fold_change_count_vs_proportion.csv")
    fc_df.to_csv(fc_csv, index=False)
    print(f"      Saved: {fc_csv}")

    # Summary proportions
    print(f"\n  T0 mean proportions:")
    t0 = cum_df[cum_df["Timepoint"] == 0].groupby("Sequence")["Proportion"].mean().sort_values(ascending=False)
    for seq, prop in t0.items():
        print(f"    {seq}: {100*prop:.1f}%")
    print(f"\n  T{max_tp} mean proportions:")
    tf = cum_df[cum_df["Timepoint"] == max_tp].groupby("Sequence")["Proportion"].mean().sort_values(ascending=False)
    for seq, prop in tf.items():
        print(f"    {seq}: {100*prop:.1f}%")


# ═══════════════════════════════════════════════════════
# Stage 4: Visualization
# ═══════════════════════════════════════════════════════

def _load_cum_df(config):
    """Load cumulative FC data for plotting."""
    output_dir = config["output_dir"]
    cum_csv = os.path.join(output_dir, "cumulative_proportion_fold_changes.csv")
    if not os.path.exists(cum_csv):
        print(f"  [error] {cum_csv} not found. Run stage 3 first.")
        return None
    return pd.read_csv(cum_csv)


def plot_log2fc_lineplot(df, output_dir):
    """Line plot of log2 FC from T0 (mean +/- SD across replicates)."""
    summary = (df.groupby(["Sequence", "Hours"])["Cumulative_Fold_Change"]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["Sequence", "Hours", "mean_fc", "std_fc"]
    summary["std_fc"] = summary["std_fc"].fillna(0)

    final_hour = summary["Hours"].max()
    final_order = (summary[summary["Hours"] == final_hour]
                   .sort_values("mean_fc", ascending=False)["Sequence"].tolist())
    all_seqs = sorted(df["Sequence"].unique())
    final_order += [s for s in all_seqs if s not in final_order]

    palette = sns.color_palette("tab20", len(final_order))
    color_map = {seq: palette[i] for i, seq in enumerate(final_order)}

    fig, ax = plt.subplots(figsize=(12, 7))
    for seq in final_order:
        seq_df = summary[summary["Sequence"] == seq].sort_values("Hours")
        if seq_df.empty:
            continue
        real = seq_df[seq_df["mean_fc"].notna()]
        ax.plot(real["Hours"], real["mean_fc"],
                marker="o", linewidth=2, markersize=4,
                color=color_map[seq], label=seq)
        ax.fill_between(real["Hours"],
                        real["mean_fc"] - real["std_fc"],
                        real["mean_fc"] + real["std_fc"],
                        color=color_map[seq], alpha=0.15)

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("log2 FC from T0", fontsize=12)
    ax.set_title("log2 Proportion Fold Change from T0 (Mean +/- SD across Replicates)", fontsize=13)
    ax.set_xticks(sorted(summary["Hours"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", title="Phage")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "log2fc_lineplot.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.savefig(out_path.replace(".svg", ".png"), format="png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_log2fc_facet(df, output_dir):
    """Small multiples: one panel per phage, mean +/- SD, blue winners / pink losers."""
    last_fc = (df.groupby("Sequence")
               .apply(lambda g: g.loc[g["Hours"] == g["Hours"].max(), "Cumulative_Fold_Change"].mean(),
                      include_groups=False)
               .sort_values(ascending=False))
    phage_order = last_fc.index.tolist()
    n = len(phage_order)

    summary = (df.groupby(["Sequence", "Hours"])["Cumulative_Fold_Change"]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["Sequence", "Hours", "mean_fc", "std_fc"]
    summary["std_fc"] = summary["std_fc"].fillna(0)

    final_fc = summary.groupby("Sequence").apply(
        lambda g: g.loc[g["Hours"] == g["Hours"].max(), "mean_fc"].values[0],
        include_groups=False)

    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 2.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for i, seq in enumerate(phage_order):
        ax = axes[i]
        seq_df = summary[summary["Sequence"] == seq].sort_values("Hours")
        color = "#e377c2" if final_fc[seq] < 0 else "#1f77b4"

        ax.fill_between(seq_df["Hours"],
                        seq_df["mean_fc"] - seq_df["std_fc"],
                        seq_df["mean_fc"] + seq_df["std_fc"],
                        color=color, alpha=0.2)
        ax.plot(seq_df["Hours"], seq_df["mean_fc"], linewidth=1.2, color=color)
        ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_title(seq, fontsize=7, fontweight="bold", pad=3)
        ax.tick_params(axis="both", labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

    axes[0].set_ylabel("log2 FC", fontsize=8)
    fig.supxlabel("Time (hours)", fontsize=9)
    fig.suptitle("log2 Proportion Fold Change from T0 (Mean +/- SD)", fontsize=10, y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)

    out_path = os.path.join(output_dir, "log2fc_facet.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.savefig(out_path.replace(".svg", ".png"), format="png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_auc(auc_df, output_dir, title):
    """Signed AUC bar + strip plot."""
    summary = (
        auc_df.groupby("Sequence")["SignedAUC"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "AUC_mean", "sem": "AUC_SEM", "count": "n"})
    )
    order = summary.sort_values("AUC_mean", ascending=False)["Sequence"].tolist()

    print(f"\n  Signed AUC ranking (mean +/- SEM):")
    for _, row in summary.set_index("Sequence").loc[order].iterrows():
        print(f"    {_:>12s}: {row['AUC_mean']:>8.1f} +/- {row['AUC_SEM']:.1f}")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=summary, x="Sequence", y="AUC_mean",
                order=order, color="lightgray", errorbar=None)
    plt.errorbar(x=np.arange(len(order)),
                 y=summary.set_index("Sequence").loc[order, "AUC_mean"],
                 yerr=summary.set_index("Sequence").loc[order, "AUC_SEM"],
                 fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1)
    sns.stripplot(data=auc_df, x="Sequence", y="SignedAUC",
                  order=order, color="black", jitter=0.2, alpha=0.8)
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.ylabel("Signed AUC of cumulative log2 FC (h * log2)")
    plt.xlabel("Phage")
    plt.title(title)
    plt.xticks(rotation=90, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "signed_auc_cumulative_log2FC_barpoints.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_read_fate(stats_df, output_dir, title, max_timepoint, cw_df=None, nd_df=None):
    """
    Two-panel read fate figure.
    Panel A: Stacked bar — assigned / chimera / ambiguous (% of filter-passing) by timepoint.
    Panel B: Chimera window distribution — which windows disagree and how many per chimera.
    """
    stats_df = stats_df.copy()
    stats_df["Timepoint"] = stats_df["Sample"].apply(lambda x: int(x.split("_")[0][1:]))
    stats_df["Replicate"] = stats_df["Sample"].apply(lambda x: int(x.split("_")[1][3:]))
    stats_df = stats_df[stats_df["Timepoint"] <= max_timepoint]

    timepoints = sorted(stats_df["Timepoint"].unique())
    tp_labels = [f"T{t}" for t in timepoints]

    assigned_means, chimera_means, ambig_means = [], [], []

    for tp in timepoints:
        tp_df = stats_df[stats_df["Timepoint"] == tp]
        a_pcts = (100 * tp_df["assigned"] / tp_df["pass_filters"]).fillna(0).tolist()
        c_pcts = (100 * tp_df["chimera"] / tp_df["pass_filters"]).fillna(0).tolist()
        b_pcts = (100 * tp_df["ambiguous"] / tp_df["pass_filters"]).fillna(0).tolist()
        assigned_means.append(np.mean(a_pcts))
        chimera_means.append(np.mean(c_pcts))
        ambig_means.append(np.mean(b_pcts))

    x = np.arange(len(timepoints))
    w = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Read fate stacked bars
    ax1.bar(x, assigned_means, w, label="Assigned", color="#7fb3d8",
            edgecolor="white", linewidth=0.5)
    ax1.bar(x, chimera_means, w, bottom=assigned_means, label="Chimera (discarded)",
            color="#e74c3c", alpha=0.8, edgecolor="white", linewidth=0.5)
    bottoms2 = [a + c for a, c in zip(assigned_means, chimera_means)]
    ax1.bar(x, ambig_means, w, bottom=bottoms2, label="Ambiguous (discarded)",
            color="#f39c12", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tp_labels)
    ax1.set_ylabel("% of filter-passing reads")
    ax1.set_title("A. Read fate by timepoint", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: Chimera window distribution
    if cw_df is not None and nd_df is not None:
        n_windows = len(cw_df)
        window_labels = [f"W{i+1}" for i in range(n_windows)]
        window_counts = cw_df["chimera_disagree_count"].values

        # Bar chart of which windows disagree
        colors = ["#e74c3c"] * n_windows
        ax2.bar(np.arange(n_windows), window_counts, 0.6, color=colors, alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax2.set_xticks(np.arange(n_windows))
        ax2.set_xticklabels(window_labels)
        ax2.set_ylabel("Number of chimeric disagreements")
        ax2.set_title("B. Chimera breakpoint distribution", fontweight="bold", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)

        # Add inset with number of disagreeing windows per chimera
        total_chimeras = nd_df["chimera_count"].sum()
        inset_text = "Windows per chimera:\n"
        for _, row in nd_df.iterrows():
            n = int(row["n_disagreeing_windows"])
            cnt = int(row["chimera_count"])
            pct = 100 * cnt / total_chimeras if total_chimeras > 0 else 0
            inset_text += f"  {n}: {cnt:,} ({pct:.0f}%)\n"
        ax2.text(0.97, 0.97, inset_text.strip(), transform=ax2.transAxes,
                 fontsize=8, verticalalignment="top", horizontalalignment="right",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"))
    else:
        ax2.text(0.5, 0.5, "No chimera window data\n(re-run stage 2)",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=12, color="gray")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "read_fate_by_timepoint.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.savefig(out_path.replace(".svg", ".png"), format="png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def run_stage4(config):
    """Generate all visualizations."""
    print(f"\n{'='*60}")
    print(f"  STAGE 4: Visualization — {config['name']}")
    print(f"{'='*60}")

    output_dir = config["output_dir"]
    title = config["name"]
    max_tp = config["max_timepoint"]

    cum_df = _load_cum_df(config)
    if cum_df is None:
        return

    print(f"  Loaded {len(cum_df)} rows, {cum_df['Sequence'].nunique()} phages")

    # Line plots
    print(f"\n  [1] log2 FC line plot...")
    plot_log2fc_lineplot(cum_df, output_dir)

    print(f"  [2] log2 FC facet plot...")
    plot_log2fc_facet(cum_df, output_dir)

    # AUC bar plot
    print(f"  [3] Signed AUC bar plot...")
    auc_csv = os.path.join(output_dir, "signed_auc_cumulative_log2FC.csv")
    if os.path.exists(auc_csv):
        auc_df = pd.read_csv(auc_csv)
        plot_auc(auc_df, output_dir, title)
    else:
        auc_df = compute_signed_auc(cum_df)
        plot_auc(auc_df, output_dir, title)

    # Read fate plot
    print(f"  [4] Read fate by timepoint...")
    stats_csv = os.path.join(output_dir, "read_fate_stats.csv")
    if os.path.exists(stats_csv):
        stats_df = pd.read_csv(stats_csv)
        cw_csv = os.path.join(output_dir, "chimera_window_stats.csv")
        nd_csv = os.path.join(output_dir, "chimera_ndisagree_stats.csv")
        cw_df = pd.read_csv(cw_csv) if os.path.exists(cw_csv) else None
        nd_df = pd.read_csv(nd_csv) if os.path.exists(nd_csv) else None
        plot_read_fate(stats_df, output_dir, title, max_tp, cw_df, nd_df)
    else:
        print(f"  [skip] No read_fate_stats.csv — run stage 2 first")

    print(f"\n  Stage 4 complete. All plots in: {output_dir}")


# ═══════════════════════════════════════════════════════
# Dataset Configs
# ═══════════════════════════════════════════════════════

EXPERIMENTS_DIR = "/home/samuelking/phage_gen/experiments"

NATURAL_CONFIG = {
    "name": "Natural Phage Competition",
    "experiment_dir": f"{EXPERIMENTS_DIR}/20260304_naturalphage_competition_analysis_final",
    "raw_fastq_dir": f"{EXPERIMENTS_DIR}/20260303_naturalphage_competition_analysis/VBL8QM_fastq",
    "reference_fasta": f"{EXPERIMENTS_DIR}/20260303_naturalphage_competition_analysis/rokyta2006_phix174like_genomes_with_evophi69_nc001422_s13_med1_SK390SK359_amplicons.fasta",
    "samples": [
        ("T0_rep1", "VBL8QM_1_T0_rep1.fastq"),
        ("T0_rep2", "VBL8QM_2_T0_rep2.fastq"),
        ("T0_rep3", "VBL8QM_3_T0_rep3.fastq"),
        ("T1_rep1", "VBL8QM_4_T1_rep1.fastq"),
        ("T1_rep2", "VBL8QM_5_T1_rep2.fastq"),
        ("T1_rep3", "VBL8QM_6_T1_rep3.fastq"),
        ("T2_rep1", "VBL8QM_7_T2_rep1.fastq"),
        ("T2_rep2", "VBL8QM_8_T2_rep2.fastq"),
        ("T2_rep3", "VBL8QM_9_T2_rep3.fastq"),
        ("T3_rep1", "VBL8QM_10_T3_rep1.fastq"),
        ("T3_rep2", "VBL8QM_11_T3_rep2.fastq"),
        ("T3_rep3", "VBL8QM_12_T3_rep3.fastq"),
        ("T4_rep1", "VBL8QM_13_T4_rep1.fastq"),
        ("T4_rep2", "VBL8QM_14_T4_rep2.fastq"),
        ("T4_rep3", "VBL8QM_15_T4_rep3.fastq"),
        ("T5_rep1", "VBL8QM_16_T5_rep1.fastq"),
        ("T5_rep2", "VBL8QM_17_T5_rep2.fastq"),
        ("T5_rep3", "VBL8QM_18_T5_rep3.fastq"),
        ("T6_rep1", "VBL8QM_19_T6_rep1.fastq"),
        ("T6_rep2", "VBL8QM_20_T6_rep2.fastq"),
        ("T6_rep3", "VBL8QM_21_T6_rep3.fastq"),
        ("T7_rep1", "VBL8QM_22_T7_rep1.fastq"),
        ("T7_rep2", "VBL8QM_23_T7_rep2.fastq"),
        ("T7_rep3", "VBL8QM_24_T7_rep3.fastq"),
        ("T8_rep1", "VBL8QM_25_T8_rep1.fastq"),
        ("T8_rep2", "VBL8QM_26_T8_rep2.fastq"),
        ("T8_rep3", "VBL8QM_27_T8_rep3.fastq"),
        ("T9_rep1", "VBL8QM_28_T9_rep1.fastq"),
        ("T9_rep2", "VBL8QM_29_T9_rep2.fastq"),
        ("T9_rep3", "VBL8QM_30_T9_rep3.fastq"),
    ],
    "scoring_mode": "direct",
    "min_pid": 0.95,
    "min_align_frac": 0.90,
    "accession_to_name": {
        "NC_001422.1": "phiX174", "Evo-φ69": "Evo-φ69",
        "DQ079891.1": "NC51", "DQ079894.1": "WA10", "DQ079883.1": "ID45",
        "DQ079885.1": "NC5", "DQ079890.1": "NC41", "DQ079888.1": "NC16",
        "DQ079889.1": "NC37", "DQ079887.1": "NC11",
        "DQ079884.1": "NC1", "DQ079886.1": "NC7", "DQ079881.1": "ID22",
        "DQ079882.1": "ID34", "DQ079893.1": "WA4",
        "DQ079895.1": "WA11", "KJ997912.1": "MED1", "AF274751.1": "S13",
    },
    "exclude_phages": set(),
    "timepoint_to_hours": {
        0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5,
        6: 3, 7: 4, 8: 5, 9: 6
    },
    "max_timepoint": 9,
    "output_dir": f"{EXPERIMENTS_DIR}/20260304_naturalphage_competition_analysis_final",
}

EVO_CONFIG = {
    "name": "Evo Phage Competition",
    "experiment_dir": f"{EXPERIMENTS_DIR}/20260304_evophage_competition_analysis_final",
    "raw_fastq_dir": f"{EXPERIMENTS_DIR}/20260303_evophage_competition_analysis/fastq",
    "reference_fasta": f"{EXPERIMENTS_DIR}/20260303_evophage_competition_analysis/final_evo_phage_genomes_seqverified_SK324SK359_amplicons.fasta",
    "samples": [
        ("T0_rep1", "DLLNRN_3_competition_T0_rep1_SK324SK359.fastq"),
        ("T0_rep2", "DLLNRN_4_competition_T0_rep2_SK324SK359.fastq"),
        ("T0_rep3", "DLLNRN_5_competition_T0_rep3_SK324SK359.fastq"),
        ("T1_rep1", "Q8VFDH_1_competition_T1_rep1.fastq"),
        ("T1_rep2", "Q8VFDH_2_competition_T1_rep2.fastq"),
        ("T1_rep3", "Q8VFDH_3_competition_T1_rep3.fastq"),
        ("T2_rep1", "Q8VFDH_4_competition_T2_rep1.fastq"),
        ("T2_rep2", "Q8VFDH_5_competition_T2_rep2.fastq"),
        ("T2_rep3", "Q8VFDH_6_competition_T2_rep3.fastq"),
        ("T3_rep1", "88M8NX_1_competition_T3_rep1.fastq"),
        ("T3_rep2", "88M8NX_2_competition_T3_rep2.fastq"),
        ("T3_rep3", "88M8NX_3_competition_T3_rep3.fastq"),
        ("T4_rep1", "88M8NX_4_competition_T4_rep1.fastq"),
        ("T4_rep2", "88M8NX_5_competition_T4_rep2.fastq"),
        ("T4_rep3", "88M8NX_6_competition_T4_rep3.fastq"),
        ("T5_rep1", "88M8NX_7_competition_T5_rep1.fastq"),
        ("T5_rep2", "88M8NX_8_competition_T5_rep2.fastq"),
        ("T5_rep3", "88M8NX_9_competition_T5_rep3.fastq"),
        ("T6_rep1", "DLLNRN_6_competition_T6_rep1_SK324SK359.fastq"),
        ("T6_rep2", "DLLNRN_7_competition_T6_rep2_SK324SK359.fastq"),
        ("T6_rep3", "DLLNRN_8_competition_T6_rep3_SK324SK359.fastq"),
        ("T7_rep1", "88M8NX_10_competition_T7_rep1.fastq"),
        ("T7_rep2", "88M8NX_11_competition_T7_rep2.fastq"),
        ("T7_rep3", "88M8NX_12_competition_T7_rep3.fastq"),
        ("T8_rep1", "88M8NX_13_competition_T8_rep1.fastq"),
        ("T8_rep2", "88M8NX_14_competition_T8_rep2.fastq"),
        ("T8_rep3", "88M8NX_15_competition_T8_rep3.fastq"),
        ("T9_rep1", "88M8NX_16_competition_T9_rep1.fastq"),
        ("T9_rep2", "3HFS2P_1_competition_T9_rep2.fastq"),
        ("T9_rep3", "3HFS2P_2_competition_T9_rep3.fastq")
    ],
    "scoring_mode": "msa",
    "min_pid": 0.90,
    "min_align_frac": 0.90,
    "accession_to_name": {"NC_001422.1": "phiX174"},
    "exclude_phages": set(),
    "timepoint_to_hours": {
        0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5,
        6: 3, 7: 4, 8: 5, 9: 6
    },
    "max_timepoint": 9,
    "output_dir": f"{EXPERIMENTS_DIR}/20260304_evophage_competition_analysis_final",
}


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phage competition analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", default="both",
                        choices=["natural", "evo", "both"],
                        help="Which dataset to process (default: both)")
    parser.add_argument("--stages", default="1234",
                        help="Which stages to run, e.g. '34' for fold change + plots (default: 1234)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip samples with existing output files")
    args = parser.parse_args()

    configs = []
    if args.dataset in ("natural", "both"):
        configs.append(NATURAL_CONFIG)
    if args.dataset in ("evo", "both"):
        configs.append(EVO_CONFIG)

    stages = set(args.stages)

    for config in configs:
        print(f"\n{'#'*60}")
        print(f"  {config['name']}")
        print(f"  Output: {config['output_dir']}")
        print(f"  Stages: {args.stages}")
        print(f"{'#'*60}")

        if "1" in stages:
            run_stage1(config, skip_existing=args.skip_existing)
        if "2" in stages:
            run_stage2(config, skip_existing=args.skip_existing)
        if "3" in stages:
            run_stage3(config)
        if "4" in stages:
            run_stage4(config)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
