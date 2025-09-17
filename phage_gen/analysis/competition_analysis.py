"""
Usage:

conda activate competition_analysis
python /path/to/competition_analysis.py

Phage competition assay read mapping and counting
Processes multiple time points and replicates from raw long-read sequencing data
"""

import subprocess
import os
import csv
import pandas as pd
import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


### Helper functions ###

def run_fastqc(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[FastQC] {input_file}")
    subprocess.run(["fastqc", input_file, "-o", output_dir], check=True)

def run_fastp(input_file, output_file, report_file, min_length=250, quality_threshold=20, unqualified_limit=30):
    print(f"[fastp] {input_file}")
    subprocess.run([
        "fastp",
        "-i", input_file,
        "-o", output_file,
        "-l", str(min_length),
        "-q", str(quality_threshold),
        "-u", str(unqualified_limit),
        "--html", report_file
    ], check=True)

def sam_to_sorted_bam(sam_file, sorted_bam_file):
    subprocess.run(["samtools", "view", "-bS", sam_file, "-o", "temp_unsorted.bam"], check=True)
    subprocess.run(["samtools", "sort", "temp_unsorted.bam", "-o", sorted_bam_file], check=True)
    subprocess.run(["samtools", "index", sorted_bam_file], check=True)
    os.remove("temp_unsorted.bam")

def align_reads_primary_only(filtered_reads, reference_sequences, output_sam):
    """Align reads using minimap2 with primary alignments only"""
    print(f"[minimap2] Aligning {filtered_reads} to {reference_sequences}")
    subprocess.run([
        "minimap2", "-ax", "map-ont", "--secondary=no", "-t", "8",
        reference_sequences, filtered_reads, "-o", output_sam
    ], check=True)

def get_reference_lengths(fasta_path):
    """Read reference sequence lengths from FASTA file"""
    ref_lengths = {}
    try:
        from Bio import SeqIO
        for record in SeqIO.parse(fasta_path, "fasta"):
            ref_lengths[record.id] = len(record.seq)
    except ImportError:
        # Fallback: parse FASTA manually if BioPython not available
        with open(fasta_path, 'r') as f:
            current_id = None
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        ref_lengths[current_id] = len(current_seq)
                    current_id = line[1:].split()[0]  # Take first part of header
                    current_seq = ""
                else:
                    current_seq += line
            # Don't forget the last sequence
            if current_id is not None:
                ref_lengths[current_id] = len(current_seq)
    
    return ref_lengths

def sam_to_counts(sam_path, out_csv, ref_lengths, min_align_len):
    """Count primary, high-quality, near-full-length, high-identity alignments"""
    import pysam
    import csv
    
    aln = pysam.AlignmentFile(sam_path, "r")
    counts = {}
    
    for r in aln:
        # Skip unmapped, secondary, or supplementary alignments
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
            
        # Filter by mapping quality
        if r.mapping_quality < 20:
            continue
            
        # Get expected length for this reference sequence
        ref_name = r.reference_name
        if ref_name not in ref_lengths:
            print(f"Warning: Reference {ref_name} not found in reference lengths")
            continue
        
        exp_len = ref_lengths[ref_name]
        
        # Filter by alignment length (≥XX% of expected amplicon length)
        qlen = r.query_alignment_length or 0
        if qlen < min_align_len * exp_len:
            continue
            
        # Filter by percent identity (≥99% for highly similar amplicons)
        nm = r.get_tag("NM") if r.has_tag("NM") else 0
        pid = 1.0 - (nm / max(1, qlen))   # fraction identity
        if pid < 0.9:
            continue
            
        # Count this alignment
        counts[ref_name] = counts.get(ref_name, 0) + 1
    
    aln.close()

    # Write counts table with proportions
    total = sum(counts.values())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Sequence", "Read Count", "Proportion"])
        w.writeheader()
        for ref, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            proportion = (c / total) if total else 0.0
            w.writerow({"Sequence": ref, "Read Count": c, "Proportion": proportion})

def parse_sample_info(sample_name):
    """Extract time point and replicate from sample name"""
    # Expected format: T{timepoint}_rep{replicate}
    parts = sample_name.split('_')
    timepoint = int(parts[0][1:])  # Remove 'T' and convert to int
    replicate = int(parts[1][3:])  # Remove 'rep' and convert to int
    return timepoint, replicate

def calculate_fold_changes(df):
    """Calculate fold changes relative to previous time point for each sequence"""
    # Parse sample information
    df['Timepoint'] = df['Sample'].apply(lambda x: parse_sample_info(x)[0])
    df['Replicate'] = df['Sample'].apply(lambda x: parse_sample_info(x)[1])
    
    # Calculate mean read count for each sequence at each timepoint
    mean_counts = df.groupby(['Sequence', 'Timepoint'])['Read Count'].mean().reset_index()
    mean_counts = mean_counts.rename(columns={'Read Count': 'Mean_Read_Count'})
    
    # Calculate fold changes
    fold_changes = []
    sequences = mean_counts['Sequence'].unique()
    
    for seq in sequences:
        seq_data = mean_counts[mean_counts['Sequence'] == seq].sort_values('Timepoint')
        
        for i, row in seq_data.iterrows():
            tp = row['Timepoint']
            current_count = row['Mean_Read_Count']
            
            if tp == 0:
                # T0 is set to 0 fold change (reference point)
                fold_change = 0.0
            else:
                # Find previous timepoint
                prev_data = seq_data[seq_data['Timepoint'] == tp - 1]
                if len(prev_data) > 0:
                    prev_count = prev_data['Mean_Read_Count'].iloc[0]
                    if prev_count > 0:
                        fold_change = np.log2(current_count / prev_count)
                    else:
                        # Handle case where previous count was 0
                        fold_change = np.inf if current_count > 0 else 0.0
                else:
                    # No previous timepoint data, skip
                    continue
            
            fold_changes.append({
                'Sequence': seq,
                'Timepoint': tp,
                'Mean_Read_Count': current_count,
                'Fold_Change': fold_change
            })
    
    return pd.DataFrame(fold_changes)


### Main analysis ###

reference_sequences = "/large_storage/hielab/samuelking/phage_design/generation/data/20250807_competition_analysis/final_evo_phage_genomes_seqverified_SK324SK359_amplicons.fasta"
data_dir = "/large_storage/hielab/samuelking/phage_design/generation/data/20250807_competition_analysis"
results_dir = f"{data_dir}/final_mapq20_pid90_len70"

# Create output directory
os.makedirs(results_dir, exist_ok=True)
print(f"Reading FASTQ files from: {data_dir}")
print(f"Saving analyses to: {results_dir}")

fastqc_dir = f"{results_dir}/qc_reports"
fastqc_filtered_dir = f"{results_dir}/qc_reports_filtered"

# Read reference sequence lengths from FASTA file
print("Reading reference sequence lengths...")
ref_lengths = get_reference_lengths(reference_sequences)
print(f"Found {len(ref_lengths)} reference sequences:")
for ref_name, length in ref_lengths.items():
    print(f"  {ref_name}: {length} bp")

# Debug: Check if first file exists
print(f"\nDebugging - Looking for FASTQ files in: {data_dir}")
try:
    files = os.listdir(data_dir)
    fastq_files = [f for f in files if f.endswith('.fastq')]
    print(f"Found {len(fastq_files)} .fastq files:")
    for f in sorted(fastq_files)[:5]:  # Show first 5
        print(f"  {f}")
    if len(fastq_files) > 5:
        print(f"  ... and {len(fastq_files) - 5} more")
except Exception as e:
    print(f"Error listing directory: {e}")

# Adjust as needed for FASTQ files
samples = [
    # prefix, fastq filename
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
    ("T9_rep3", "3HFS2P_2_competition_T9_rep3.fastq"),

    ("T10_rep1", "3HFS2P_3_competition_T10_rep1.fastq"),
    ("T10_rep2", "3HFS2P_4_competition_T10_rep2.fastq"),
    ("T10_rep3", "3HFS2P_5_competition_T10_rep3.fastq"),

    ("T11_rep1", "3HFS2P_6_competition_T11_rep1.fastq"),
    ("T11_rep2", "3HFS2P_7_competition_T11_rep2.fastq"),
    ("T11_rep3", "3HFS2P_8_competition_T11_rep3.fastq"),

    ("T12_rep1", "3HFS2P_9_competition_T12_rep1.fastq"),
    ("T12_rep2", "3HFS2P_10_competition_T12_rep2.fastq"),
    ("T12_rep3", "3HFS2P_11_competition_T12_rep3.fastq")
]

all_counts = []

for sample_name, fastq_file in samples:
    print(f"\n=== Processing {sample_name} ===")
    
    input_fastq = f"{data_dir}/{fastq_file}"
    
    # Check if input file exists
    if not os.path.exists(input_fastq):
        print(f"Warning: File not found: {input_fastq}")
        print(f"Skipping {sample_name}")
        continue
    
    filtered_fastq = f"{results_dir}/{sample_name}_filtered.fastq"
    fastp_report = f"{results_dir}/{sample_name}_fastp.html"
    output_sam = f"{results_dir}/{sample_name}.sam"
    sorted_bam = f"{results_dir}/{sample_name}.sorted.bam"
    counts_csv = f"{results_dir}/{sample_name}_counts_primary.csv"

    # QC & filtering
    run_fastqc(input_fastq, fastqc_dir)
    run_fastp(input_fastq, filtered_fastq, fastp_report)
    run_fastqc(filtered_fastq, fastqc_filtered_dir)

    # Alignment with primary hits only
    align_reads_primary_only(filtered_fastq, reference_sequences, output_sam)

    # Convert to sorted BAM (optional, for visualization)
    sam_to_sorted_bam(output_sam, sorted_bam)

    # Count primary, high-quality alignments
    sam_to_counts(output_sam, counts_csv, ref_lengths, min_align_len=0.7)
    print(f"[counts] wrote {counts_csv}")

    # Append to master DataFrame
    df = pd.read_csv(counts_csv)
    df["Sample"] = sample_name
    all_counts.append(df)

# Merge all counts into one file
if all_counts:
    merged_df = pd.concat(all_counts, ignore_index=True)
    output_file = f"{results_dir}/all_timepoints_read_counts_primary.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged read counts written to {output_file}")

    # Calculate fold changes
    print("\n=== Calculating fold changes ===")
    fold_change_df = calculate_fold_changes(merged_df)

    # Save fold change data
    fold_change_file = f"{results_dir}/fold_changes_over_time.csv"
    fold_change_df.to_csv(fold_change_file, index=False)
    print(f"Fold change data written to: {fold_change_file}")

    # Save cumulative fold change data
    cumulative_file = f"{results_dir}/cumulative_fold_changes.csv"
    cumulative_df.to_csv(cumulative_file, index=False)
    print(f"Cumulative fold change data written to: {cumulative_file}")

    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Processed {len(all_counts)} samples successfully")
    print(f"Total unique sequences detected: {merged_df['Sequence'].nunique()}")
    print(f"Total reads counted: {merged_df['Read Count'].sum()}")
    print(f"Time points analyzed: {sorted(fold_change_df['Timepoint'].unique())}")

    # Print top sequences by final timepoint abundance
    final_tp = fold_change_df['Timepoint'].max()
    final_abundance = merged_df[merged_df['Sample'].str.contains(f'T{final_tp}')].groupby('Sequence')['Read Count'].mean().sort_values(ascending=False)
    print(f"\nTop sequences at final timepoint (T{final_tp}):")
    for seq, count in final_abundance.head(5).items():
        print(f"  {seq}: {count:.0f} reads (avg)")
else:
    print("Error: No samples were processed successfully. Please check file paths.")
    print("Available .fastq files found in directory are listed above.")