"""
Usage:

conda activate competition_analysis
python /path/to/plot_competition_analysis.py

Standalone plotting script for phage competition cumulative fold changes.
Plots three separate subplots (rows), one per replicate (rep1/rep2/rep3).
Computes fold changes per replicate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


### Helper functions ###

def timepoint_to_hours(timepoint: int) -> float:
    """Convert time point index to hours of collection."""
    mapping = {
        0: 0,   1: 0.5, 2: 1,   3: 1.5, 4: 2,  5: 2.5,
        6: 3,   7: 4,   8: 5,   9: 6,  10: 8, 11: 10, 12: 12
    }
    return mapping.get(int(timepoint), float(timepoint))

def parse_sample_info(sample_name: str):
    """Extract (timepoint, replicate) from Sample strings like 'T6_rep2'."""
    parts = sample_name.split('_')
    tp = int(parts[0][1:])      # 'T6' -> 6
    rep = int(parts[1][3:])     # 'rep2' -> 2
    return tp, rep

def calculate_cumulative_fc_per_replicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Sequence and Replicate, compute per-timepoint log2 fold change
    relative to the previous timepoint, then the cumulative log2 fold change
    relative to T0. Returns tidy DataFrame with one row per (Sequence, Replicate, Timepoint).
    """
    # Add parsed columns
    df = df.copy()
    df['Timepoint'] = df['Sample'].apply(lambda x: parse_sample_info(x)[0])
    df['Replicate'] = df['Sample'].apply(lambda x: parse_sample_info(x)[1])
    df['Hours'] = df['Timepoint'].apply(timepoint_to_hours)

    out_rows = []

    # Work per (Sequence, Replicate)
    for (seq, rep), sub in df.groupby(['Sequence', 'Replicate']):
        sub = sub.sort_values('Timepoint')
        cumulative_fc = 0.0

        # iterate in time
        for i, row in sub.iterrows():
            tp = row['Timepoint']
            hours = row['Hours']
            curr = row['Read Count']

            if tp == 0:
                fold_change = 0.0
                cumulative_fc = 0.0
            else:
                # previous timepoint row (if missing, skip this tp)
                prev_rows = sub[sub['Timepoint'] == tp - 1]
                if len(prev_rows) == 0:
                    continue
                prev = prev_rows['Read Count'].iloc[0]

                if prev > 0 and curr > 0:
                    fold_change = np.log2(curr / prev)
                    cumulative_fc += fold_change
                elif prev == 0 and curr > 0:
                    fold_change = np.inf
                    cumulative_fc = np.inf
                else:
                    # curr == 0 (or both zero) -> define FC as -inf or 0; choose 0 to keep axis finite
                    fold_change = 0.0
                    # cumulative stays as-is

            out_rows.append({
                'Sequence': seq,
                'Replicate': rep,
                'Timepoint': tp,
                'Hours': hours,
                'Read_Count': curr,
                'Fold_Change': fold_change,
                'Cumulative_Fold_Change': cumulative_fc
            })

    return pd.DataFrame(out_rows)

def plot_cumulative_by_replicate(cum_df: pd.DataFrame, output_dir: str, max_timepoint: int = 9):
    """
    Make a 3-row figure: one subplot per replicate, showing cumulative log2 fold change over time
    for all sequences in that replicate.
    """
    # Filter to desired timepoints
    data = cum_df[cum_df['Timepoint'] <= max_timepoint].copy()
    reps = sorted(data['Replicate'].unique())

    # Consistent palette across subplots
    sequences = sorted(data['Sequence'].unique())
    palette = sns.color_palette("tab20", len(sequences))
    color_map = {seq: palette[i % len(palette)] for i, seq in enumerate(sequences)}

    fig, axes = plt.subplots(nrows=len(reps), ncols=1, figsize=(14, 9), sharex=True, sharey=True)

    if len(reps) == 1:
        axes = [axes]  # ensure iterable

    # Set x ticks across all panels
    all_hours = sorted(data['Hours'].unique())

    for ax, rep in zip(axes, reps):
        rep_df = data[data['Replicate'] == rep].copy()

        # Optional: order sequences by their final cumulative FC in this replicate (descending)
        final_tp = rep_df['Timepoint'].max()
        final_vals = (rep_df[rep_df['Timepoint'] == final_tp]
                      .set_index('Sequence')['Cumulative_Fold_Change'])
        ordered_sequences = [s for s in sequences if s in final_vals.index]
        ordered_sequences += [s for s in sequences if s not in ordered_sequences]  # keep any missing at end

        for seq in ordered_sequences:
            seq_df = rep_df[rep_df['Sequence'] == seq].sort_values('Hours')
            if seq_df.empty:
                continue
            ax.plot(seq_df['Hours'], seq_df['Cumulative_Fold_Change'],
                    marker='o', linewidth=2, markersize=5,
                    color=color_map[seq], alpha=0.9, label=seq)

        ax.set_title(f"Cumulative log2 Fold Change — Replicate {rep}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel("Cumulative log2 fold change")
        ax.set_xticks(all_hours)

    axes[-1].set_xlabel("Time (hours)")

    # Single legend outside (right)
    # Build legend handles from the first axes lines (unique by label)
    handles, labels = [], []
    for line in axes[0].get_lines():
        lab = line.get_label()
        if lab not in labels:
            handles.append(line)
            labels.append(lab)
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', title="Sequence")

    plt.tight_layout(rect=[0, 0, 0.86, 1])  # leave space on right for legend

    out_path = os.path.join(output_dir, "cumulative_fold_change_by_replicate_T0_to_T9.svg")
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    print(f"Saved plot: {out_path}")
    plt.show()


### Main analysis ###

if __name__ == "__main__":
    results_dir = "/large_storage/hielab/samuelking/phage_design/generation/data/20250807_competition_analysis/final_mapq20_pid90_len70"
    merged_csv = f"{results_dir}/all_timepoints_read_counts_primary.csv"

    if not os.path.exists(merged_csv):
        print(f"Error: Could not find merged CSV file: {merged_csv}")
        exit(1)

    print("Loading merged data...")
    merged_df = pd.read_csv(merged_csv)
    print(f"Loaded {len(merged_df)} rows from {merged_csv}")

    print("Calculating cumulative fold changes per replicate...")
    cum_df = calculate_cumulative_fc_per_replicate(merged_df)

    # Restrict to T0–T9 as before
    print("Filtering to T0–T9...")
    cum_df = cum_df[cum_df['Timepoint'] <= 9].copy()

    print("Plotting cumulative fold changes as three separate replicate panels...")
    plot_cumulative_by_replicate(cum_df, results_dir, max_timepoint=10)

    # Small summary
    print("\n=== Summary ===")
    print(f"Sequences: {len(cum_df['Sequence'].unique())}")
    print(f"Replicates: {sorted(cum_df['Replicate'].unique())}")
    print(f"Timepoints (T): {sorted(cum_df['Timepoint'].unique())}")
    print(f"Hours: {sorted(cum_df['Hours'].unique())}")