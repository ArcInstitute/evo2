#!/usr/bin/env python3
"""Combine prokaryote + eukaryote per-gene stats into the headline AA recovery.

The headline gene-completion number is the equal-weighted mean of the per-gene
mean AA recovery across all 12 benchmark genes (4 prokaryotic + 8 eukaryotic).

Usage::

    python summarize.py out/evo2_7b/evo2_7b_prokaryote_per_gene_stats.csv \
                        out/evo2_7b/evo2_7b_eukaryote_per_gene_stats.csv
"""
from __future__ import annotations

import sys

import pandas as pd


def main(paths) -> None:
    if not paths:
        sys.exit("Usage: python summarize.py <per_gene_stats.csv> [more.csv ...]")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    per_gene = df.set_index("gene")["mean_aa_recovery"]
    print(per_gene.to_string(float_format=lambda x: f"{x:.2f}"))
    print("-" * 28)
    print(f"genes:   {len(per_gene)}")
    print(f"overall: {per_gene.mean():.2f}%  AA recovery")


if __name__ == "__main__":
    main(sys.argv[1:])
