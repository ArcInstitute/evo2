#!/usr/bin/env python3
"""Run the Evo 2 Gene Completion benchmark (% amino-acid recovery).

For each gene in a panel the model is prompted with the start of the gene and
asked to complete it; the generated protein is recovered and compared to the
reference over the non-prompt region. See README.md for the methodology.

Examples
--------
Prokaryote/archaea panel (MSA-style protein alignment)::

    python run_gene_completion.py --panel prokaryote --model_name evo2_7b \
        --output_dir results/evo2_7b

Eukaryote panel (requires exonerate)::

    python run_gene_completion.py --panel eukaryote --model_name evo2_7b \
        --output_dir results/evo2_7b
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prompts import prokaryote_prompt, eukaryote_prompt, PROK_UPSTREAM_LEN
from scoring import (
    score_prokaryote,
    score_eukaryote,
    exonerate_spliced_protein,
)

HERE = Path(__file__).resolve().parent

# Panel-specific generation defaults, matching the published runs.
PANEL_DEFAULTS = {
    "prokaryote": {"temperature": 0.7, "data": HERE / "data" / "prokaryote_genes.csv"},
    "eukaryote": {"temperature": 1.0, "data": HERE / "data" / "eukaryote_genes.csv"},
}


def clean_seq(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def generate_completions(
    evo_model,
    prompt: str,
    n_tokens: int,
    num_generations: int,
    temperature: float,
    top_k: int,
    batch_size: int,
) -> List[str]:
    """Generate ``num_generations`` completions for one prompt; return the full
    sequences (prompt + continuation), cleaned of whitespace."""
    out: List[str] = []
    for start in range(0, num_generations, max(1, batch_size)):
        n = min(max(1, batch_size), num_generations - start)
        seqs, _ = evo_model.generate(
            prompt_seqs=[prompt] * n,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            batched=True,
            cached_generation=True,
            verbose=0,
            force_prompt_threshold=min(200, len(prompt)),
        )
        if isinstance(seqs, str):
            seqs = [seqs]
        for s in seqs:
            s = clean_seq(s)
            if not s.startswith(prompt):
                s = prompt + s
            out.append(s)
    return out[:num_generations]


def evaluate(args) -> None:
    data_path = Path(args.data) if args.data else PANEL_DEFAULTS[args.panel]["data"]
    df = pd.read_csv(data_path)
    if args.genes:
        wanted = {g.strip().lower() for g in args.genes.split(",")}
        df = df[df["gene"].str.lower().isin(wanted)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No genes selected.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazily import Evo 2 so --help / dataset inspection works without a GPU.
    from evo2 import Evo2
    evo_model = Evo2(args.model_name)

    raw_rows: List[Dict] = []
    for _, row in df.iterrows():
        gene = row["gene"]
        genomic = clean_seq(row["genomic_sequence"])
        ref_protein = str(row["reference_protein"]).strip()

        if args.panel == "prokaryote":
            prompt, prompt_cds_aa = prokaryote_prompt(genomic, int(row["cds_start"]))
            remaining_aa = max(1, len(ref_protein) - prompt_cds_aa)
            n_tokens = args.n_tokens or (remaining_aa * 3 + 150)
        else:
            prompt, prompt_len = eukaryote_prompt(genomic)
            n_tokens = args.n_tokens or (len(genomic) - prompt_len + 50)
            # Reference spliced protein from the TRUE genomic (exonerate), once.
            _, ref_spliced = exonerate_spliced_protein(genomic, ref_protein, args.exonerate_path)

        print(f"[{gene}] prompt={len(prompt)} nt, n_tokens={n_tokens}, "
              f"generations={args.num_generations}")
        completions = generate_completions(
            evo_model, prompt, n_tokens, args.num_generations,
            args.temperature, args.top_k, args.batch_size,
        )

        for gen_idx, gen_full in enumerate(completions):
            if args.panel == "prokaryote":
                recovery = score_prokaryote(
                    gen_full, ref_protein, upstream_len=PROK_UPSTREAM_LEN,
                    prompt_cds_aa=prompt_cds_aa,
                )
            else:
                recovery = score_eukaryote(
                    gen_full, ref_spliced, prompt_len, ref_protein, args.exonerate_path,
                )
            raw_rows.append({
                "gene": gene,
                "organism": row.get("organism", ""),
                "gen_idx": gen_idx,
                "aa_recovery_non_prompt": recovery,
                "prompt_len": len(prompt),
            })

    raw = pd.DataFrame(raw_rows)
    raw_path = out_dir / f"{args.model_name}_{args.panel}_completions.csv"
    raw.to_csv(raw_path, index=False)

    stats = (
        raw.groupby("gene")
        .agg(
            n_samples=("aa_recovery_non_prompt", "count"),
            mean_aa_recovery=("aa_recovery_non_prompt", "mean"),
            std_aa_recovery=("aa_recovery_non_prompt", "std"),
        )
        .reset_index()
    )
    stats["sem_aa_recovery"] = stats["std_aa_recovery"] / np.sqrt(stats["n_samples"])
    stats_path = out_dir / f"{args.model_name}_{args.panel}_per_gene_stats.csv"
    stats.to_csv(stats_path, index=False)

    print(f"\nWrote {raw_path}\nWrote {stats_path}\n")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nPanel mean AA recovery: {stats['mean_aa_recovery'].mean():.2f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", required=True, choices=["prokaryote", "eukaryote"])
    p.add_argument("--model_name", required=True, help="e.g. evo2_7b, evo2_40b")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data", default=None, help="Override panel dataset CSV")
    p.add_argument("--genes", default="", help="Comma-separated subset of genes")
    p.add_argument("--num_generations", type=int, default=50)
    p.add_argument("--temperature", type=float, default=None, help="Default: panel-specific")
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_tokens", type=int, default=None, help="Default: cover the remainder")
    p.add_argument("--exonerate_path", default="exonerate")
    args = p.parse_args()

    if args.temperature is None:
        args.temperature = PANEL_DEFAULTS[args.panel]["temperature"]
    evaluate(args)


if __name__ == "__main__":
    main()
