# Gene Completion benchmark (% amino-acid recovery)

The gene completion benchmark from the Evo 2 paper
([Brixi et al., *Nature* 2026](https://www.nature.com/articles/s41586-026-10176-5)):
dataset, evaluation code, and precomputed results for measuring percent
amino-acid (AA) recovery on gene completion. This evaluation serves as a basic sanity check for recovering generated genes and correlates closely with the training loss.

## What the benchmark measures

A model is prompted with the start of a gene (upstream context + the first
~30% of its coding region) and asked to complete the rest. The generated
DNA is translated into protein and compared to the gene's reference protein. The
score is percent amino-acid (AA) recovery — the percent identity between the
generated and reference proteins, measured over the generated portion.

The benchmark spans two panels because eukaryotic genes contain introns and so
require a splice-aware alignment:

| Panel | Genes | Organisms | Scoring |
|-------|-------|-----------|---------|
| **Prokaryote / archaea** | ftsZ, secY, dnaK, gyrA | *Haloferax volcanii*, *Escherichia coli* | translate the generated CDS → global protein alignment to the reference → identity over the non-prompt region |
| **Eukaryote** | PGK1, ADH2, CYC1, HSP70, RPS15, ACT7, ActB, GAPDH | *S. cerevisiae*, *C. reinhardtii*, *A. thaliana*, *H. sapiens* | splice the generated genomic sequence with `exonerate` (protein2genome) → translate → identity over the non-prompt region |

## Results

`results/summary.csv` — mean AA recovery for the headline models:

| Model | Prokaryote | Eukaryote | Overall (12 genes) |
|-------|:----------:|:---------:|:------------------:|
| Evo 2 1B base | 64.9 | 50.4 | **55.2** |
| Evo 2 7B      | 78.7 | 67.4 | **71.2** |
| Evo 2 20B     | 90.9 | 79.5 | **83.3** |
| Evo 2 40B     | 92.0 | 83.4 | **86.2** |

AA recovery rises monotonically with model scale.
Per-gene values are in `results/eukaryote_per_gene.csv` and
`results/prokaryote_per_gene.csv`, matching the paper with random variance from sampling.

## Dataset

`data/` holds the two gene panels. Each `*_genes.csv` has one row per gene with
its `organism`, `accession`, full `genomic_sequence`, and `reference_protein`;
the proteins are also provided as FASTA (`*_ref.faa`).

| File | Contents |
|------|----------|
| `data/prokaryote_genes.csv` | 4 archaeal/bacterial genes; `cds_start` marks the coding start within the genomic window |
| `data/eukaryote_genes.csv` | 8 eukaryotic genes (genomic windows including introns) |
| `data/prokaryote_ref.faa`, `data/eukaryote_ref.faa` | reference proteins |

## Prompt construction

- **Prokaryote**: `prompt = [1000 nt upstream of the CDS] + [first 30% of the CDS]`
  (rounded to whole codons). The model completes the remaining ~70% of the CDS.
- **Eukaryote**: `prompt = first 30% of the genomic window`. The model completes
  the remaining genomic sequence, which is spliced for scoring.

See `prompts.py`. Generation uses `temperature=0.7` (prokaryote) / `1.0`
(eukaryote), `top_k=4`, and 50 generations per gene, matching the paper.

## Running the benchmark

**Install** Evo 2 (see the [repository README](../../README.md)) plus the
scoring dependencies:

```bash
pip install biopython            # protein alignment + translation
conda install -c bioconda exonerate   # required for the eukaryote panel only
```

**Run** a panel for any Evo 2 model:

```bash
cd scripts/gene_completion

# Prokaryote / archaea panel (no external tools needed)
python run_gene_completion.py --panel prokaryote --model_name evo2_7b \
    --output_dir out/evo2_7b

# Eukaryote panel (requires exonerate)
python run_gene_completion.py --panel eukaryote --model_name evo2_7b \
    --output_dir out/evo2_7b
```

Each run writes per-generation scores (`*_completions.csv`) and per-gene
summary statistics (`*_per_gene_stats.csv`). The overall headline number is the
mean of the per-gene means across both panels (4 + 8 = 12 genes).

To evaluate a **non-Evo 2 model**, generate completions for the prompts produced
by `prompts.py` and score them with `score_prokaryote` / `score_eukaryote` in
`scoring.py`.

## Files

```
scripts/gene_completion/
├── README.md
├── run_gene_completion.py        # main CLI: generate completions + score
├── prompts.py                    # prompt construction for each panel
├── scoring.py                    # translation, protein alignment, exonerate splicing
├── summarize.py                  # combine panels -> 12-gene headline number
├── data/
│   ├── prokaryote_genes.csv      # 4 genes: genomic + reference protein
│   ├── prokaryote_ref.faa
│   ├── eukaryote_genes.csv       # 8 genes: genomic + reference protein
│   └── eukaryote_ref.faa
└── results/
    ├── summary.csv               # per-model panel means + 12-gene headline
    ├── prokaryote_per_gene.csv
    └── eukaryote_per_gene.csv
```

## Citation

```
@article{Brixi2026,
    author  = {Brixi, Garyk and Durrant, Matthew G. and Ku, Jerome and others},
    title   = {Genome modelling and design across all domains of life with Evo 2},
    journal = {Nature},
    year    = {2026},
    doi     = {10.1038/s41586-026-10176-5},
    url     = {https://doi.org/10.1038/s41586-026-10176-5},
}
```
