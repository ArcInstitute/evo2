# Phage genome design

This directory contains scripts, data, and analysis pipelines for phage genome design.

## Contents

```
./phage_gen/
├── analysis/                                                    # Analysis scripts
│   ├── competition_analysis.py                                  # Phage competition sequencing analysis (QC, SNV-based read assignment, fold change, visualization)
│   ├── mutation_type_analysis.py                                # BLASTn + mutation classification (syn/nonsyn/indel/intergenic) per gene
│   ├── genome_annotator.py                                      # Genome annotation (ORF prediction with Prodigal/Orfipy, MMseqs2 search)
│   ├── genome_gibson_assembly.py                                # Automated Gibson assembly fragment design from genome sequences
│   ├── shannon_diversity_analysis.sh                            # Shannon diversity analysis of generated sequences
├── data/                                                        # Reference data and genome collections
│   ├── NC_001422_1.fna                                          # PhiX174 reference genome
│   ├── NC_001422.1_Gprotein.fasta                               # PhiX174 spike (G) protein
│   ├── NC_001422.1_pseudocircular.gff                           # PhiX174 pseudo-circularized annotation
│   ├── all_generated_phages.fasta                               # All AI-generated phage genomes
│   ├── viable_generated_phage_genomes.fasta                     # Viable (functional) AI-generated phage genomes
│   ├── nonviable_generated_phage_genomes.fasta                  # Nonviable AI-generated phage genomes
│   ├── rokyta2006_phix174like_genomes.fasta                     # PhiX174-like wild isolates (Rokyta et al. 2006)
│   ├── wichman2005_lt180_genomes.fasta                          # Lab-evolved PhiX174 genomes (Wichman et al. 2005)
│   ├── phage_sft_genomes_phix174_variants.fna                   # PhiX174 variant genomes used for SFT
├── environments/                                                # Conda environments
│   ├── competition_analysis.yaml                                # Environment for phage competition analysis
│   ├── genome_annotator.yaml                                    # Environment for genome annotation
│   ├── genome_design.yaml                                       # Environment for genome design pipeline
│   ├── genome_visualization.yaml                                # Environment for genome visualization
├── pipelines/                                                   # Genome design filtering and visualization
│   ├── genome_design_filtering_pipeline.py                      # Main pipeline script for genome design filtering
│   ├── genome_design_filtering_pipeline.sh                      # Slurm launch script
│   ├── genome_design_filtering_pipeline_config_template.yaml    # Config template for filtering parameters
│   ├── genetic_architecture.py                                  # Architectural similarity scoring
│   ├── genetic_architecture_visualization.py                    # Genome visualization with LoVis4u
└── README.md
```

## Related Documentation

For more comprehensive documentation, please refer to:
- The main repository README for setting up generation or fine-tuning with Evo

## Citation

Please cite the following publication when referencing phage design.

```
@article {king2025,
   author = {King, Samuel H and Driscoll, Claudia L and Li, David B and Guo, Daniel and Merchant, Aditi T and Brixi, Garyk and Wilkinson, Max E and Hie, Brian L},
   title = {Generative design of novel bacteriophages with genome language models},
   year = {2025},
   doi = {10.1101/2025.09.12.675911},
   publisher = {Cold Spring Harbor Laboratory},
   URL = {https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1},
   journal = {bioRxiv}
}
```
