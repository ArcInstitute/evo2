# Phage genome design

This directory contains useful scripts and data for phage genome design.

## Contents

```
./phage_gen/
├── analysis/                                                    # General analysis scripts
│   ├── competition_analysis.py                                  # Analyzing raw sequencing data from phage competition
│   ├── genome_gibson_assembly.py                                # Automated Gibson assembly fragment design from genome sequences
│   ├── plot_competition_analysis.py                             # Plotting phage competition results
├── data/                                                        # Useful data for running genome design filtering pipeline
│   ├── microviridae_genomes_NC_001422_1.fna                     # PhiX174 reference genome
│   ├── NC_001422.1_Gprotein.fasta                               # PhiX174 spike (G) protein
│   ├── NC_001422.1_pseudocircular.gff                           # PhiX174 pseudo-circularized and annotated using our method
├── environments/                                                # Conda environments
│   ├── competition_analysis.yaml                                # Environment for running phage competition analysis
│   ├── genome_design.yaml                                       # Environment for running genome design pipeline
│   ├── genome_visualization.yaml                                # Environment for visualizing genomes in genome design pipeline
├── pipelines/                                                   # Key genome design filtering and visualization scripts
│   ├── genetic_architecture_visualization.py                    # Helper script for visualizing genomes with lovis4u
│   ├── genetic_architecture.py                                  # Helper script for calculating architectural similarity scores
│   ├── genome_design_filtering_pipeline_config_template.yaml    # Main config for genome design filtering parameters
│   ├── genome_design_filtering_pipeline.py                      # Main pipeline script for genome design filteirng
│   ├── genome_design_filtering_pipeline.sh                      # Slurm script for launching genome design filtering pipeline
└── README.md                                                    # Important information for navigating repo
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
