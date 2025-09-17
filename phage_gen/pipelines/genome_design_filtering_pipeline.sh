#!/bin/bash
#SBATCH --job-name=genome_design_pipeline
#SBATCH --output=/path/to/phage_filter_%j.log
#SBATCH --error=/path/to/phage_filter_%j.err
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --partition=cpu_batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --ntasks-per-node=1
#SBATCH --mem=320G


### Run genome design filtering pipeline ###

# Requirements:
# Conda environment from genome_design.yaml, and set Conda environment in the Slurm script
# Conda environment from genome_visualization.yaml, and set the environment at lovis4u_conda_env in the config

# Usage:
# 1. Create a save directory for your results
# 2. Copy genome_design_filtering_pipeline_config_template.yaml to the results directory
# 3. Modify genome_design_filtering_pipeline_config_template.yaml as needed
# 4. Set up this Slurm script: job output, error output, and config file path
# 5. Run pipeline by: sbatch /path/to/genome_design_filtering_pipeline.sh


START_TIME=$(date +%s)
HOSTNAME=$(hostname)
echo "Running on hostname: $HOSTNAME"
eval "$(conda shell.bash hook)"


conda activate genome_design

### Set design constraints config (change to your config file) ###
CONFIG_FILE="/path/to/genome_design_filtering_pipeline_config_template.yaml"

### Run filtering pipeline (change to your path) ###
python /path/to/genome_design_filtering_pipeline.py $CONFIG_FILE


END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_TIME_HMS=$(printf '%02d:%02d:%02d\n' $((ELAPSED_TIME/3600)) $(((ELAPSED_TIME%3600)/60)) $((ELAPSED_TIME%60)))
echo "Elapsed time: ${ELAPSED_TIME_HMS}"
