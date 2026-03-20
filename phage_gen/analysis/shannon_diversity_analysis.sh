#!/bin/bash
#SBATCH --job-name=genome_design_pipeline
#SBATCH --output=/path/to/phage_analysis_%j.log
#SBATCH --error=/path/to/phage_analysis_%j.err
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --partition=cpu_batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --ntasks-per-node=1
#SBATCH --mem=320G

### Calculate Shannon diversity of fasta files after MMseqs ###

START_TIME=$(date +%s)
HOSTNAME=$(hostname)
echo "Running on hostname: $HOSTNAME"
eval "$(conda shell.bash hook)"


conda activate genome_design

##### USER PATHS #####
# Controls: arbitrary FASTA files
CONTROLS_DIR="/large_storage/hielab/samuelking/phage_design/generation/data/20250830_shannondiversity_analysis/data/controls"

# Evo1/Evo2 qc4 directories like evo1_temp03_0bp/, each with qc4_homology_filter_* files
ANALYSIS_DIR="/large_storage/hielab/samuelking/phage_design/generation/data/20250829_filtering_for_figure/analysis_up_to_archscoreremoval"

# Where to store per-run mmseqs DBs/results + the summary CSV
CLUSTER_ROOT="/large_storage/hielab/samuelking/phage_design/generation/data/20250830_shannondiversity_analysis/mmseqs_clustering"
SUMMARY_CSV="${CLUSTER_ROOT}/shannon_diversity_summary.csv"

# MMseqs threading
THREADS=64

##### ENV / SETUP #####
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate genome_design 2>/dev/null

# prevent odd MMseqs env causing errors
unset MMSEQS_CALL_DEPTH

mkdir -p "${CLUSTER_ROOT}"
echo "source_type,model,temp,bp,dir,filename,n_sequences,n_clusters,shannon_ln,shannon_bits,reason" > "${SUMMARY_CSV}"

timestamp () { date +"%Y-%m-%d %H:%M:%S"; }

validate_fasta () {
  local f="$1"
  [[ -s "$f" ]] || return 1
  grep -q '^>' "$f"
}

compute_shannon_from_tsv () {
  local tsv="$1"
  awk '
    {
      rep1[$1]++; rep2[$2]++; total++;
    }
    END {
      n1=0; for (k in rep1) n1++;
      n2=0; for (k in rep2) n2++;
      # choose the column with fewer uniques as cluster representatives
      if (n1 <= n2) {
        H=0; for (k in rep1) { p=rep1[k]/total; if (p>0) H+=-p*log(p) }
        printf "%d %.10f %.10f\n", n1, H, H/log(2);
      } else {
        H=0; for (k in rep2) { p=rep2[k]/total; if (p>0) H+=-p*log(p) }
        printf "%d %.10f %.10f\n", n2, H, H/log(2);
      }
    }
  ' "$tsv"
}

run_mmseqs_and_shannon () {
  local fasta="$1" outbase="$2"
  local db="${outbase}/mmseqs_db"
  local res="${outbase}/mmseqs_results"
  local tmp="${outbase}/tmp"
  local log="${outbase}/mmseqs.log"
  mkdir -p "$db" "$res" "$tmp"
  : > "$log"

  # reuse DB if present
  if [[ ! -s "${db}/sequences" ]]; then
    mmseqs createdb "$fasta" "${db}/sequences" >>"$log" 2>&1
  fi

  # cluster + membership tsv, send all chatter to log
  mmseqs cluster   "${db}/sequences" "${res}/clusters" "${tmp}" \
    --min-seq-id 0.99 --threads "${THREADS}"            >>"$log" 2>&1
  mmseqs createtsv "${db}/sequences" "${db}/sequences" \
    "${res}/clusters" "${res}/clusters.tsv"             >>"$log" 2>&1

  # only echo the three numbers
  if [[ -s "${res}/clusters.tsv" ]]; then
    compute_shannon_from_tsv "${res}/clusters.tsv"
  else
    echo "0 0 0"
  fi
}

echo "[$(timestamp)] Starting Shannon diversity aggregation..."

##### 1) CONTROLS #####
if [[ -d "${CONTROLS_DIR}" ]]; then
  shopt -s nullglob
  for fasta in "${CONTROLS_DIR}"/*.fa "${CONTROLS_DIR}"/*.fna "${CONTROLS_DIR}"/*.fasta; do
    [[ -e "$fasta" ]] || continue
    base="$(basename "$fasta")"
    dir_name="$(basename "$(dirname "$fasta")")"
    source_type="controls"
    model="controls"; temp="NA"; bp="NA"
    outtag="${source_type}_${base%.*}"
    outbase="${CLUSTER_ROOT}/${outtag}"

    if ! validate_fasta "$fasta"; then
      echo "[$(timestamp)] controls invalid FASTA → ${base}"
      echo "${source_type},${model},${temp},${bp},${dir_name},${base},0,0,0,0,invalid_fasta" >> "${SUMMARY_CSV}"
      continue
    fi

    nseq=$(grep -c '^>' "$fasta" 2>/dev/null || echo 0)

    if [[ -s "${outbase}/mmseqs_results/clusters.tsv" ]]; then
      read -r ncl hnat hbit <<< "$(compute_shannon_from_tsv "${outbase}/mmseqs_results/clusters.tsv")"
      echo "${source_type},${model},${temp},${bp},${dir_name},${base},${nseq},${ncl},${hnat},${hbit},cached" >> "${SUMMARY_CSV}"
      continue
    fi

    read -r ncl hnat hbit <<< "$(run_mmseqs_and_shannon "$fasta" "$outbase")"
    echo "${source_type},${model},${temp},${bp},${dir_name},${base},${nseq},${ncl},${hnat},${hbit},ok" >> "${SUMMARY_CSV}"
  done
fi

##### 2) EVO qc4 FASTAS #####
if [[ -d "${ANALYSIS_DIR}" ]]; then
  for subdir in "${ANALYSIS_DIR}"/evo*_temp*_*bp; do
    [[ -d "$subdir" ]] || continue
    run_name="$(basename "$subdir")"      # e.g., evo1_temp03_0bp
    model="${run_name%%_*}"
    temp="$(echo "$run_name" | grep -o 'temp[0-9]\+')"
    bp="$(echo "$run_name" | grep -o '[0-9]\+bp' | sed 's/bp//')"; [[ -z "$bp" ]] && bp="NA"

    counts_csv="${subdir}/qc4_homology_filter_counts.csv"
    fasta="${subdir}/qc4_homology_filter_seqs.fasta"
    base="$(basename "$fasta")"
    dir_name="${run_name}"
    source_type="evo_qc4"
    outtag="${model}_${temp}_${bp}bp_qc4"
    outbase="${CLUSTER_ROOT}/${outtag}"

    # Is the tropism column present?
    tropism_present=false
    if [[ -f "$counts_csv" ]]; then
      if head -1 "$counts_csv" | tr -d '\r' | awk -F',' '{for(i=1;i<=NF;i++) if($i=="count_tropism_protein_sequence_identity_filter") f=1} END{exit !f}'; then
        tropism_present=true
      fi
    fi

    if [[ "$tropism_present" != true ]]; then
      echo "[$(timestamp)] ${run_name}: missing tropism column → Shannon=0"
      echo "${source_type},${model},${temp},${bp},${dir_name},${base},0,0,0,0,missing_tropism_col" >> "${SUMMARY_CSV}"
      continue
    fi

    if [[ ! -s "$fasta" ]] || ! validate_fasta "$fasta"; then
      echo "[$(timestamp)] ${run_name}: qc4 fasta missing/invalid → Shannon=0"
      echo "${source_type},${model},${temp},${bp},${dir_name},${base},0,0,0,0,invalid_or_empty_fasta" >> "${SUMMARY_CSV}"
      continue
    fi

    nseq=$(grep -c '^>' "$fasta" 2>/dev/null || echo 0)

    if [[ -s "${outbase}/mmseqs_results/clusters.tsv" ]]; then
      read -r ncl hnat hbit <<< "$(compute_shannon_from_tsv "${outbase}/mmseqs_results/clusters.tsv")"
      echo "${source_type},${model},${temp},${bp},${dir_name},${base},${nseq},${ncl},${hnat},${hbit},cached" >> "${SUMMARY_CSV}"
      continue
    fi

    read -r ncl hnat hbit <<< "$(run_mmseqs_and_shannon "$fasta" "$outbase")"
    echo "${source_type},${model},${temp},${bp},${dir_name},${base},${nseq},${ncl},${hnat},${hbit},ok" >> "${SUMMARY_CSV}"
  done
fi

echo "[$(timestamp)] Done. Summary: ${SUMMARY_CSV}"


END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_TIME_HMS=$(printf '%02d:%02d:%02d\n' $((ELAPSED_TIME/3600)) $(((ELAPSED_TIME%3600)/60)) $((ELAPSED_TIME%60)))
echo "Elapsed time: ${ELAPSED_TIME_HMS}"