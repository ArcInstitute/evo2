"""
BACTERIOPHAGE CDS PREDICTION AND GENOME ANNOTATION

This script is for running bacteriophage genome annotation independently of the genome design pipeline.
The individual environment requirements are provided in environments/genome_annotator.yaml.

Predicts ORFs in (optionally pseudo-circularized) phage genomes using Prodigal or Orfipy.
Searches predicted ORFs against a given protein database with MMseqs2.
Keeps ORFs with significant MMseqs2 hits.
Optionally visualizes annotated genome with LoVis4u.

### Required arguments
-i --input                  Input FASTA filepath
-d --database               Protein database filepath to search predicted ORFs against
-o --output                 Output results directory filepath

### General arguments
--orf-caller                ORF caller to use (choose one of orfipy (default), prodigal, pyrodigal-gv)
--skip-circularization      Skip pseudo-circularization of genome (will not capture wrap-around ORFs)

### Orfipy-specific arguments
--start-codons              Codons to use as start codons for ORF predictions (default is ATG)
--stop-codons               Codons to use as stop codons for ORF predictions (default is TAA,TAG,TGA)
--strand                    DNA strand to use for ORF calling (sense (f), antisense (r), or both (b; default))
--min-orf-length            Minimum length of predicted ORFs (default is 90)
--max-orf-length            Maximum length of predicted ORFs (default is 1800)
--annotation-file           TSV filepath for annotations matching proteins in given protein database

### MMseqs2-specific arguments
-s --sensitivity            MMseqs2 sensitivity (default is 4.0)
-t --threads                Number of threads to use for MMseqs2 search (default is 8)
-e --e-value                E-value cutoff (default is 1e-3)

### LoVis4u-specific arguments
--visualize                 Visualize genome annotations with LoVis4u
--reference-genome          Genome GFF filepath to visualize synteny against with LoVis4u
--viz-workers               Number of parallel workers for creating LoVis4u genome visualizations
--create-gbk                Create gbk files for annotated genomes


Examples:

python genome_annotator.py \
    -i /large_storage/hielab/samuelking/phage_design/generation/data/20251101_genome_annotator_tests/phix174.fna \
        -d /large_storage/hielab/samuelking/phrogs/phrogs_mmseqs_db/phrogs_mmseqs_db \
            -o my_results/ \
                --orf-caller orfipy \
                    --skip-circularization \
                        --start-codons ATG \
                            --stop-codons TAA,TAG,TGA \
                                --strand f \
                                    --min-orf-length 90 \
                                        --max-orf-length 1800 \
                                            -s 4.0 \
                                                -t 8 \
                                                    --annotation-file /large_storage/hielab/samuelking/phrogs/phrog_annot_v4_sk2.tsv \
                                                        --visualize \
                                                            --reference-genome /large_storage/hielab/samuelking/phage_design/data/phix174_only/NC_001422.1_pseudocircular.gff \
                                                                --viz-workers 8 \
                                                                    --create-gbk

python genome_annotator.py \
    -i /large_storage/hielab/samuelking/phage_design/generation/data/20251101_genome_annotator_tests/phix174.fna \
        -d /large_storage/hielab/samuelking/phrogs/phrogs_mmseqs_db/phrogs_mmseqs_db \
            -o my_results3/ \
                --orf-caller prodigal \
                    --visualize \
                        --annotation-file /large_storage/hielab/samuelking/phrogs/phrog_annot_v4_sk2.tsv \
                            --create-gbk

"""

import sys
import os
import re
import tempfile
import subprocess
import shutil
import argparse
import time
import pandas as pd
import numpy as np
import concurrent.futures
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation


def mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, 
                          threads: int = 8, split: int = 0, sensitivity: float = 4.0, e_value: float=1e-3) -> str:
    """Run MMseqs2 search on the input protein fasta file against the given MMseqs database."""
    os.makedirs(results_dir, exist_ok=True)
    mmseqs_out = os.path.join(results_dir, "mmseqs_result.m8")
    log_file = os.path.join(results_dir, "mmseqs_search.log")

    # Run MMseqs search
    cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} -e {e_value} --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    print(f"  Running command: {cmd}")
    start_time = time.time()
    try:
        with open(log_file, "w") as log:
            result = subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=log, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  MMseqs2 search failed with error: {e.stderr}")
        raise e
    end_time = time.time()
    print(f"  MMseqs2 search completed in {end_time - start_time:.2f} seconds.")
    
    if not os.path.isfile(mmseqs_out):
        print(f"  Output file not found: {mmseqs_out}")
    return mmseqs_out


def assign_numeric_genome_ids(fasta_file: str, query_genomes: set) -> dict:
    """Assigns numeric IDs to each unique genome in the FASTA file that matches the genomes in query_genomes."""
    genome_id_map = {}
    next_id = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_name = record.id
        if genome_name in query_genomes and genome_name not in genome_id_map:
            genome_id_map[genome_name] = f"genome_{next_id}"
            next_id += 1
    return genome_id_map


def extract_orf_positions_from_protein_database_hits(
    genomes_fasta_file: str, 
    orfs_fasta_file: str, 
    protein_database_hits_df: pd.DataFrame, 
    genome_id_map: dict
) -> dict:
    """
    Parses the ORFs FASTA file, extracts ORF positions and matching headers of genomes using a protein database hits dataframe,
    and maps them to their corresponding genome IDs using the genome ID map. Adds a GFF entry for the original whole genome length
    using the genomes fasta file (should be the non-pseudo-circularized genomes fasta).
    
    Handles both Prodigal and orfipy ORF formats.
    
    Args:
        genomes_fasta_file: Original (non-pseudocircular) genomes FASTA file
        orfs_fasta_file: ORFs predicted from pseudocircular genomes
        protein_database_hits_df: DataFrame with MMseqs hits
        genome_id_map: Mapping of genome names to numeric IDs
    
    Returns:
        Dictionary with ORF data for each genome
    """
    orf_data = {}
    protein_database_hits_dict = protein_database_hits_df.set_index('id_prompt')[['sequence', 'protein_database_mmseqs_target', 'protein_database_mmseqs_percent_identity', 'annot', 'category']].to_dict('index')
    
    # Process genome lengths from the genomes_fasta_file and get list of genome names
    genome_lengths = {}
    genome_names_set = set()
    for record in SeqIO.parse(genomes_fasta_file, "fasta"):
        genome_name = record.id
        genome_length = len(record.seq)
        genome_lengths[genome_name] = genome_length
        genome_names_set.add(genome_name)

    # Process ORF data
    for record in SeqIO.parse(orfs_fasta_file, "fasta"):
        if record.id not in protein_database_hits_dict:
            continue
            
        header = record.description
        
        # Detect format: orfipy has '_ORF' in ID, Prodigal doesn't
        if '_ORF' in record.id:
            # Orfipy format: genome_name_ORF.#
            genome_name = record.id.split('_ORF')[0]
            
            # Extract the 'ORF.#' part from the record ID
            match = re.search(r'ORF\.\d+', record.id)
            orf_id = match.group(0) if match else record.id
            
            # Extract start and end positions from header: [start-end](+/-)
            match = re.search(r'\[(\d+)-(\d+)\]', header)
            if not match:
                continue
            start, end = match.groups()
            
            # Extract strand
            strand_match = re.search(r'\[\d+-\d+\]\((\+|\-)\)', header)
            strand = strand_match.group(1) if strand_match else '+'
            
        else:
            # Prodigal format: genome_name_orf# # start # end # strand
            # Find genome name by matching against known genome names
            genome_name = None
            for gname in genome_names_set:
                if record.id.startswith(gname + '_'):
                    genome_name = gname
                    break
            
            if genome_name is None:
                # Couldn't find matching genome name
                continue
            
            # Extract ORF number from the end of the ID
            orf_suffix = record.id[len(genome_name)+1:]  # Everything after "genome_name_"
            orf_id = f"ORF_{orf_suffix}"
            
            # Parse Prodigal header: ID # start # end # strand # attributes
            header_parts = header.split(' # ')
            if len(header_parts) < 4:
                continue
                
            try:
                start = header_parts[1]
                end = header_parts[2]
                strand_val = header_parts[3]
                strand = '+' if strand_val == '1' else '-'
            except (IndexError, ValueError):
                continue

        # Only process ORFs for genomes in the ID map
        if genome_name not in genome_id_map:
            continue
            
        genome_id = genome_id_map[genome_name]
        
        # Retrieve function and product information
        sequence_value = protein_database_hits_dict[record.id]['sequence']
        percent_identity_value = protein_database_hits_dict[record.id]['protein_database_mmseqs_percent_identity']
        function_value = protein_database_hits_dict[record.id]['category']
        product_value = protein_database_hits_dict[record.id]['annot']
        
        orf_entry = {
            'seq_id': genome_id,
            'feature_type': 'CDS',
            'start': start,
            'end': end,
            'score': '.',
            'strand': strand,
            'phase': '0',
            'attributes': f"ID={orf_id};function={function_value};product={product_value};seq={sequence_value};percent_identity={percent_identity_value}"
        }
        
        # Append ORF entry to the corresponding genome ID
        orf_data.setdefault(genome_id, []).append(orf_entry)

    # Add genome length entries to ORF data
    for genome_name, genome_length in genome_lengths.items():
        if genome_name in genome_id_map:
            genome_id = genome_id_map[genome_name]
            genome_entry = {
                'seq_id': genome_id,
                'feature_type': 'region',
                'start': 1,
                'end': genome_length,
                'score': '.',
                'strand': '+',
                'phase': '.',
                'attributes': f"ID={genome_id};length={genome_length}"
            }
            # Append genome length entry to the corresponding genome ID
            orf_data.setdefault(genome_id, []).append(genome_entry)

    return orf_data


def parse_mmseqs_results(mmseqs_out):
    """Parse MMseqs search results including percent identity."""
    if not os.path.isfile(mmseqs_out):
        raise FileNotFoundError(f"Output file not found: {mmseqs_out}")
    
    hits = []
    with open(mmseqs_out, "r") as f:
        for line in f:
            query, target, evalue, pident = line.strip().split('\t')
            hits.append((query, target, float(evalue), float(pident)))
    return hits


def mmseqs_results_to_df(hits, query_fasta: str, output_csv: str, descriptive_prefix: str, only_top_hits: bool=True) -> pd.DataFrame:
    """Write the hits and corresponding sequences to a CSV file."""
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(query_fasta, "fasta")}
    
    data = []
    for query, target, evalue, pident in hits:
        if query in sequences:
            data.append([query, sequences[query], target, evalue, pident])
    
    df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", 
                                     f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])
    if only_top_hits:
        df_top_hits = df.loc[df.groupby("id_prompt")[f"{descriptive_prefix}_mmseqs_e_value"].idxmin()]
        df_top_hits.to_csv(output_csv, index=False)
        return df_top_hits
    else:
        df.to_csv(output_csv, index=False)
        return df


def run_mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str,
                               output_csv: str, descriptive_prefix: str,
                               threads: int=8, split: int=0, sensitivity: float=4.0,
                               e_value: float=1e-3, only_top_hits: bool=True) -> pd.DataFrame:
    """Run MMseqs search with the specified directories and files."""
    if not os.path.isfile(query_fasta):
        raise FileNotFoundError(f"FASTA file not found: {query_fasta}")
    if not os.path.isdir(mmseqs_db) and not os.path.isfile(mmseqs_db):
        raise FileNotFoundError(f"MMseqs database not found: {mmseqs_db}")
    
    mmseqs_out = mmseqs_search_proteins(query_fasta, mmseqs_db, results_dir, threads, split, sensitivity, e_value)
    hits = parse_mmseqs_results(mmseqs_out)
    df = mmseqs_results_to_df(hits, query_fasta, output_csv, descriptive_prefix, only_top_hits)
    return df


def append_upstream_of_last_frame_stop(input_fasta: str, output_fasta: str) -> None:
    """
    Pseudo-circularize sequences by appending upstream sequence before the last frame stop codon.
    """
    def find_last_frame_stop(seq: Seq) -> int:
        """Find the first stop codon in each reading frame and return the furthest one."""
        stop_codons = ['TAA', 'TAG', 'TGA']
        first_stops = []

        for frame in range(3):
            sub_seq = seq[frame:]
            for i in range(0, len(sub_seq) - 2, 3):
                codon = str(sub_seq[i:i + 3])
                if codon in stop_codons:
                    first_stops.append(i + frame + 3)
                    break

        return max(first_stops) if first_stops else len(seq)

    records = []
    counter = 0
    
    with open(input_fasta, "r") as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            seq = record.seq
            last_stop = find_last_frame_stop(seq)
            new_seq = seq + seq[:last_stop]
            
            new_record = SeqRecord(new_seq, id=record.id, description=record.description)
            records.append(new_record)
            
            counter += 1
            if counter % 1000 == 0:
                print(f"  Pseudo-circularized {counter} sequences")
        
    with open(output_fasta, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    
    print(f"  Completed pseudo-circularization of {counter} sequences")


def run_prodigal(input_sequences: str, output_orf_file: str, output_protein_file: str) -> None:
    """Run Prodigal, log stdout/stderr + runtime."""
    log_path = Path(output_orf_file).with_suffix(".log")
    cmd = [
        "prodigal",
        "-i", input_sequences,
        "-d", output_orf_file,
        "-a", output_protein_file,
        "-p", "meta",
    ]

    start_time = time.time()
    with open(log_path, "w") as log:
        log.write(f"=== Prodigal started ===\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()

        subprocess.run(cmd, stdout=log, stderr=log, check=True)

        elapsed = time.time() - start_time
        log.write(f"\n=== Prodigal finished ===\n")
        log.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")

def run_pyrodigal_gv(input_sequences: str, output_orf_file: str, output_protein_file: str) -> None:
    """Run Pyrodigal-gv, log stdout/stderr + runtime."""
    log_path = Path(output_orf_file).with_suffix(".log")
    cmd = [
        "pyrodigal-gv",
        "-i", input_sequences,
        "-d", output_orf_file,
        "-a", output_protein_file,
        "-p", "meta",
    ]

    start_time = time.time()
    with open(log_path, "w") as log:
        log.write(f"=== Pyrodigal-gv started ===\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()

        subprocess.run(cmd, stdout=log, stderr=log, check=True)

        elapsed = time.time() - start_time
        log.write(f"\n=== Pyrodigal-gv finished ===\n")
        log.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")


def clean_orfipy_fasta_file(input_fasta: str, output_fasta: str) -> None:
    """Remove * symbols from protein sequences in given fasta file."""
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        for line in infile:
            if line.startswith(">"):
                outfile.write(line)
            else:
                outfile.write(line.replace('*', ''))


def run_orfipy(input_fasta: str, threads: int, start_codons: str, stop_codons: str, 
              strand: str, min_len: int, max_len: int, output_dir: str, 
              output_nt: str, output_aa_tmp: str, output_aa: str) -> None:
    """Call ORFs using Orfipy and save nucleotide and cleaned amino acid sequences."""
    orfipy_command = [
        'orfipy', '--procs', str(threads), input_fasta, '--start', start_codons, 
        '--stop', stop_codons, '--strand', strand, '--include-stop', '--min', str(min_len), 
        '--max', str(max_len), '--outdir', output_dir, '--dna', output_nt, '--pep', output_aa_tmp
    ]
    subprocess.run(orfipy_command, check=True)
    
    clean_orfipy_fasta_file(f'{output_dir}/{output_aa_tmp}', f'{output_dir}/{output_aa}')
    os.remove(f'{output_dir}/{output_aa_tmp}')


def annotate_protein_hits_deprecated(mmseqs_hits_file: str, database_annotation_file: str) -> pd.DataFrame:
    """Add protein database annotations to MMseqs hits."""
    mmseqs_hits_df = pd.read_csv(mmseqs_hits_file)
    database_annotation_df = pd.read_csv(database_annotation_file, sep='\t')

    mmseqs_hits_df['protein_database_mmseqs_target'] = mmseqs_hits_df['protein_database_mmseqs_target'].astype(str)
    database_annotation_df['protein_database_mmseqs_target'] = database_annotation_df['hit_label'].astype(str)

    mmseqs_hits_df = mmseqs_hits_df.merge(
        database_annotation_df[['protein_database_mmseqs_target', 'annot', 'category']],
        on='protein_database_mmseqs_target',
        how='left'
    )

    return mmseqs_hits_df


def annotate_protein_hits(mmseqs_hits_file: str, database_annotation_file: str) -> pd.DataFrame:
    """Add protein database annotations to MMseqs hits."""
    mmseqs_hits_df = pd.read_csv(mmseqs_hits_file)
    database_annotation_df = pd.read_csv(database_annotation_file, sep='\t')

    mmseqs_hits_df['protein_database_mmseqs_target'] = mmseqs_hits_df['protein_database_mmseqs_target'].astype(str)
    database_annotation_df['protein_database_mmseqs_target'] = database_annotation_df['hit_label'].astype(str)

    # Update: De-duplicate any hit labels in the input annotation dataframe
    database_annotation_df = database_annotation_df.drop_duplicates(
        subset='protein_database_mmseqs_target',
        keep='first'
    )

    mmseqs_hits_df = mmseqs_hits_df.merge(
        database_annotation_df[['protein_database_mmseqs_target', 'annot', 'category']],
        on='protein_database_mmseqs_target',
        how='left'
    )

    return mmseqs_hits_df


def create_gff_file(orf_data: dict, genome_name: str, genome_id: str, 
                   genome_seq: str, output_dir: str) -> None:
    """Write a .gff file for each genome."""
    output_gff = os.path.join(output_dir, f"{genome_id}.gff")

    with open(output_gff, 'w') as gff:
        gff.write("##gff-version 3\n")
        gff.write(f"##sequence-region {genome_id} 1 {len(genome_seq)}\n")
        gff.write(f"##description {genome_name}\n")

        if genome_id in orf_data:
            for orf in orf_data[genome_id]:
                gff_line = (
                    f"{orf['seq_id']}\tPredicted genome annotation\t{orf['feature_type']}\t"
                    f"{orf['start']}\t{orf['end']}\t{orf['score']}\t{orf['strand']}\t"
                    f"{orf['phase']}\t{orf['attributes']}\n"
                )
                gff.write(gff_line)

        gff.write("##FASTA\n")
        gff.write(f">{genome_id}\n")
        gff.write(f"{genome_seq}\n")


def batch_create_gff_files(genomes_fasta_file: str, pseudocircular_genomes_fasta_file: str,
                           pseudocircular_orfs_fasta_file: str, protein_database_hits_df: pd.DataFrame,
                           output_dir: str) -> dict:
    """
    Create GFF files for all genomes with ORF annotations.
    
    Args:
        genomes_fasta_file: Original (non-pseudocircular) genomes FASTA file
        pseudocircular_genomes_fasta_file: Pseudocircular genomes FASTA file
        pseudocircular_orfs_fasta_file: ORFs predicted from pseudocircular genomes
        protein_database_hits_df: DataFrame with MMseqs hits
        output_dir: Directory to save GFF files
    
    Returns:
        Dictionary mapping genome names to numeric IDs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all genome names from the original fasta
    genome_names_set = {record.id for record in SeqIO.parse(genomes_fasta_file, "fasta")}
    
    # Extract genome names from ORF IDs by matching against known genome names
    matching_queries = set()
    for orf_id in protein_database_hits_df['id_prompt']:
        if '_ORF' in orf_id:
            # Orfipy format: genome_name_ORF.#
            genome_name = orf_id.split('_ORF')[0]
            matching_queries.add(genome_name)
        else:
            # Prodigal format: find matching genome name prefix
            for gname in genome_names_set:
                if orf_id.startswith(gname + '_'):
                    matching_queries.add(gname)
                    break
    
    genome_id_map = assign_numeric_genome_ids(genomes_fasta_file, matching_queries)
    orf_data = extract_orf_positions_from_protein_database_hits(
        genomes_fasta_file, pseudocircular_orfs_fasta_file, protein_database_hits_df, genome_id_map
    )

    for genome_record in SeqIO.parse(pseudocircular_genomes_fasta_file, "fasta"):
        genome_name = genome_record.id
        if genome_name in genome_id_map:
            genome_id = genome_id_map[genome_name]
            genome_seq = str(genome_record.seq)
            create_gff_file(orf_data, genome_name, genome_id, genome_seq, output_dir)
    
    return genome_id_map


def parse_gff_attributes(attributes_str: str) -> dict:
    """Parse the attributes field in a GFF line."""
    attributes = {}
    for attribute in attributes_str.split(';'):
        key_value = attribute.strip().split('=', 1)
        if len(key_value) == 2:
            key, value = key_value
            attributes[key] = value
    return attributes


def convert_gff_to_gbk(gff_file: str, output_gbk: str) -> None:
    """Convert a GFF file (with embedded FASTA) to GenBank format."""
    features = []
    fasta_sequence = None
    seq_id = None

    with open(gff_file) as gff_handle:
        for line in gff_handle:
            if line.startswith("##sequence-region"):
                seq_id = line.split()[1]
            elif line.startswith("##FASTA"):
                break
            elif not line.startswith("#") and line.strip():
                columns = line.strip().split("\t")
                
                feature_type = columns[2]
                start = int(columns[3]) - 1  # Convert to 0-based
                end = int(columns[4])
                strand = 1 if columns[6] == '+' else -1
                attributes_str = columns[8]

                attributes = parse_gff_attributes(attributes_str)
                
                feature = SeqFeature(
                    location=FeatureLocation(start, end, strand=strand),
                    type=feature_type,
                    qualifiers={k: v for k, v in attributes.items() if k != 'seq'}
                )
                
                if 'seq' in attributes:
                    feature.qualifiers["translation"] = attributes['seq']
                features.append(feature)
        
        fasta_sequence = next(SeqIO.parse(gff_handle, "fasta"))
    
    seq_record = SeqRecord(
        fasta_sequence.seq,
        id=seq_id,
        name=seq_id,
        description=fasta_sequence.description,
        annotations={"molecule_type": "DNA"}
    )
    seq_record.features.extend(features)
    
    SeqIO.write(seq_record, output_gbk, "genbank")


def batch_convert_gff_to_gbk(input_dir: str, output_dir: str) -> None:
    """Convert all GFF files in a directory to GenBank format."""
    os.makedirs(output_dir, exist_ok=True)

    for gff_file in sorted(os.listdir(input_dir)):
        if gff_file.endswith(".gff"):
            input_gff_path = os.path.join(input_dir, gff_file)
            output_gbk_path = os.path.join(output_dir, f"{os.path.splitext(gff_file)[0]}.gbk")
            convert_gff_to_gbk(input_gff_path, output_gbk_path)


def run_lovis4u(input_gff_dir: str, output_dir: str) -> tuple:
    """Run lovis4u to visually annotate/compare genomes."""
    command = [
        'lovis4u', '-gff', input_gff_dir, '-hl', '--set-category-colour',
        '-c', 'A4p2', '-o', output_dir, '-alip'
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running lovis4u: {result.stderr}")
    
    return result.returncode, output_dir


def process_single_genome_lovis4u(gff_file: str, query_gff_dir: str, 
                                  reference_genome_gff: str, output_results_dir: str) -> tuple:
    """Process a single genome file with lovis4u."""
    start_time = time.time()
    
    try:
        gff_path = os.path.join(query_gff_dir, gff_file)
        genome_name = gff_file.replace(".gff", "")
        
        pid = os.getpid()
        temp_dir = Path(query_gff_dir) / f"temp_{genome_name}_{pid}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        shutil.copy(gff_path, temp_dir)
        if reference_genome_gff and os.path.exists(reference_genome_gff):
            shutil.copy(reference_genome_gff, temp_dir)
        
        output_lovis4u_dir = os.path.join(output_results_dir, genome_name)
        return_code, _ = run_lovis4u(str(temp_dir), output_lovis4u_dir)
        
        shutil.rmtree(temp_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return genome_name, return_code, processing_time
    
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Error processing {gff_file}: {str(e)}")
        return gff_file, 1, processing_time


def run_lovis4u_batch(query_gff_dir: str, reference_genome_gff: str, 
                     output_results_dir: str, max_workers: int = None) -> None:
    """Run lovis4u on multiple genomes in parallel."""
    os.makedirs(output_results_dir, exist_ok=True)

    gff_files = sorted(
        [f for f in os.listdir(query_gff_dir) if f.startswith("genome_") and f.endswith(".gff")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    total_files = len(gff_files)
    print(f"  Found {total_files} genome files to visualize")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_genome_lovis4u, gff_file, query_gff_dir, 
                          reference_genome_gff, output_results_dir)
            for gff_file in gff_files
        ]
        
        successful, failed = 0, 0
        for future in concurrent.futures.as_completed(futures):
            _, return_code, _ = future.result()
            if return_code == 0:
                successful += 1
            else:
                failed += 1
    
    print(f"  Visualization complete. Successful: {successful}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Bacteriophage CDS Annotation: Predict genes and annotate genomes\n\nNote: This script will overwrite existing output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Minimal example (orfipy + MMseqs):
  python genome_annotator.py -i genomes.fasta -d protein_db -o results/

  # Using Prodigal with visualization:
  python genome_annotator.py -i genomes.fasta -d protein_db -o results/ \\
    --orf-caller prodigal --visualize

  # Complete example with all flags:
  python genome_annotator.py \\
    -i input_genomes.fasta \\
    -d /path/to/mmseqs_protein_database \\
    -o my_results/ \\
    --orf-caller orfipy \\
    --skip-circularization \\
    --start-codons ATG,GTG,TTG \\
    --stop-codons TAA,TAG,TGA \\
    --strand b \\
    --min-orf-length 150 \\
    --max-orf-length 50000 \\
    -s 7.5 \\
    -t 32 \\
    --annotation-file phrogs_annotations.tsv \\
    --visualize \\
    --reference-genome reference.gff \\
    --viz-workers 8 \\
    --create-gbk

Note: Running the script multiple times with the same output directory will overwrite all previous results.
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, 
                       help='Input genome FASTA file')
    parser.add_argument('-d', '--database', required=True,
                       help='Path to MMseqs2 protein database')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for results')
    
    # ORF prediction method
    parser.add_argument('--orf-caller', choices=['prodigal', 'orfipy', 'pyrodigal-gv'], default='orfipy',
                       help='ORF prediction tool to use (default: orfipy)')
    parser.add_argument('--skip-circularization', action='store_true',
                       help='Skip pseudo-circularization of genomes (use original sequences)')
    
    # Orfipy-specific parameters
    parser.add_argument('--start-codons', default='ATG',
                       help='Start codons for orfipy (default: ATG)')
    parser.add_argument('--stop-codons', default='TAA,TAG,TGA',
                       help='Stop codons for orfipy (default: TAA,TAG,TGA)')
    parser.add_argument('--strand', choices=['f', 'r', 'b'], default='b',
                       help='Strand for orfipy: f=forward, r=reverse, b=both (default: b)')
    parser.add_argument('--min-orf-length', type=int, default=90,
                       help='Minimum ORF length for orfipy (default: 90)')
    parser.add_argument('--max-orf-length', type=int, default=1800,
                       help='Maximum ORF length for orfipy (default: 1800)')
    
    # MMseqs parameters
    parser.add_argument('-s', '--sensitivity', type=float, default=4.0,
                       help='MMseqs2 sensitivity (default: 4.0)')
    parser.add_argument('-t', '--threads', type=int, default=8,
                       help='Number of threads (default: 8)')
    parser.add_argument('-e', '--e-value', type=float, default=1e-3,
                    help='E-value cutoff (default: 1e-3)')
    
    # Annotation parameters
    parser.add_argument('--annotation-file', 
                       help='TSV file with database annotations (columns: hit_label, annot, category)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate genome architecture visualizations with lovis4u')
    parser.add_argument('--reference-genome',
                       help='Reference genome GFF file for lovis4u comparison')
    parser.add_argument('--viz-workers', type=int, default=4,
                       help='Number of parallel workers for visualization (default: 4)')
    
    # Output format
    parser.add_argument('--create-gbk', action='store_true',
                       help='Also create GenBank files from GFF files')
    
    args = parser.parse_args()
    
    # Convert input paths to absolute paths
    input_fasta = Path(args.input).resolve()
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {args.input}")
    
    database_path = Path(args.database).resolve()
    if not database_path.exists():
        raise FileNotFoundError(f"MMseqs database not found: {args.database}")
    
    # Handle optional paths
    annotation_file = None
    if args.annotation_file:
        annotation_file = Path(args.annotation_file).resolve()
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")
    
    reference_genome = None
    if args.reference_genome:
        reference_genome = Path(args.reference_genome).resolve()
        if not reference_genome.exists():
            raise FileNotFoundError(f"Reference genome file not found: {args.reference_genome}")
    
    # Create output directory structure
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pseudocircular_dir = output_dir / "pseudocircular_genomes"
    orfs_dir = output_dir / "orfs"
    mmseqs_dir = output_dir / "mmseqs_results"
    gff_dir = output_dir / "gff_files"
    
    # Note: All operations will overwrite existing files
    for d in [pseudocircular_dir, orfs_dir, mmseqs_dir, gff_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    print("=" * 70)
    print("BACTERIOPHAGE CDS ANNOTATION")
    print("=" * 70)
    
    # Step 1: Pseudo-circularize genomes (optional)
    if not args.skip_circularization:
        print("\n[1/6] Pseudo-circularizing genomes...")
        pseudocircular_fasta = pseudocircular_dir / "pseudocircular_genomes.fasta"
        append_upstream_of_last_frame_stop(str(input_fasta), str(pseudocircular_fasta))
        input_for_orfs = str(pseudocircular_fasta)
        pseudocircular_fasta_for_gff = str(pseudocircular_fasta)
    else:
        print("\n[1/6] Skipping pseudo-circularization (using original sequences)")
        input_for_orfs = str(input_fasta)
        pseudocircular_fasta_for_gff = str(input_fasta)
    
    # Step 2: Predict ORFs
    print(f"\n[2/6] Predicting ORFs using {args.orf_caller}...")
    
    if args.orf_caller == 'prodigal':
        orf_nt = orfs_dir / "orfs.fasta"
        orf_aa = orfs_dir / "proteins.fasta"
        run_prodigal(input_for_orfs, str(orf_nt), str(orf_aa))
        protein_fasta = str(orf_aa)
    elif args.orf_caller == 'pyrodigal-gv':
        orf_nt = orfs_dir / "orfs.fasta"
        orf_aa = orfs_dir / "proteins.fasta"
        run_pyrodigal_gv(input_for_orfs, str(orf_nt), str(orf_aa))
        protein_fasta = str(orf_aa)
    else:  # orfipy
        orf_nt = "orfs_nt.fasta"
        orf_aa_tmp = "orfs_aa_tmp.fasta"
        orf_aa = "orfs_aa.fasta"
        run_orfipy(
            input_for_orfs, args.threads, args.start_codons, args.stop_codons,
            args.strand, args.min_orf_length, args.max_orf_length,
            str(orfs_dir), orf_nt, orf_aa_tmp, orf_aa
        )
        protein_fasta = str(orfs_dir / orf_aa)
    
    print(f"  ORF prediction complete. Proteins saved to: {protein_fasta}")
    
    # Step 3: Run MMseqs2 search
    print("\n[3/6] Running MMseqs2 search against protein database...")
    mmseqs_csv = mmseqs_dir / "mmseqs_hits.csv"
    hits_df = run_mmseqs_search_proteins(
        query_fasta=protein_fasta,
        mmseqs_db=str(database_path),
        results_dir=str(mmseqs_dir),
        output_csv=str(mmseqs_csv),
        descriptive_prefix='protein_database',
        threads=args.threads,
        sensitivity=args.sensitivity,
        e_value=args.e_value,
        only_top_hits=True
    )
    print(f"  Found {len(hits_df)} protein hits")
    
    # Step 4: Annotate hits if annotation file provided
    if annotation_file:
        print("\n[4/6] Adding functional annotations...")
        hits_df = annotate_protein_hits(str(mmseqs_csv), str(annotation_file))
        hits_df.to_csv(mmseqs_csv, index=False)
        print(f"  Annotations added and saved to: {mmseqs_csv}")
    else:
        print("\n[4/6] Skipping functional annotation (no annotation file provided)")
        # Add empty annot and category columns if they don't exist for downstream compatibility
        if 'annot' not in hits_df.columns:
            hits_df['annot'] = 'Unknown'
        if 'category' not in hits_df.columns:
            hits_df['category'] = 'Unknown'
        hits_df.to_csv(mmseqs_csv, index=False)
    
    # Step 5: Create GFF files
    print("\n[5/6] Creating GFF files...")
    genome_id_map = batch_create_gff_files(
        genomes_fasta_file=str(input_fasta),
        pseudocircular_genomes_fasta_file=pseudocircular_fasta_for_gff,
        pseudocircular_orfs_fasta_file=protein_fasta,
        protein_database_hits_df=hits_df,
        output_dir=str(gff_dir)
    )
    
    num_gffs = len(list(gff_dir.glob("*.gff")))
    print(f"  Created {num_gffs} GFF files in: {gff_dir}")
    
    # Save genome ID mapping
    genome_map_df = pd.DataFrame(list(genome_id_map.items()), 
                                 columns=["genome_name", "genome_id"])
    genome_map_df.to_csv(output_dir / "genome_id_mapping.csv", index=False)
    
    # Optional: Create GenBank files
    if args.create_gbk:
        print("\n  Creating GenBank files...")
        gbk_dir = output_dir / "gbk_files"
        if gbk_dir.exists():
            shutil.rmtree(gbk_dir)
        gbk_dir.mkdir(parents=True)
        batch_convert_gff_to_gbk(str(gff_dir), str(gbk_dir))
        print(f"  GenBank files created in: {gbk_dir}")
    
    # Step 6: Visualize with lovis4u
    if args.visualize:
        print("\n[6/6] Generating genome architecture visualizations...")
        viz_dir = output_dir / "visualizations"
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        viz_dir.mkdir(parents=True)
        
        run_lovis4u_batch(
            query_gff_dir=str(gff_dir),
            reference_genome_gff=str(reference_genome) if reference_genome else None,
            output_results_dir=str(viz_dir),
            max_workers=args.viz_workers
        )
        print(f"  Visualizations saved to: {viz_dir}")
    else:
        print("\n[6/6] Skipping visualization")
    
    # Summary
    print("\n" + "=" * 70)
    print("CDS ANNOTATION PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nInput: {input_fasta}")
    print(f"Results saved to: {output_dir}")
    if not args.skip_circularization:
        print(f"  - Pseudo-circular genomes: {pseudocircular_dir}")
    print(f"  - ORFs: {orfs_dir}")
    print(f"  - MMseqs results: {mmseqs_csv}")
    print(f"  - GFF files: {gff_dir}")
    print(f"  - Genome ID mapping: {output_dir / 'genome_id_mapping.csv'}")
    if args.create_gbk:
        print(f"  - GenBank files: {output_dir / 'gbk_files'}")
    if args.visualize:
        print(f"  - Visualizations: {output_dir / 'visualizations'}")
    print("\nNote: All files have been overwritten if they previously existed.")
    print()


if __name__ == "__main__":
    main()