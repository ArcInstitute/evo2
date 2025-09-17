#################################################
### PHAGE GENOME DESIGN // FILTERING PIPELINE ###
#################################################

"""
Usage:

eval "$(conda shell.bash hook)"
conda activate genome_design
CONFIG_FILE="/path/to/config.yaml"
python /path/to/genome_design_filtering_pipeline.py $CONFIG_FILE

Requirements:
- Environment equivalent to genome_design
- lovis4u environment equivalent to genome_visualization
- Config file with chosen parameters
- genetic_architecture.py in same directory
- genetic_architecture_visualization.py in same directory
- MMseqs databases created for any MMseqs search steps
- Local installation of Prodigal binary (if using Prodigal)
- CheckV database downloaded (if using CheckV)
- PHROGs database and annotation file downloaded
"""

import sys
import json
import yaml
import re
import itertools
import csv
import shutil
import uuid
import os
import tempfile
import subprocess
import time
import numpy as np
import pandas as pd
import glob
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
import genetic_architecture as ga
from typing import List, Tuple, Union
from itertools import chain
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed
from pathlib import Path
from Bio import SeqIO, SeqRecord, SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation


def load_fasta_to_df(sequences_path: str) -> pd.DataFrame:
    """
    Load a fasta file of sequences and return a pandas DataFrame with ID/prompt (first column) and sequence (second column).
    Note that sequences with spaces are concatenated together.
    """
    records = list(SeqIO.parse(sequences_path, "fasta"))
    
    # Extract IDs and sequences from the FASTA records
    ids = [record.id for record in records]
    sequences = [str(record.seq) for record in records]
    
    # Create a DataFrame with IDs and sequences
    df = pd.DataFrame({"id_prompt": ids, "sequence": sequences})
    
    return df


def load_fasta_to_df_eos_aware(sequences_path: str) -> pd.DataFrame:
    """
    Load a fasta file, preserve spaces, and then cut off sequences after the first space (end of sequence (EOS) token).
    Return a pandas DataFrame with ID/prompt (first column) and sequence (second column)
    """
    ids = []
    sequences = []
    
    with open(sequences_path, "r") as file:
        current_seq_id = None
        current_seq = []
        
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # If the line is a sequence ID (header)
                if current_seq_id is not None:  # Save the previous sequence
                    full_sequence = "".join(current_seq)
                    sequences.append(full_sequence.split(" ")[0])  # Keep part only before the first EOS token
                    ids.append(current_seq_id)
                
                current_seq_id = line[1:]  # Remove '>' from the header
                current_seq = []  # Reset for the new sequence
            else:
                current_seq.append(line)  # Append raw sequence lines (with spaces)
        
        # Save the last sequence
        if current_seq_id is not None:
            full_sequence = "".join(current_seq)
            sequences.append(full_sequence.split(" ")[0])  # Keep part only before the first EOS token
            ids.append(current_seq_id)
    
    # Create DataFrame with IDs and sequences
    df = pd.DataFrame({
        "id_prompt": ids,
        "sequence": sequences
    })

    return df


def load_csv_to_df(input_sequences: str) -> pd.DataFrame:
    """Load a csv file of sequences and create a pandas DataFrame with generation ID (first column), prompt (second column), and sequence (third column)."""
    sequences_df = pd.read_csv(input_sequences)
    return sequences_df


def load_csv_to_df_eos_aware(input_sequences: str) -> pd.DataFrame:
    """
    Load a CSV file of sequences and create a pandas DataFrame with generation ID (first column), prompt (second column),
    and sequence (third column). Trims sequences to the first EOS token.
    """
    sequences_df = pd.read_csv(input_sequences)
    
    # Ensure the sequence column is trimmed at the first EOS token
    if "sequence" in sequences_df.columns:
        sequences_df["sequence"] = sequences_df["sequence"].apply(lambda seq: seq.split(" ")[0])
    
    return sequences_df


def ensure_directory_exists(directory_path):
    """Ensure that the given directory exists. Create it if it does not."""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)  # Create directory, including parents, only if directory does not exist
    print(f"Results will be saved to: {directory_path}")


def valid_nt_chars(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences containing only A, C, G, T characters."""
    filtered_df = sequences_df[sequences_df['sequence'].apply(lambda seq: not re.search(r'[^ACGTacgt]', seq))]
    return filtered_df


def valid_genome_len(sequences_df: pd.DataFrame, length_range: range) -> pd.DataFrame:
    """Filter for sequences within a specified range of lengths."""
    min_len = min(length_range)
    max_len = max(length_range)
    sequences_df = sequences_df.copy()

    # Calculate genome lengths, filter, and record
    sequences_df['genome_length'] = sequences_df['sequence'].apply(len)
    filtered_df = sequences_df[
        (sequences_df['genome_length'] >= min_len) & 
        (sequences_df['genome_length'] <= max_len)]
    return filtered_df


def calculate_gc_content(sequence) -> float:
    """Calculate % GC content of a sequence."""
    gc_count = sequence.count('G') + sequence.count('C') + sequence.count('g') + sequence.count('c')
    gc_content = (gc_count / len(sequence)) * 100
    return gc_content


def valid_gc_content(sequences_df: pd.DataFrame, gc_range: range) -> pd.DataFrame:
    """Filter for sequences within a specified range of % GC content."""
    min_gc = min(gc_range)
    max_gc = max(gc_range)
    sequences_df = sequences_df.copy()

    # Calculate GC contents, filter, and record
    sequences_df['gc_content'] = sequences_df['sequence'].apply(calculate_gc_content)
    filtered_df = sequences_df[
        (sequences_df['gc_content'] >= min_gc) & 
        (sequences_df['gc_content'] <= max_gc)]
    return filtered_df


def calculate_nt_homopolymer_len(sequence) -> int:
    """Calculate longest homopolymer in a given sequence."""
    sequence = sequence.upper()
    
    # Create regex patterns for each nucleotide
    patterns = [r'(A+)', r'(C+)', r'(G+)', r'(T+)']
    
    longest_homopolymer = 0
    for pattern in patterns:
        matches = re.findall(pattern, sequence)
        if matches:
            # Find the longest match for the current nucleotide
            max_len = max(len(match) for match in matches)
            # Update longest_homopolymer if a longer stretch is found
            if max_len > longest_homopolymer:
                longest_homopolymer = max_len
    
    return longest_homopolymer


def valid_nt_homopolymer_len(sequences_df: pd.DataFrame, homopolymer_range: range) -> pd.DataFrame:
    """Filter for sequences that contain a longest homopolymer within a specified range."""
    min_len = min(homopolymer_range)
    max_len = max(homopolymer_range)
    sequences_df = sequences_df.copy()

    # Calculate homopolymer lengths, filter, and record
    sequences_df['max_nt_homopolymer_length'] = sequences_df['sequence'].apply(calculate_nt_homopolymer_len)
    filtered_df = sequences_df[
        (sequences_df['max_nt_homopolymer_length'] >= min_len) & 
        (sequences_df['max_nt_homopolymer_length'] <= max_len)]
    return filtered_df


def calculate_dinucleotide_freqs(sequence: str) -> dict:
    """Calculate frequencies of all dinucleotides in a sequence."""
    seq = sequence.upper()
   
    # Generate all 16 combinations of dinucleotides
    alphabet = ['A', 'C', 'G', 'T']
    dinucleotides = [''.join(comb) for comb in itertools.product(alphabet, repeat=2)]
    
    # Calculate dinucleotide frequencies
    dinucleotide_counts = []
    for dinucleotide in dinucleotides:
        observed_freq = seq.count(dinucleotide)
        dinucleotide_counts.append(observed_freq)
    total_dinucleotide_counts = np.sum(dinucleotide_counts)
    dinucleotide_freqs = {}
    for dinucleotide, count in zip(dinucleotides, dinucleotide_counts):
        freq = (count / total_dinucleotide_counts)
        dinucleotide_freqs[dinucleotide] = freq
    return dinucleotide_freqs


def valid_dinucleotide_content(sequences_df: pd.DataFrame, dinucleotide_freq_range: range) -> pd.DataFrame:
    """Filter for sequences where all dinucleotide frequencies fall within a specified range."""
    min_freq = min(dinucleotide_freq_range)
    max_freq = max(dinucleotide_freq_range)
    filtered_df = sequences_df[sequences_df['sequence'].apply(
        lambda seq: all(min_freq <= freq <= max_freq for freq in calculate_dinucleotide_freqs(seq).values())
    )]
    return filtered_df


def calculate_tud(sequence: str, tetranucleotide: str) -> float:
    """Calculates the tetranucleotide usage departure (TUD) for a specified tetranucleotide using the zero-order Markov method."""

    # Check that tetranucleotide is formatted properly
    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")
    else:
        pass

    seq = sequence.upper()
    tetra = tetranucleotide.upper()
    
    # Calculate frequencies of each nucleotide in the whole sequence
    total_bases = len(seq)
    A_freq = seq.count("A") / total_bases
    C_freq = seq.count("C") / total_bases
    G_freq = seq.count("G") / total_bases
    T_freq = seq.count("T") / total_bases

    # Calculate expected frequency for the tetranucleotide using zero-order Markov method
    tetra_count = seq.count(tetra)
    tetra_expected_freq = (G_freq * A_freq * T_freq * C_freq) * total_bases
    
    # Calculate TUD for the tetranucleotide
    tetra_tud = tetra_count / tetra_expected_freq if tetra_expected_freq != 0 else 0

    return tetra_tud


def valid_tud(sequences_df: pd.DataFrame, tetranucleotide: str, tud_range: range) -> pd.DataFrame:
    """Filters sequences based on whether the tetranucleotide usage deviation (TUD) falls within the specified range and records the TUD in a new column."""
    min_tud = min(tud_range)
    max_tud = max(tud_range)

    sequences_df = sequences_df.copy()

    # Calculate the TUD for each sequence, store it, and filter
    sequences_df['tud'] = sequences_df['sequence'].apply(lambda seq: calculate_tud(seq, tetranucleotide))
    filtered_df = sequences_df[
        (sequences_df['tud'] >= min_tud) & 
        (sequences_df['tud'] <= max_tud)
    ]

    return filtered_df


def run_prodigal(input_sequences: str, output_orf_file: str, output_protein_file: str, sequences_df: pd.DataFrame) -> None:
    """Run Prodigal to identify ORF candidate sequences and replace record IDs with full descriptions."""
    cmd = f'/home/samuelking/prodigal/prodigal -i {input_sequences} -d {output_orf_file} -a {output_protein_file} -p meta'
    #cmd = f'/old_home/cirrascale/miniconda3/envs/evo-design/bin/prodigal -i {input_sequences} -d {output_orf_file} -a {output_protein_file} -p meta'

    # Run the command and redirect stdout and stderr to /dev/null
    with open('/dev/null', 'w') as devnull:
        subprocess.run(cmd, shell=True, stdout=devnull, stderr=devnull)

    # Parse and replace record IDs in ORF file
    #update_prodigal_output_headers(output_orf_file, sequences_df)
    
    # Parse and replace record IDs in Protein file
    #update_prodigal_output_headers(output_protein_file, sequences_df)


def update_prodigal_output_headers(output_file: str, sequences_df: pd.DataFrame) -> None:
    """Update the headers in the Prodigal output file by replacing short record IDs with full descriptions."""
    
    updated_lines = []
    unmatched_ids = []  # To store any unmatched IDs for debugging
    
    # Read the Prodigal output file with Biopython
    with open(output_file, 'r') as infile:
        for record in SeqIO.parse(infile, "fasta"):
            # Extract the short record ID (everything before the last underscore)
            header_parts = record.id.rsplit('_', 1)  # Split on the last underscore
            short_id = header_parts[0]  # Everything before the last underscore
            orf_number = header_parts[1]  # The part after the last underscore
            
            # Find the matching full description in sequences_df by searching for short_id in id_prompt
            full_description = sequences_df.loc[sequences_df['id_prompt'].str.contains(short_id, regex=False), 'id_prompt']

            if not full_description.empty:
                full_description_str = full_description.iloc[0]
                # Construct the new header with the full description and the ORF number
                # Retain the original description but replace the ID
                new_header = f'>{full_description_str}_{orf_number} {record.description.split(" ", 1)[1]}'
            else:
                unmatched_ids.append(short_id)
                # Leave the header unchanged if no match is found
                new_header = f'>{record.description}'

            # Add the updated header and sequence to the list
            updated_lines.append(new_header + '\n')
            updated_lines.append(str(record.seq) + '\n')

    # Write the updated lines back to the same output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(updated_lines)

    # Debugging: Log any unmatched IDs
    if unmatched_ids:
        print(f"Unmatched IDs: {unmatched_ids}")


def calculate_orf_counts(prodigal_orfs: str) -> pd.DataFrame:
    """Count the number of ORFs per sequence ID in a given prodigal FASTA file and return as a pandas DataFrame."""
    orf_counts = defaultdict(int)

    for record in SeqIO.parse(prodigal_orfs, "fasta"):
        # Split the record ID by underscores
        parts = record.id.split('_')
        if len(parts) > 1:
            # The last part is the ORF count, the rest form the sequence ID
            sequence_id = '_'.join(parts[:-1])
            orf_number = int(parts[-1])
            orf_counts[sequence_id] = max(orf_counts[sequence_id], orf_number)
    
    # Convert the dictionary to a pandas DataFrame
    orf_counts_df = pd.DataFrame(list(orf_counts.items()), columns=['id_prompt', 'orf_count'])
    return orf_counts_df


def valid_orf_count(prodigal_orfs: str, orf_count_range: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences with an ORF count within a specified range."""
    
    # Calculate ORF counts using the calculate_orf_counts function
    orf_counts_df = calculate_orf_counts(prodigal_orfs)

    # Check for matches between orf_counts_df and sequences_df in the id_prompt column
    for index, row in orf_counts_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['id_prompt'], regex=False)]
        
        # Replace the id_prompt in orf_counts_df with the matching one from sequences_df
        if not matching_prompt.empty:
            orf_counts_df.at[index, 'id_prompt'] = matching_prompt.values[0]
    
    # Merge the ORF counts with the original sequences_df DataFrame
    sequences_df = pd.merge(sequences_df, orf_counts_df, on='id_prompt', how='left')
    
    # Filter based on ORF count range
    min_orf = min(orf_count_range)
    max_orf = max(orf_count_range)
    filtered_df = sequences_df[(sequences_df['orf_count'] >= min_orf) & (sequences_df['orf_count'] <= max_orf)]
    
    return filtered_df


def calculate_orf_lengths(prodigal_orfs: str) -> pd.DataFrame:
    """Calculate the lengths of all ORFs per sequence ID in a given prodigal FASTA file and return as a pandas DataFrame with lengths stored as lists."""
    orf_lengths = defaultdict(list)  # Store ORF lengths as lists for each sequence ID

    for record in SeqIO.parse(prodigal_orfs, "fasta"):
        # Extract ORF length from the header (length = end position - start position + 1)
        header_info = record.description.split('#')
        start_pos = int(header_info[1].strip())
        end_pos = int(header_info[2].strip())
        orf_length = end_pos - start_pos + 1
        
        # Extract sequence ID and ORF number
        parts = record.id.split('_')
        if len(parts) > 1:
            sequence_id = '_'.join(parts[:-1])  # All parts except the last one
            orf_number = parts[-1]  # The last part
        else:
            sequence_id = record.id
            orf_number = None
        
        # Append ORF length to the list for this sequence ID
        orf_lengths[sequence_id].append(orf_length)
    
    # Convert the dictionary to a pandas DataFrame
    orf_lengths_df = pd.DataFrame([(seq_id, lengths) for seq_id, lengths in orf_lengths.items()], columns=['id_prompt', 'orf_lengths'])
    
    return orf_lengths_df


def valid_orf_lengths(prodigal_orfs: str, orf_length_range: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences with ORF lengths within a specified range."""
    
    # Calculate ORF lengths using the calculate_orf_lengths function
    orf_lengths_df = calculate_orf_lengths(prodigal_orfs)

    # Check for matches between orf_lengths_df and sequences_df in the id_prompt column
    for index, row in orf_lengths_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['id_prompt'], regex=False)]
        
        # Replace the id_prompt in orf_counts_df with the matching one from sequences_df
        if not matching_prompt.empty:
            orf_lengths_df.at[index, 'id_prompt'] = matching_prompt.values[0]
    
    # Merge the ORF lengths with the original sequences_df DataFrame
    sequences_df = pd.merge(sequences_df, orf_lengths_df, on='id_prompt', how='left')
    
    # Define the range
    min_len = min(orf_length_range)
    max_len = max(orf_length_range)
    
    # Filter sequences where all ORF lengths are within the range
    def orf_length_within_range(orf_lengths):
        # Check if all ORF lengths are within the specified range
        return all(min_len <= orf_len <= max_len for orf_len in orf_lengths)
    
    filtered_df = sequences_df[sequences_df['orf_lengths'].apply(orf_length_within_range)]
    
    return filtered_df


def valid_coding_density(sequences_df: pd.DataFrame, coding_density_range: tuple) -> pd.DataFrame:
    """Filter for sequences with a coding density within a specified range."""

    sequences_df['total_orfs_length'] = sequences_df['orf_lengths'].apply(sum)

    min_density = min(coding_density_range)
    max_density = max(coding_density_range)
    
    # Check for NaN values in critical columns
    if sequences_df[['total_orfs_length', 'genome_length']].isnull().values.any():
        raise ValueError("NaN values found in 'total_orfs_length' or 'genome_length' columns.")
    
    # Check for zero ORFs length and raise an error
    if (sequences_df['total_orfs_length'] == 0).any():
        raise ValueError("Total length of ORFs cannot be zero.")
    
    # Check for zero genome length which would lead to division errors
    if (sequences_df['genome_length'] == 0).any():
        raise ValueError("Genome length cannot be zero.")
    
    # Calculate coding density
    sequences_df = sequences_df.copy()  # Avoid modifying the original DataFrame
    sequences_df['coding_density'] = (sequences_df['total_orfs_length'] / sequences_df['genome_length']) * 100
    
    # Filter the DataFrame based on the coding density range
    filtered_df = sequences_df[(sequences_df['coding_density'] >= min_density) & (sequences_df['coding_density'] <= max_density)]

    return filtered_df


def calculate_aa_homopolymer_len(sequence: str) -> int:
    """Calculate the longest homopolymer of any amino acid in a given sequence."""
    sequence = sequence.upper()
    
    # Create regex patterns for each amino acid
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    patterns = [f'({aa}+)' for aa in amino_acids]
    
    longest_homopolymer = 0
    for pattern in patterns:
        matches = re.findall(pattern, sequence)
        if matches:
            # Find the longest match for the current amino acid
            max_len = max(len(match) for match in matches)
            # Update longest_homopolymer if a longer stretch is found
            if max_len > longest_homopolymer:
                longest_homopolymer = max_len
    
    return longest_homopolymer


def valid_aa_homopolymer_len(prodigal_proteins: str, homopolymer_length_range: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences that contain a longest amino acid homopolymer within a specified range and records the homopolymer length."""
    
    min_len = min(homopolymer_length_range)
    max_len = max(homopolymer_length_range)

    # Parse the prodigal_proteins FASTA file and calculate the longest homopolymer for each sequence
    homopolymer_lengths = []

    for record in SeqIO.parse(prodigal_proteins, "fasta"):
        # Extract sequence ID
        parts = record.id.split('_')
        if len(parts) > 1:
            sequence_id = '_'.join(parts[:-1])  # All parts except the last one
            orf_number = parts[-1]  # The last part
        else:
            sequence_id = record.id
            orf_number = None
        
        # Calculate the longest amino acid homopolymer for this sequence
        homopolymer_len = calculate_aa_homopolymer_len(str(record.seq))

        # Store the results in a list
        homopolymer_lengths.append({'id_prompt': sequence_id, 'max_aa_homopolymer_len': homopolymer_len})

    # Convert the list of homopolymer lengths to a pandas DataFrame
    homopolymer_lengths_df = pd.DataFrame(homopolymer_lengths)
    homopolymer_lengths_df = homopolymer_lengths_df.groupby('id_prompt', as_index=False)['max_aa_homopolymer_len'].max()

    # Check for matches between homopolymer_lengths_df and sequences_df in the id_prompt column
    for index, row in homopolymer_lengths_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['id_prompt'], regex=False)]
        
        # Replace the id_prompt in homopolymer_lengths_df with the matching one from sequences_df
        if not matching_prompt.empty:
            homopolymer_lengths_df.at[index, 'id_prompt'] = matching_prompt.values[0]

    # Merge the homopolymer lengths with the original sequences_df DataFrame
    sequences_df = pd.merge(sequences_df, homopolymer_lengths_df[['id_prompt', 'max_aa_homopolymer_len']], on='id_prompt', how='left')

    # Filter based on the specified homopolymer length range
    filtered_df = sequences_df[(sequences_df['max_aa_homopolymer_len'] >= min_len) & (sequences_df['max_aa_homopolymer_len'] <= max_len)]

    return filtered_df


def append_upstream_of_last_frame_stop(input_fasta: str, output_fasta: str) -> None:
    """
    Finds the first stop codon but the last relative to the three reading frames in a sequence, and appends
    the upstream portion before it to the end of the sequence. This process can be called 'pseudo-circularization'.
    """
    def find_last_frame_stop(seq: Seq) -> int:
        """Find the first stop codon in each reading frame and return the furthest one (including the stop codon itself)."""
        stop_codons = ['TAA', 'TAG', 'TGA']
        first_stops = []

        # Check each of the three reading frames
        for frame in range(3):
            sub_seq = seq[frame:]  # Start from the current reading frame
            for i in range(0, len(sub_seq) - 3, 3):  # Iterate through the sequence in codon steps
                codon = sub_seq[i:i + 3]
                if codon in stop_codons:
                    first_stops.append(i + frame + 3)  # Include the stop codon itself
                    break  # Stop at the first stop codon in this frame

        # Return the furthest stop codon position (including the stop codon)
        return max(first_stops) if first_stops else len(seq)  # If no stop codon found, return full length

    records = []

    # Read input FASTA
    with open(input_fasta, "r") as input_handle:
        counter = 0
        for record in SeqIO.parse(input_handle, "fasta"):
            seq = record.seq

            # Find the last relative stop codon
            last_stop = find_last_frame_stop(seq)

            # Append the sequence upstream of the last stop codon to the end of the full sequence
            new_seq = seq + seq[:last_stop]

            # Create a new record with the updated sequence
            new_record = record
            new_record.seq = new_seq
            records.append(new_record)

            # Increment counter and print every 1000 iterations
            counter += 1
            if counter % 1000 == 0:
                print(f"Pseudo-circularized {counter} sequences")
        
    # Save to output FASTA
    with open(output_fasta, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


def run_orfipy(input_fasta: str, threads: int, start_codons: str, stop_codons: str, strand: str, min_len: int, max_len: int, output_dir: str, output_nt: str, output_aa_tmp, output_aa) -> None:
    """Call ORFs in given input fasta file using Orfipy and save nucleotide and cleaned (no *) amino acid sequences."""
    # Run orfipy command
    orfipy_command = [
        'orfipy', '--procs', str(threads), input_fasta, '--start', start_codons, '--stop', stop_codons, '--strand', strand, 
        '--include-stop', '--min', str(min_len), '--max', str(max_len), '--outdir', output_dir, 
        '--dna', output_nt, '--pep', output_aa_tmp #, '--table, translation_table'
    ]
    subprocess.run(orfipy_command, check=True)
    
    clean_orfipy_fasta_file(f'{output_dir}/{output_aa_tmp}', f'{output_dir}/{output_aa}') # Remove stop codon (*) symbols from output protein fasta file
    os.remove(f'{output_dir}/{output_aa_tmp}') # Remove the temporary output file

    
def clean_orfipy_fasta_file(input_fasta: str, output_fasta: str) -> None:
    """Remove * symbols from protein sequences in given fasta file."""
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        for line in infile:
            if line.startswith(">"):
                outfile.write(line)  # Write header line unchanged
            else:
                outfile.write(line.replace('*', ''))  # Remove * from sequence lines


def run_checkv(input_file: str, output_dir: str, num_threads: int=16) -> None:
    """Run the CheckV to identify viral sequences. Input file is FASTA. Default number of threads is 16."""
    env = {**os.environ, 'CHECKVDB': "/large_experiments/hielab/brianhie/dna-gen/checkv/checkv-db-v1.5"}
    cmd = ['checkv', 'end_to_end', input_file, output_dir, '-t', str(num_threads)]
    subprocess.run(cmd, env=env, check=True)


def valid_checkv_quality(input_file: str, checkv_quality_range: list, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences that are classified by CheckV as one or more of the specified quality levels."""
    quality_summary_df = pd.read_csv(input_file, delimiter="\t")
    
    # Filter the DataFrame based on the 'checkv_quality' column
    results_df = quality_summary_df[quality_summary_df['checkv_quality'].isin(checkv_quality_range)]
    results_df = results_df[['contig_id', 'checkv_quality']]

    # Check for matches between results_df and sequences_df in the id_prompt column
    for index, row in results_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['contig_id'], regex=False)]
        
        # Replace the id_prompt in results_df with the matching one from sequences_df
        if not matching_prompt.empty:
            results_df.at[index, 'contig_id'] = matching_prompt.values[0]
    
    # Merge the CheckV qualities with the original sequences_df DataFrame
    results_df.rename(columns={'contig_id': 'id_prompt'}, inplace=True)
    sequences_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')

    return sequences_df


def run_mmseqs_search_genomes(query_genomes: str, 
                              target_genomes: str,
                              query_db_dir: str,
                              target_db_dir: str, 
                              tmp_dir: str, 
                              results_dir: str,
                              results_filename: str,
                              search_type: int=3,
                              threads: int=8,
                              sensitivity: float=7.5):
    """
    Run MMseqs search with the specified directories and files.
    
    Arguments:
    - query_genomes: Path to the query genomes FASTA file.
    - target_genomes: Path to the target genomes FASTA file.
    - query_db_dir: Directory to store the query database.
    - target_db_dir: Directory to store the target database.
    - tmp_dir: Directory for temporary files.
    - results_dir: Directory for storing results.
    - results_filename: Name of .m8 file for storing results.
    - search_type: MMseqs search mode (2=translated, 3=nucleotide, default: 3)
    - threads: Number of threads for parallel processing (default: 8).
    - sensitivity: Sensitivity for the MMseqs search (default: 7.5).
    """

    # Ensure the directories exist and are clean
    for dir_path in [query_db_dir, target_db_dir]:
        if os.path.exists(dir_path):
            subprocess.run(f"rm -rf {dir_path}", shell=True, check=True) 

    for dir_path in [tmp_dir, results_dir]:
        if os.path.exists(dir_path):
            subprocess.run(f"rm -rf {dir_path}", shell=True, check=True)
        os.makedirs(dir_path, exist_ok=True)
    
    # Create query and target databases
    subprocess.run(f"mmseqs createdb {query_genomes} {query_db_dir}", shell=True, check=True)
    subprocess.run(f"mmseqs createdb {target_genomes} {target_db_dir}", shell=True, check=True)

    # Index the target database
    subprocess.run(f"mmseqs createindex {target_db_dir} {tmp_dir} --search-type {search_type} --threads {threads}", shell=True, check=True)

    # Run MMseqs search
    subprocess.run(f"mmseqs search {query_db_dir} {target_db_dir} {results_dir} {tmp_dir} --search-type {search_type} --threads {threads} -s {sensitivity}", shell=True, check=True)

    # Convert results to .m8 format
    subprocess.run(f"mmseqs convertalis {query_db_dir} {target_db_dir} {results_dir} {results_dir}/{results_filename} --format-output 'query,target,pident,evalue'", shell=True, check=True)

    print(f"MMseqs search completed. Results saved in {results_dir}/{results_filename}")


def convert_m8_to_df(m8_file_path: str, descriptive_prefix: str) -> pd.DataFrame:
    """
    Convert a .m8 file (tab-delimited) to a pandas DataFrame and keep only top MMseqs hits.
    A descriptive prefix for each MMseqs result label (target, percent_identity, e_value) must be provided.
    """

    # Define column names (adjust according to the .m8 file structure)
    column_names = ["id_prompt", f"{descriptive_prefix}_mmseqs_target", f"{descriptive_prefix}_mmseqs_percent_identity", f"{descriptive_prefix}_mmseqs_e_value"]
    df = pd.read_csv(m8_file_path, sep="\t", header=None, names=column_names)
    
    # For each query, keep only the row with the highest percent_identity
    df_top_hits = df.loc[df.groupby('id_prompt')[f'{descriptive_prefix}_mmseqs_percent_identity'].idxmax()]

    return df_top_hits


def valid_mmseqs_pident(mmseqs_df: pd.DataFrame, descriptive_prefix: str, pident_range: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for sequences that have a % sequence identity to an MMseqs hit within a specified range."""

    min_pident = min(pident_range)
    max_pident = max(pident_range)

    # Drop the sequences column in mmseqs_df (if any) since we don't want to redundantly add it to sequences_df
    if 'sequence' in mmseqs_df.columns:
        mmseqs_df = mmseqs_df.drop(columns=['sequence'])

    # Drop the '_ORF.#' from orfipy prediction results if applicable
    if 'id_prompt' in mmseqs_df.columns and 'ORF' in mmseqs_df['id_prompt'].iloc[0]:
        mmseqs_df['id_prompt'] = mmseqs_df['id_prompt'].str.split('_').str[:-1].str.join('_')
        mmseqs_df = mmseqs_df.loc[mmseqs_df.groupby('id_prompt')[f'{descriptive_prefix}_mmseqs_percent_identity'].idxmax()].reset_index(drop=True)

    # Fix id_prompt to match those in sequences_df
    for index, row in mmseqs_df.iterrows():
        matching_prompt = sequences_df['id_prompt'][sequences_df['id_prompt'].str.contains(row['id_prompt'], regex=False)]
        if not matching_prompt.empty:
            mmseqs_df.at[index, 'id_prompt'] = matching_prompt.values[0]

    # Handle filtering based on pident range
    #results_df = mmseqs_df[
    #    (mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] >= min_pident) &
    #    (mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] <= max_pident)
    #]

    #if min_pident > 0:
    #    merged_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')
    #    filtered_df = merged_df.dropna(subset=[f'{descriptive_prefix}_mmseqs_percent_identity'])
    #elif min_pident == 0:
    #    # Identify hits that exceed max_pident
    #    over_max_df = mmseqs_df[mmseqs_df[f'{descriptive_prefix}_mmseqs_percent_identity'] > max_pident]
    #    over_max_ids = set(over_max_df['id_prompt'])
    #    merged_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')
    #    merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] = (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'].fillna(0))
    #    filtered_df = merged_df[~merged_df['id_prompt'].isin(over_max_ids)]

    # Merge full MMseqs results to preserve no-hit sequences as NaN
    merged_df = pd.merge(sequences_df, mmseqs_df, on='id_prompt', how='left')

    # Fill missing (no-hit) percent identities with 0
    merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] = (
        merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'].fillna(0)
    )

    # Apply the min/max filter (now can include 0% identity sequences)
    filtered_df = merged_df[
        (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] >= min_pident) &
        (merged_df[f'{descriptive_prefix}_mmseqs_percent_identity'] <= max_pident)
    ]

    return filtered_df


def calculate_pident_to_ref(query_sequence: str, reference_sequence: str) -> float:
    """Calculates % sequence identity to reference sequence using global alignment."""

    # Ensure sequences are uppercase and remove any unexpected whitespaces
    query_sequence = query_sequence.replace("*", "").upper().strip()
    reference_record = next(SeqIO.parse(reference_sequence, "fasta"))
    reference_sequence = str(reference_record.seq).upper().strip()

    # Check sequences for unexpected characters
    valid_nucleotides = set("ACGT")
    if any(char not in valid_nucleotides for char in query_sequence) or any(char not in valid_nucleotides for char in reference_sequence):
        raise ValueError(f"Invalid character in query or reference sequence: {query_sequence}, {reference_sequence}")

    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    query = seq.NucleotideSequence(query_sequence)
    ref = seq.NucleotideSequence(reference_sequence)
    alignments = align.align_optimal(query, ref, matrix, terminal_penalty=False) # Default is global alignment (based on Needleman-Wunsch algorithm)

    # Calculate sequence identity
    seq_ident = align.get_pairwise_sequence_identity(alignments[0])
    return seq_ident[0][1] * 100


def valid_reference_genome_pident(sequences_df: pd.DataFrame, reference_sequence: str, pident_range: tuple) -> pd.DataFrame:
    """Filter for sequences that have a % sequence identity to a reference sequence."""
    
    min_pident, max_pident = pident_range  # Unpack the range

    def calculate_and_check_identity(query_sequence):
        """Helper function to calculate % identity and return it."""
        pident = calculate_pident_to_ref(query_sequence, reference_sequence)
        return pident

    # Calculate % identity for each sequence
    sequences_df['reference_genome_percent_identity'] = sequences_df['sequence'].apply(calculate_and_check_identity)

    # Filter the DataFrame based on the provided percent identity range
    filtered_df = sequences_df[(sequences_df['reference_genome_percent_identity'] >= min_pident) & (sequences_df['reference_genome_percent_identity'] <= max_pident)]

    return filtered_df


def calculate_genetic_architecture_score_parallel(sequences_df: pd.DataFrame,
                                                  truth_matrix: np.array,
                                                  weight_vector: np.array,
                                                  normalization_vector: np.array,
                                                  n_jobs: int=-1) -> pd.DataFrame:
    """
    Processes each sequence in parallel and calculates the genetic architecture score.
    
    Arguments:
    - sequences_df: pandas DataFrame containing sequence data.
    - truth_matrix: The truth matrix used for scoring.
    - weight_vector: The weight vector used for scoring.
    - normalization_vector: The vector used for normalizing the genetic architecture score.
    - n_jobs: Number of parallel jobs to run (default: -1, which uses all available cores).
    
    Returns:
    - A pandas DataFrame with the genetic architecture scores.
    """
    
    def process_single_sequence(sequence, sequence_id, truth_matrix, weight_vector, normalization_vector):
        """Helper function to calculate the genetic architecture score of a single sequence."""
        max_dot_products_normalized = ga.genetic_architecture_score(truth_matrix, sequence, weight_vector, normalization_vector)
        return (sequence_id, max_dot_products_normalized)

    # Run in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_sequence)(row['sequence'], row['id_prompt'], truth_matrix, weight_vector, normalization_vector)
        for _, row in sequences_df.iterrows()
    )

    results_df = ga.save_score(results)
    sequences_df = pd.merge(sequences_df, results_df, on='id_prompt', how='left')

    return sequences_df


def valid_genetic_architecture_score(sequences_df: pd.DataFrame, 
                                     truth_matrix: np.array, 
                                     weight_vector: np.array, 
                                     normalization_vector: np.array, 
                                     genetic_architecture_score_range: tuple,
                                     keep_or_remove_range: str="keep", 
                                     mode: str="compound",
                                     n_jobs: int=-1) -> pd.DataFrame:
    """
    Filter sequences based on the genetic architecture score within a specified range.

    Arguments:
    - sequences_df: pandas DataFrame containing sequence data.
    - truth_matrix: The truth matrix used for scoring.
    - weight_vector: The weight vector used for scoring.
    - normalization_vector: The vector used for normalizing the genetic architecture score.
    - genetic_architecture_score_range: A tuple specifying the (min, max) score range.
    - keep_or_remove_range: "keep" if sequences within the score range are to be kept, and "remove" if they are to be removed. Default is "keep".
    - mode: The type of score to use;
        if "compound", the score is the product of the genome score and entangled module scores;
        if "genome", the score is the genome score only. Default is "compound".
    - n_jobs: Number of parallel jobs to run (default: -1, which uses all available cores).
    
    Returns:
    - A filtered DataFrame with sequences that have a genetic architecture score within the range.
    """
    
    min_score, max_score = genetic_architecture_score_range

    # Calculate scores and filter the DataFrame based on the genetic architecture score range
    if keep_or_remove_range == "keep":
        if mode == "compound":
            # Calculate the genetic architecture score for each sequence in parallel and filter
            results_df = calculate_genetic_architecture_score_parallel(sequences_df, truth_matrix, weight_vector, normalization_vector, n_jobs=n_jobs)
            filtered_df = results_df[
                (results_df['genetic_architecture_score'] >= min_score) & 
                (results_df['genetic_architecture_score'] <= max_score)
            ]
        elif mode == "genome":
            # Calculate the genetic architecture score for each sequence in parallel and filter
            results_df = calculate_genetic_architecture_score_parallel(sequences_df, truth_matrix, weight_vector, normalization_vector, n_jobs=n_jobs)
            filtered_df = results_df[
                (results_df['genome_score'] >= min_score) & 
                (results_df['genome_score'] <= max_score)
            ]
            
    elif keep_or_remove_range == "remove":
        if mode == "compound":
            filtered_df = sequences_df[
                (sequences_df['genetic_architecture_score'] < min_score) | 
                (sequences_df['genetic_architecture_score'] > max_score)
            ]
        elif mode == "genome":
            filtered_df = sequences_df[
                (sequences_df['genome_score'] < min_score) | 
                (sequences_df['genome_score'] > max_score)
            ]            
    
    return filtered_df


def mmseqs_search_proteins(query_fasta: str, mmseqs_db: str, results_dir: str, threads: int=8, split: int=0, sensitivity: float=4.0) -> None:
    """Run MMseqs2 search on the input protein fasta file against the given MMseqs database."""

    os.makedirs(results_dir, exist_ok=True)
    mmseqs_out = os.path.join(results_dir, "mmseqs_result.m8")
    log_file = os.path.join(results_dir, "mmseqs_search.log")

    # Run MMseqs search
    cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    #cmd = f"mmseqs easy-search {query_fasta} {mmseqs_db} {mmseqs_out} {results_dir} --threads {threads} --split {split} -s {sensitivity} -e 1e-15 --remove-tmp-files 1 --format-output 'query,target,evalue,pident'"
    print(f"Running command: {cmd}")
    start_time = time.time()
    try:
        with open(log_file, "w") as log:
            result = subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=log, text=True)
    except subprocess.CalledProcessError as e:
        print(f"MMseqs2 search failed with error: {e.stderr}")
        raise e
    end_time = time.time()
    print(f"MMseqs2 search completed in {end_time - start_time:.2f} seconds.")
    
    if not os.path.isfile(mmseqs_out):
        print(f"Output file not found: {mmseqs_out}")
    return mmseqs_out


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
    sequences = {record.id: record.seq for record in SeqIO.parse(query_fasta, "fasta")}
    
    data = []
    for query, target, evalue, pident in hits:
        if query in sequences:
            data.append([query, sequences[query], target, evalue, pident])
    
    df = pd.DataFrame(data, columns=["id_prompt", "sequence", f"{descriptive_prefix}_mmseqs_target", f"{descriptive_prefix}_mmseqs_e_value", f"{descriptive_prefix}_mmseqs_percent_identity"])
    if only_top_hits==True:
        df_top_hits = df.loc[df.groupby("id_prompt")[f"{descriptive_prefix}_mmseqs_e_value"].idxmin()]  # Keep only the top hit per query based on e-value
        df_top_hits.to_csv(output_csv, index=False)
        return df_top_hits
    else:
        df.to_csv(output_csv, index=False)
        return df


def run_mmseqs_search_proteins(query_fasta: str,
                               mmseqs_db: str,
                               results_dir: str,
                               output_csv: str,
                               descriptive_prefix: str,
                               threads: int=8,
                               split: int=0,
                               sensitivity: float=4.0,
                               only_top_hits: bool=True) -> pd.DataFrame:
    """
    Run MMseqs search with the specified directories and files.
    
    Arguments:
    - query_fasta: Path to the query proteins FASTA file.
    - mmseqs_db: Path to the target proteins MMseqs database.
    - results_dir: Directory to store the MMseqs results.
    - output_csv: Desired filename for MMseqs results.
    - threads: Number of threads for parallel processing (default: 8).
    - split: Desired chunk size for splitting queries in MMseqs search (default: 0).
    - sensitivity: Sensitivity for the MMseqs search (default: 4.0).
    - only_top_hits: Whether or not to keep only the top MMseqs hit for each protein.
    """
    if not os.path.isfile(query_fasta):
        raise FileNotFoundError(f"FASTA file not found: {query_fasta}")
    if not os.path.isdir(mmseqs_db) and not os.path.isfile(mmseqs_db):
        raise FileNotFoundError(f"MMseqs database not found: {mmseqs_db}")
    
    mmseqs_out = mmseqs_search_proteins(query_fasta, mmseqs_db, results_dir, threads, split, sensitivity)
    hits = parse_mmseqs_results(mmseqs_out)
    df = mmseqs_results_to_df(hits, query_fasta, output_csv, descriptive_prefix, only_top_hits)
    df.to_csv(output_csv, index=False)
    return df


def valid_protein_database_hit_count(protein_database_hits_df: pd.DataFrame, sequences_df: pd.DataFrame, id_column: str='query', min_hits: int=7) -> pd.DataFrame:
    """
    Filters the sequences dataframe to keep only entries with genome IDs that have at least a specified number of MMseqs hits to a given protein database.
    
    Args:
    - protein_database_hits_df (pd.DataFrame): DataFrame of MMseqs protein database hits.
    - id_column (str): Column name in `protein_database_hits_df` containing query IDs (default: 'query').
    - min_hits (int): Minimum number of hits required to keep a genome (default: 7).
    - sequences_df (pd.DataFrame): DataFrame containing genome IDs to filter.
    
    Returns:
    - pd.DataFrame: Filtered sequences_df with a column recording the number of hits per genome.
    """
    
    # Extract genome ID part before the last underscore
    protein_database_hits_df['genome_id'] = protein_database_hits_df[id_column].str.split('_').str[:-1].str.join('_')
    
    # Count hits for each genome ID
    genome_counts = protein_database_hits_df['genome_id'].value_counts()
    
    # Filter to keep only genome IDs with at least min_hits
    genomes_to_keep = genome_counts[genome_counts >= min_hits].index
    
    # Filter protein_database_hits_df to include only relevant genome IDs
    filtered_protein_database_hits_df = protein_database_hits_df[protein_database_hits_df['genome_id'].isin(genomes_to_keep)].copy()
    
    # Filter the sequences DataFrame
    filtered_df = sequences_df[sequences_df['id_prompt'].isin(genomes_to_keep)].copy()
    
    # Add a column to filtered_df with the hit counts
    filtered_df['protein_database_hit_count'] = filtered_df['id_prompt'].map(genome_counts)

    return filtered_df


def run_mmseqs_clustering(input_fasta: str, output_dir: str, min_seq_id: float=0.99) -> None:
    """
    Runs MMseqs clustering pipeline:
    1. Create a sequence database for the input fasta.
    2. Cluster sequences with the specified identity threshold.
    3. Extract cluster representatives.
    4. Save results as a TSV file.

    Parameters:
        input_fasta (str): Path to the input FASTA file.
        output_dir (str): Path to the output directory for results.
        min_seq_id (float): Minimum sequence identity for clustering (default: 0.99).
    """
    # Define directories
    db_dir = os.path.join(output_dir, "mmseqs_db")
    results_dir = os.path.join(output_dir, "mmseqs_results")
    
    # Create directories if they don't exist
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define MMseqs commands
    createdb_cmd = ["mmseqs", "createdb", input_fasta, f"{db_dir}/sequences"]
    cluster_cmd = [
        "mmseqs", "cluster", f"{db_dir}/sequences", f"{results_dir}/clusters", 
        "tmp", "--min-seq-id", str(min_seq_id)
    ]
    createsubdb_cmd = [
        "mmseqs", "createsubdb", f"{results_dir}/clusters", f"{db_dir}/sequences", 
        f"{results_dir}/representative_sequences"
    ]
    createtsv_cmd = [
        "mmseqs", "createtsv", f"{db_dir}/sequences", f"{db_dir}/sequences", 
        f"{results_dir}/clusters", f"{results_dir}/clusters.tsv"
    ]
    
    # Run commands
    try:
        subprocess.run(createdb_cmd, check=True) # Create MMseqs sequence database
        subprocess.run(cluster_cmd, check=True) # Cluster sequences
        subprocess.run(createsubdb_cmd, check=True) # Extract representative sequences from clusters
        subprocess.run(createtsv_cmd, check=True) # Save results as TSV
    except subprocess.CalledProcessError as e:
        print(f"Error during MMseqs execution: {e}")
        raise


def extract_mmseqs_cluster_representatives(clusters_tsv: str, input_fasta: str, output_fasta: str, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract representative sequences from MMseqs clusters.
    
    Args:
    - clusters_tsv: Path to the tsv file from MMseqs clustering containing representative sequence IDs (first column) and clustered sequence IDs (second column).
    - input_fasta: Path to the fasta file to parse through to extract only representative sequences from.
    - output_fasta: Path for an output fasta file containing only extracted representative sequences.
    - input_df: Dataframe to filter using cluster representatives.

    Returns:
    - pd.DataFrame: A DataFrame containing the sequence IDs and sequences of the output fasta file.
    """

    df = pd.read_csv(clusters_tsv, sep='\t', header=None)

    # Filter for representative sequences in the fasta file
    filter_ids = set(df[0])

    with open(output_fasta, 'w') as output_handle:
        for record in SeqIO.parse(input_fasta, 'fasta'):
            if record.id in filter_ids:
                SeqIO.write(record, output_handle, 'fasta')

    # Parse the output_fasta and convert to DataFrame
    output_records = list(SeqIO.parse(output_fasta, 'fasta'))
    output_df = pd.DataFrame({
        "id_prompt": [record.id for record in output_records],
        "sequence": [str(record.seq) for record in output_records]
    })

    # Filter input_df for matching id_prompt values
    filtered_input_df = input_df[input_df['id_prompt'].isin(output_df['id_prompt'])].copy()

    return filtered_input_df


def save_df_as_fasta(sequences_df: pd.DataFrame, output_fasta: str) -> None:
    """Save a pandas DataFrame as a fasta file using the 'id_prompt' as headers and 'sequence' as sequences."""
    records = []

    # Automatically select all columns except 'id_prompt' and 'sequence' as description columns
    description_cols = [col for col in sequences_df.columns if col not in ['id_prompt', 'sequence']]

    for _, row in sequences_df.iterrows():
        # Create a description string by joining all columns except 'id_prompt' and 'sequence'
        description = " ".join([f"{col}:{row[col]}" for col in description_cols])

        # Create SeqRecord with id, sequence, and description
        record = SeqRecord(Seq(row['sequence']), id=row['id_prompt'], description=description)
        records.append(record)
    
    SeqIO.write(records, output_fasta, "fasta")


def extract_tropism_protein_queries(tropism_protein_hits_df: pd.DataFrame) -> set:
    """
    Trim the characters after and including the last underscore from each 'query' entry to get matching genome names.
    Each 'query' in tropism_protein_hits_df ends with '_ORF.#', which needs to be trimmed off to get the genome name.
    """
    return set(tropism_protein_hits_df['query'].str.split('_').str[:-1].str.join('_').tolist())


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
    """
    orf_data = {}
    protein_database_hits_dict = protein_database_hits_df.set_index('id_prompt')[['sequence', 'protein_database_mmseqs_target', 'protein_database_mmseqs_percent_identity', 'annot', 'category']].to_dict('index')  # Fast lookup
    
    # Process genome lengths from the genomes_fasta_file
    genome_lengths = {}
    for record in SeqIO.parse(genomes_fasta_file, "fasta"):
        genome_name = record.id
        genome_length = len(record.seq)
        genome_lengths[genome_name] = genome_length

    # Process ORF data
    for record in SeqIO.parse(orfs_fasta_file, "fasta"):
        header = record.description
        genome_name = record.id.split('_ORF')[0]

        # Only process ORFs for genomes in the ID map
        if genome_name in genome_id_map and record.id in protein_database_hits_dict:
            genome_id = genome_id_map[genome_name]  # Map to numeric ID
            # Retrieve function and product information
            sequence_value = protein_database_hits_dict[record.id]['sequence']
            percent_identity_value = protein_database_hits_dict[record.id]['protein_database_mmseqs_percent_identity']
            function_value = protein_database_hits_dict[record.id]['category']
            product_value = protein_database_hits_dict[record.id]['annot']
            # Extract the 'ORF.#' part from the record ID
            match = re.search(r'ORF\.\d+', record.id)
            if match:
                orf_id = match.group(0)  # Extract the entire matched 'ORF.#' string
            else:
                orf_id = None  # Handle cases where no match is found
            # Extract start and end positions
            match = re.search(r'\[(\d+)-(\d+)\]', header)
            if match:
                start, end = match.groups()
                # Extract strand (this is new)
                strand_match = re.search(r'\[\d+-\d+\]\((\+|\-)\)', header)
                strand = strand_match.group(1) if strand_match else '+'  # Default to '+' if no strand is found
                orf_entry = {
                    'seq_id': genome_id,
                    'feature_type': 'CDS',
                    'start': start,
                    'end': end,
                    'score': '.',  # Placeholder for score
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
                'score': '.',  # Placeholder for score
                'strand': '+',
                'phase': '.',  # Not applicable for regions
                'attributes': f"ID={genome_id};length={genome_length}"
            }
            # Append genome length entry to the corresponding genome ID
            orf_data.setdefault(genome_id, []).append(genome_entry)

    return orf_data


def create_gff_file(orf_data: dict, genome_name: str, genome_id: str, genome_seq: str, output_dir) -> None:
    """Write a .gff file for each genome using ORF data, its given name, ID, and sequence, and saves to an output directory."""
    output_gff = os.path.join(output_dir, f"{genome_id}.gff")

    with open(output_gff, 'w') as gff:
        gff.write("##gff-version 3\n")
        gff.write(f"##sequence-region {genome_id} 1 {len(genome_seq)}\n")
        gff.write(f"##description {genome_name}\n")

        # Write ORF features if they exist for this genome
        if genome_id in orf_data:
            for orf in orf_data[genome_id]:
                gff_line = (
                    f"{orf['seq_id']}\tPredicted genome annotation\t{orf['feature_type']}\t"
                    f"{orf['start']}\t{orf['end']}\t{orf['score']}\t{orf['strand']}\t"
                    f"{orf['phase']}\t{orf['attributes']}\n"
                )
                gff.write(gff_line)

        # Append genome sequence in FASTA format
        gff.write("##FASTA\n")
        gff.write(f">{genome_id}\n")
        gff.write(f"{genome_seq}\n")


def annotate_phrogs_hits(phrogs_mmseqs_hits_file: str, phrogs_database_annotation_file: str) -> pd.DataFrame:
    """Adds PHROGs database annotations to MMseqs hits against PHROGs database."""

    # Load dataframes
    phrogs_mmseqs_hits_df = pd.read_csv(phrogs_mmseqs_hits_file)
    phrogs_database_annotation_df = pd.read_csv(phrogs_database_annotation_file, sep='\t')

    # Ensure both target (PHROGs protein labels) and hit label are treated as strings
    phrogs_mmseqs_hits_df['protein_database_mmseqs_target'] = phrogs_mmseqs_hits_df['protein_database_mmseqs_target'].astype(str)
    phrogs_database_annotation_df['phrog'] = phrogs_database_annotation_df['phrog'].astype(str)

    # Extract the PHROGs number from target
    phrogs_mmseqs_hits_df['phrog_number'] = phrogs_mmseqs_hits_df['protein_database_mmseqs_target'].str.extract(r'phrog_(\d+)')
    phrogs_database_annotation_df['phrog_number'] = phrogs_database_annotation_df['phrog'].str.extract(r'(\d+)')

    # Merge on 'phrog_number' to add 'annot' and 'category' columns to the hits dataframe
    phrogs_mmseqs_hits_df = phrogs_mmseqs_hits_df.merge(
        phrogs_database_annotation_df[['phrog_number', 'annot', 'category']],
        on='phrog_number',
        how='left'
    )
    phrogs_mmseqs_hits_df.drop(columns=['phrog_number'], inplace=True)

    return phrogs_mmseqs_hits_df


def annotate_protein_hits(mmseqs_hits_file: str, database_annotation_file: str) -> pd.DataFrame:
    """
    Adds protein database annotations to MMseqs hits against a given database.
    The annotation database should be a tsv file and the columns must contain:
    - hit_label: The labels used in the MMseqs database
    - annot: The desired gene name/descriptor to directly annotate on the gene
    - category: The broader protein category label to color each gene by
    """

    # Load dataframes
    mmseqs_hits_df = pd.read_csv(mmseqs_hits_file)
    database_annotation_df = pd.read_csv(database_annotation_file, sep='\t')

    # Ensure both target and hit label are treated as strings
    mmseqs_hits_df['protein_database_mmseqs_target'] = mmseqs_hits_df['protein_database_mmseqs_target'].astype(str)
    database_annotation_df['protein_database_mmseqs_target'] = database_annotation_df['hit_label'].astype(str)

    # Merge on 'protein_database_mmseqs_target' to add 'annot' and 'category' columns to the hits dataframe
    mmseqs_hits_df = mmseqs_hits_df.merge(
        database_annotation_df[['protein_database_mmseqs_target', 'annot', 'category']],
        on='protein_database_mmseqs_target',
        how='left'
    )

    return mmseqs_hits_df


def batch_create_gff_files(genomes_fasta_file: str,
                           circular_genomes_fasta_file: str,
                           genomes_csv_file: pd.DataFrame,
                           circular_orfs_fasta_file: str,
                           protein_database_hits_df: pd.DataFrame,
                           output_dir: str,
                           genome_id_map_output_dir: str) -> None:
    """
    Write a .gff file for each genome in a fasta file (should be the pseudo-circularized genomes fasta)
    using information from the original genomes fasta file, and associated ORF data, protein database hits, and tropism protein hits,
    and saves to an output directory.
    Note: This function is currently written to be compatible with PHROGs database protein hits and orfipy-predicted ORFs.
    """

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(genome_id_map_output_dir, exist_ok=True)

    matching_queries = set(genomes_csv_file['id_prompt'].to_list())  # Retrieve genome descriptors

    # Filter genomes_fasta_file to include only matching queries
    filtered_genomes_fasta = f"{output_dir}/filtered_genomes.fasta"
    with open(filtered_genomes_fasta, "w") as out_handle:
        for record in SeqIO.parse(genomes_fasta_file, "fasta"):
            if record.id in matching_queries:
                SeqIO.write(record, out_handle, "fasta")

    # Filter circular_genomes_fasta_file to include only matching queries
    filtered_circular_genomes_fasta = f"{output_dir}/filtered_circular_genomes.fasta"
    with open(filtered_circular_genomes_fasta, "w") as out_handle:
        for record in SeqIO.parse(circular_genomes_fasta_file, "fasta"):
            if record.id in matching_queries:
                SeqIO.write(record, out_handle, "fasta")

    # Assign numeric IDs only to genomes in matching_queries
    genome_id_map = assign_numeric_genome_ids(filtered_genomes_fasta, matching_queries)

    # Load protein database hits as dataframe
    #protein_database_hits_df = pd.read_csv(protein_database_hits_csv_file)

    # Extract ORF positions only for genomes in genome_id_map
    orf_data = extract_orf_positions_from_protein_database_hits(filtered_genomes_fasta, circular_orfs_fasta_file, protein_database_hits_df, genome_id_map)

    # Process each genome in the filtered circular genomes FASTA and create GFF files for matches only
    for genome_record in SeqIO.parse(filtered_circular_genomes_fasta, "fasta"):
        genome_name = genome_record.id
        if genome_name in genome_id_map:  # Only write GFF files for matching genomes
            genome_id = genome_id_map[genome_name]
            genome_seq = str(genome_record.seq)
            create_gff_file(orf_data, genome_name, genome_id, genome_seq, output_dir)
            print(f"GFF file created for {genome_name} as {genome_id}.gff")

    # Save genome ID mapping
    genome_id_map_df = pd.DataFrame(list(genome_id_map.items()), columns=["genome_name", "genome_id"])
    genome_id_map_df.to_csv(f'{genome_id_map_output_dir}/genome_id_map.csv', index=False)

    # Clean up temporary filtered FASTA files
    try:
        os.remove(filtered_genomes_fasta)
        os.remove(filtered_circular_genomes_fasta)
        print("Temporary filtered FASTA files removed.")
    except OSError as e:
        print(f"Error deleting temporary files: {e}")


def add_genome_id_mapping(genome_id_map_csv: str, input_csv: str, output_csv: str) -> None:
    """
    Merges genome_id_map.csv with a given csv by adding genome_id based on matching genome_name to id_prompt.
    Ensures the genome_id column is the second column in the output CSV.
    
    Parameters:
    - genome_id_map_csv (str): Path to the genome_id_map.csv file.
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the merged CSV.
    """

    # Load the CSV files
    genome_id_map_df = pd.read_csv(genome_id_map_csv)
    input_df = pd.read_csv(input_csv)

    # Merge the files based on genome_name (from genome_id_map) and id_prompt (from QC file)
    merged_df = pd.merge(input_df, genome_id_map_df, left_on="id_prompt", right_on="genome_name", how="left")

    # Drop the redundant genome_name column
    merged_df.drop(columns=["genome_name"], inplace=True)

    # Reorder columns to make genome_id the second column
    columns_order = ["id_prompt", "genome_id"] + [col for col in merged_df.columns if col not in ["id_prompt", "genome_id"]]
    merged_df = merged_df[columns_order]

    # Save the updated DataFrame
    merged_df.to_csv(output_csv, index=False)


def parse_gff_attributes(attributes_str: str) -> dict:
    """Parses the attributes field in a GFF line to handle values with spaces."""
    attributes = {}
    for attribute in attributes_str.split(';'):
        key_value = attribute.strip().split('=', 1)  # Split on the first '=' only
        if len(key_value) == 2:
            key, value = key_value
            attributes[key] = value
    return attributes


def convert_gff_to_gbk(gff_file: str, output_gbk: str) -> None:
    """
    Converts a GFF file (with embedded FASTA) to GenBank format.
    
    Parameters:
        gff_file (str): Path to the GFF file.
        output_gbk (str): Path to save the output GenBank file.
    """
    features = []
    fasta_sequence = None
    seq_id = None

    with open(gff_file) as gff_handle:
        for line in gff_handle:
            if line.startswith("##sequence-region"):
                seq_id = line.split()[1]  # Extract sequence ID
            elif line.startswith("##FASTA"):
                break  # End of GFF section, start of FASTA
            elif not line.startswith("#") and line.strip():
                columns = line.strip().split("\t")
                
                # Extract feature information
                feature_type = columns[2]
                start = int(columns[3])
                end = int(columns[4])
                strand = 1 if columns[6] == '+' else -1
                attributes_str = columns[8]

                # Parse attributes properly
                attributes = parse_gff_attributes(attributes_str)
                
                # Extract specific fields
                feature_id = attributes.get("ID", "")
                function = attributes.get("function", "")
                product = attributes.get("product", "")
                percent_identity = attributes.get("percent_identity", "")
                sequence = attributes.get("seq", "")
                
                # Create the SeqFeature
                feature = SeqFeature.SeqFeature(
                    location=FeatureLocation(start, end, strand=strand),
                    type=feature_type,
                    qualifiers={
                        "ID": feature_id,
                        "function": function,
                        "product": product,
                        "percent_identity": percent_identity,
                        "seq": sequence
                    }
                )
                # Add translation if provided
                if sequence:
                    feature.qualifiers["translation"] = sequence
                features.append(feature)
        
        # Parse FASTA sequence after ##FASTA section
        fasta_sequence = next(SeqIO.parse(gff_handle, "fasta"))
    
    # Construct SeqRecord for GenBank format
    seq_record = SeqRecord(
        fasta_sequence.seq,
        id=seq_id,
        name=seq_id,
        description=fasta_sequence.description,
        annotations={"molecule_type": "DNA"}  # Required annotation for GenBank
    )
    seq_record.features.extend(features)
    
    # Write to GenBank file
    SeqIO.write(seq_record, output_gbk, "genbank")
    print(f"Converted {gff_file} to {output_gbk}")


def batch_convert_gff_to_gbk(input_dir: str, output_dir: str) -> None:
    """
    Converts all GFF files in an input directory to GenBank format in an output directory.

    Parameters:
    - input_dir (str): Path to the directory containing GFF files.
    - output_dir (str): Path to the directory to save the GenBank files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gff_file in sorted(os.listdir(input_dir)):
        if gff_file.endswith(".gff"):
            input_gff_path = os.path.join(input_dir, gff_file)
            output_gbk_path = os.path.join(output_dir, f"{os.path.splitext(gff_file)[0]}.gbk")
            convert_gff_to_gbk(input_gff_path, output_gbk_path)


def activate_conda_env(env_name: str) -> None:
    """Activate a Conda environment and run commands within it."""
    try:
        # Command to initialize and activate Conda environment
        command = f"""
        eval "$(conda shell.bash hook)"
        conda deactivate
        conda activate {env_name}
        """
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def run_lovis4u_in_conda_env(env_name: str, command: str) -> None:
    """
    Activate a Conda environment and run a command within it.
    
    Args:
        env_name (str): The name of the Conda environment to activate.
        command (str): The command to run inside the activated environment.
    """
    try:
        # Full command to initialize Conda and activate environment before running the given command
        full_command = f"""
        eval "$(conda shell.bash hook)"
        conda activate {env_name}
        {command}
        """
        subprocess.run(full_command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command in Conda environment {env_name}: {e}")


def move_genetic_architecture_pdfs(genetic_architecture_visualization_results_dir: str,
                                   genetic_architecture_visualization_pdf_output_dir: str) -> None:
    """Move all genetic architecture visualization pdfs into one directory."""

    input_dir = genetic_architecture_visualization_results_dir
    output_dir = genetic_architecture_visualization_pdf_output_dir

    os.makedirs(output_dir, exist_ok=True)  # Ensure collection directory exists

    # Loop through each output folder
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        
        # Check if it's a directory and matches the 'lovis4u_genome_' pattern
        if os.path.isdir(folder_path) and folder.startswith("genome_"):
            genome_suffix = folder.split("_")[-1]  # Extract genome_# from the folder name
            pdf_path = os.path.join(folder_path, "lovis4u.pdf")
            
            # Check if lovis4u.pdf exists in the folder
            if os.path.isfile(pdf_path):
                # Define new name with the genome number suffix
                new_pdf_name = f"genome_{genome_suffix}.pdf"
                new_pdf_path = os.path.join(output_dir, new_pdf_name)
                
                # Move and rename the PDF file
                shutil.move(pdf_path, new_pdf_path)
                print(f"Moved and renamed {pdf_path} to {new_pdf_path}")
            else:
                print(f"No lovis4u.pdf found in {folder_path}")


def append_unique_identifier(df: pd.DataFrame, column_name: str, overwrite_sequence_ids: bool) -> pd.DataFrame:
    """
    Add a consecutive number/unique molecular identifier (UMI) to the end of each entry in the specified column of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): The name of the column to modify.

    Returns:
        pd.DataFrame: A new DataFrame with modified entries in the specified column.
    """
    df = df.copy()  # Avoid modifying the original DataFrame

    if overwrite_sequence_ids == True:
        for i in range(len(df)):
            if pd.notna(df.at[i, column_name]):  # Check if the value is not NaN
                df.at[i, column_name] = f"umi{i + 1}"
    else:
        for i in range(len(df)):
            if pd.notna(df.at[i, column_name]):  # Check if the value is not NaN
                df.at[i, column_name] = f"{df.at[i, column_name]}_umi{i + 1}"
    return df


def remove_prefix_from_sequences(seq_df: pd.DataFrame, column_name: str, string_to_remove: str) -> pd.DataFrame:
    """
    Removes a specified prefix string from the beginning of sequences in a specified column of a DataFrame.
    
    Parameters:
    - seq_df (pd.DataFrame): Input DataFrame containing the sequences.
    - column_name (str): The name of the column containing the sequences.
    - string_to_remove (str): The prefix string to remove from the beginning of each sequence.
    
    Returns:
    - pd.DataFrame: A DataFrame with the updated column.
    """
    seq_df = seq_df.copy()

    # Check if the column exists in the DataFrame
    if column_name not in seq_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    # Remove the prefix from the sequences
    seq_df[column_name] = seq_df[column_name].apply(
        lambda seq: seq[len(string_to_remove):] if seq.startswith(string_to_remove) else seq
    )
    
    return seq_df


def replace_spaces_in_fasta_headers(input_fasta: str, output_fasta: str) -> None:
    """
    Processes the header lines of a FASTA file, replacing spaces with underscores.

    Args:
    - input_fasta (str): Path to the input FASTA file.
    - output_fasta (str): Path to save the modified FASTA file.
    """
    with open(input_fasta, "r") as in_handle, open(output_fasta, "w") as out_handle:
        for line in in_handle:
            # Replace spaces with underscores in header lines
            if line.startswith(">"):
                line = line.replace(" ", "_")
            out_handle.write(line)


def calculate_average_protein_percent_identity(gff_directory: str, results_csv: str, output_csv: str) -> None:
    """
    Calculates the average percent_identity for each genome in a directory of GFF files.
    Combines the results with the 'id_prompt' column in the results_csv and writes a final CSV file.

    Parameters:
    - gff_directory (str): Path to the directory containing GFF files.
    - results_csv (str): Path to the CSV file containing 'id_prompt'.
    - output_csv (str): Path to the output CSV file.
    """
    results = []

    # Loop through all files in the directory
    for gff_file in os.listdir(gff_directory):
        if gff_file.endswith(".gff"):
            file_path = os.path.join(gff_directory, gff_file)

            # Initialize variables for percent_identity calculations
            total_percent_identity = 0
            num_entries = 0
            id_prompt = None

            # Parse the GFF file
            with open(file_path, "r") as f:
                for line in f:
                    # Extract id_prompt from the ##description line
                    if line.startswith("##description"):
                        id_prompt = line.strip().replace("##description", "").strip()

                    # Skip other header or comment lines
                    if line.startswith("#") or line.strip() == "":
                        continue

                    # Split the line into columns
                    columns = line.strip().split("\t")

                    # Check if it's a CDS feature
                    if len(columns) > 8 and columns[2] == "CDS":
                        attributes = columns[8]
                        # Extract percent_identity from attributes
                        attributes_dict = {key: value for key, value in [attr.split("=") for attr in attributes.split(";")]}
                        if "percent_identity" in attributes_dict:
                            percent_identity = float(attributes_dict["percent_identity"])
                            total_percent_identity += percent_identity
                            num_entries += 1

            # Calculate average percent_identity for the genome
            if id_prompt is None:
                id_prompt = gff_file.replace(".gff", "")  # Fallback to filename if description is missing
            if num_entries > 0:
                average_percent_identity = total_percent_identity / num_entries
            else:
                average_percent_identity = 0  # No CDS entries

            # Append the result
            results.append({
                "id_prompt": id_prompt,
                "average_protein_percent_identity": average_percent_identity
            })

    results_df = pd.DataFrame(results)
    latest_results_df = pd.read_csv(results_csv)
    merged_df = pd.merge(latest_results_df, results_df, on="id_prompt", how="left")
    merged_df.to_csv(output_csv, index=False)


def valid_average_protein_percent_identity(gff_directory: str, gbk_directory: str, results_csv: str, output_csv: str, identity_range: tuple) -> None:
    """
    Processes GFF files to calculate average percent identity and count total genes.
    Merges results with an existing CSV and filters out entries based on identity_range.
    Deletes both GFF and GBK files that do not pass the filter.

    Parameters:
    - gff_directory (str): Path to the directory containing GFF files.
    - gbk_directory (str): Path to the directory containing GBK files.
    - results_csv (str): Path to the CSV file containing 'id_prompt'.
    - output_csv (str): Path to save the final filtered CSV file.
    - identity_range (tuple): (min_value, max_value) specifying the valid range for filtering.
    """

    results = []
    min_value, max_value = identity_range
    files_to_delete = []  # Stores file paths of GFF and GBK files to delete

    # Loop through all GFF files
    for gff_file in os.listdir(gff_directory):
        if gff_file.endswith(".gff"):
            gff_path = os.path.join(gff_directory, gff_file)
            gbk_path = os.path.join(gbk_directory, gff_file.replace(".gff", ".gbk"))  # Corresponding GBK file

            # Initialize variables
            total_percent_identity = 0
            num_entries = 0
            num_genes = 0
            id_prompt = None

            # Parse the GFF file
            with open(gff_path, "r") as f:
                for line in f:
                    # Extract id_prompt from the ##description line
                    if line.startswith("##description"):
                        id_prompt = line.strip().replace("##description", "").strip()

                    # Skip header or comment lines
                    if line.startswith("#") or line.strip() == "":
                        continue

                    # Split the line into columns
                    columns = line.strip().split("\t")

                    # Check if it's a CDS feature
                    if len(columns) > 8 and columns[2] == "CDS":
                        num_genes += 1  # Count CDS features
                        attributes = columns[8]
                        attributes_dict = {key: value for key, value in [attr.split("=") for attr in attributes.split(";")]}
                        if "percent_identity" in attributes_dict:
                            percent_identity = float(attributes_dict["percent_identity"])
                            total_percent_identity += percent_identity
                            num_entries += 1

            # Use the filename as a fallback for id_prompt
            if id_prompt is None:
                id_prompt = gff_file.replace(".gff", "")

            # Calculate average percent identity
            average_percent_identity = total_percent_identity / num_entries if num_entries > 0 else 0

            # Store results
            results.append({
                "id_prompt": id_prompt,
                "average_protein_percent_identity": average_percent_identity,
                #"total_num_genes": num_genes
            })

            # Mark for deletion if it doesn't pass the filter
            if not (min_value <= average_percent_identity <= max_value):
                files_to_delete.append(gff_path)  # Delete GFF
                if os.path.exists(gbk_path):
                    files_to_delete.append(gbk_path)  # Delete corresponding GBK if it exists

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Read the existing results CSV
    latest_results_df = pd.read_csv(results_csv)

    # Merge the results with the existing CSV
    merged_df = pd.merge(latest_results_df, results_df, on="id_prompt", how="left")

    # Apply the filtering step
    filtered_df = merged_df[(merged_df['average_protein_percent_identity'] >= min_value) & 
                            (merged_df['average_protein_percent_identity'] <= max_value)]

    # Save the updated DataFrame
    filtered_df.to_csv(output_csv, index=False)

    # Delete the GFF and GBK files that did not pass the filter
    for file_path in files_to_delete:
        os.remove(file_path)


def count_total_num_genes(gff_directory: str, results_csv: str) -> None:
    """
    Counts the number of CDS features in each GFF file and adds this count as a new column 'total_num_genes'.
    Merges the results with the 'id_prompt' column in the results_csv and writes the final CSV file.

    Parameters:
    - gff_directory (str): Path to the directory containing GFF files.
    - results_csv (str): Path to the CSV file containing 'id_prompt' which will be overwritten with the CDS counts
    """
    results = []

    # Loop through all files in the directory
    for gff_file in os.listdir(gff_directory):
        if gff_file.endswith(".gff"):
            file_path = os.path.join(gff_directory, gff_file)

            # Initialize variables
            num_genes = 0
            id_prompt = None

            # Parse the GFF file
            with open(file_path, "r") as f:
                for line in f:
                    # Extract id_prompt from the ##description line
                    if line.startswith("##description"):
                        id_prompt = line.strip().replace("##description", "").strip()

                    # Skip header or comment lines
                    if line.startswith("#") or line.strip() == "":
                        continue

                    # Split the line into columns
                    columns = line.strip().split("\t")

                    # Check if it's a CDS feature
                    if len(columns) > 8 and columns[2] == "CDS":
                        num_genes += 1  # Count CDS features

            # Use the filename as fallback if description is missing
            if id_prompt is None:
                id_prompt = gff_file.replace(".gff", "")

            # Append the result
            results.append({
                "id_prompt": id_prompt,
                "total_num_genes": num_genes
            })
    
    # Update and overwrite the csv
    results_df = pd.DataFrame(results)
    latest_results_df = pd.read_csv(results_csv)
    merged_df = pd.merge(latest_results_df, results_df, on="id_prompt", how="left")
    merged_df.to_csv(results_csv, index=False)


def count_syntenic_genes_all(root_dir: str, gff_dir: str, input_csv: str, output_csv: str) -> None:
    """
    Counts the number of syntenic genes in MMseqs results from all lovis4u folders and merges the results with an input CSV by mapping to `id_prompt`.
    Also records non-syntenic genes and their annotations as columns.
    """

    def extract_prefix(identifier: str) -> str:
        """ Extracts the portion of the string before 'ORF'. """
        match = re.match(r"^(.*?)ORF", identifier)
        return match.group(1) if match else identifier  # Return full string if no match

    def extract_gene_name(identifier: str) -> str:
        """ Extracts the gene name from the given identifier in the format 'genome_#-ORF.#'. """
        match = re.search(r"ORF\.\d+", identifier)
        return match.group(0) if match else identifier

    def count_syntenic_genes(file_path: str, gff_path: str) -> tuple:
        """
        Counts the number of syntenic genes between two genomes and identifies non-syntenic genes.
        Also retrieves gene annotations from the GFF file.

        Parameters:
            file_path (str): Path to the TSV file.
            gff_path (str): Path to the GFF file.

        Returns:
            tuple: The count of valid non-identical gene pairs, a comma-separated list of non-syntenic genes,
                and a comma-separated list of their annotations.
        """
        df = pd.read_csv(file_path, sep="\t", header=None, names=["col1", "col2"])

        # Find mismatches
        mismatched = df[df["col1"] != df["col2"]].copy()

        # Extract prefixes before "ORF"
        mismatched["prefix1"] = mismatched["col1"].apply(extract_prefix)
        mismatched["prefix2"] = mismatched["col2"].apply(extract_prefix)

        # Ensure prefixes are also not identical
        valid_syntenic_pairs = mismatched[mismatched["prefix1"] != mismatched["prefix2"]]
        #print(f'Valid syntenic pairs: {valid_syntenic_pairs}')

        # Count valid syntenic pairs
        syntenic_count = len(valid_syntenic_pairs)

        # Get syntenic genes from valid syntenic pairs where the genome prefix matches
        syntenic_genes_col1 = set(valid_syntenic_pairs[valid_syntenic_pairs["col1"].str.startswith("genome_")]["col1"].apply(extract_gene_name))
        #print(f'Syntenic genes col1: {syntenic_genes_col1}')
        syntenic_genes_col2 = set(valid_syntenic_pairs[valid_syntenic_pairs["col2"].str.startswith("genome_")]["col2"].apply(extract_gene_name))
        #print(f'Syntenic genes col2: {syntenic_genes_col2}')
        syntenic_genes = syntenic_genes_col1.union(syntenic_genes_col2)
        #print(f'Final syntenic genes: {syntenic_genes}')

        # Extract all genes and their annotations from the GFF file
        all_genes = {}
        with open(gff_path, "r") as gff_file:
            for line in gff_file:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                attributes = fields[8]

                # Extract gene ID
                gene_id_match = re.search(r"ID=([^;]+)", attributes)
                gene_id = gene_id_match.group(1) if gene_id_match else None

                # Extract gene annotation (product)
                product_match = re.search(r"product=([^;]+)", attributes)
                product = product_match.group(1) if product_match else "Unknown gene"

                # Handle cases where the product is empty or 'nan'
                if not product or product.lower() == "nan":
                    product = "Unknown gene"

                # Record the gene and its annotation if it has a valid ID
                if gene_id and gene_id.startswith("ORF"):
                    all_genes[gene_id] = product

        # Identify non-syntenic genes and their annotations (starting with ORF only)
        non_syntenic_genes = {gene for gene in all_genes.keys() if gene.startswith("ORF")} - syntenic_genes
        print(f'Non-syntenic genes: {non_syntenic_genes}')
        non_syntenic_annotations = [all_genes[gene] for gene in sorted(non_syntenic_genes)]

        return syntenic_count, ",".join(sorted(non_syntenic_genes)), ",".join(non_syntenic_annotations)

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    syntenic_counts = {}
    non_syntenic_genes_dict = {}
    non_syntenic_annotations_dict = {}

    # Iterate through lovis4u subfolders
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if os.path.isdir(subfolder_path) and subfolder.startswith("genome_"):
            mmseqs_path = os.path.join(subfolder_path, "mmseqs", "mmseqs_clustering.tsv")
            genome_name = subfolder
            gff_path = os.path.join(gff_dir, f"{genome_name}.gff")

            if os.path.exists(mmseqs_path) and os.path.exists(gff_path):
                syntenic_count, non_syntenic_genes, non_syntenic_annotations = count_syntenic_genes(mmseqs_path, gff_path)
                syntenic_counts[genome_name] = syntenic_count
                non_syntenic_genes_dict[genome_name] = non_syntenic_genes
                non_syntenic_annotations_dict[genome_name] = non_syntenic_annotations
                print(f"Processed {genome_name}: {syntenic_count} syntenic genes, non-syntenic: {non_syntenic_genes}")
            else:
                print(f"Warning: Required file not found for {genome_name}")

    # Load the input CSV
    input_df = pd.read_csv(input_csv)

    # Map syntenic counts, non-syntenic genes, and annotations to the DataFrame
    input_df["num_syntenic_genes"] = input_df["genome_id"].map(syntenic_counts).fillna(0).astype(int)
    input_df["non_syntenic_genes"] = input_df["genome_id"].map(non_syntenic_genes_dict).fillna("")
    input_df["non_syntenic_annotations"] = input_df["genome_id"].map(non_syntenic_annotations_dict).fillna("")

    # Save the updated DataFrame
    input_df.to_csv(output_csv, index=False)


def valid_syntenic_gene_count(input_csv: str, output_csv: str,
                              syntenic_gene_count_range: list, total_gene_count_range: list, syntenic_total_gene_count_remove: set,
                              gff_dir: str, gbk_dir: str, pdf_dir: str, metadata_dir: str) -> None:
    """
    Filters sequences based on combinations of `num_syntenic_genes` and `total_num_genes` ranges.
    Also deletes corresponding GFF, GBK, PDF files and metadata folders for filtered-out genomes.
    """
    df = pd.read_csv(input_csv)

    syntenic_values = range(syntenic_gene_count_range[0], syntenic_gene_count_range[1] + 1)
    total_genes_values = range(total_gene_count_range[0], total_gene_count_range[1] + 1)
    valid_combinations = set(itertools.product(syntenic_values, total_genes_values))

    syntenic_total_gene_count_remove = {tuple(pair) for pair in syntenic_total_gene_count_remove}
    valid_combinations -= syntenic_total_gene_count_remove

    filtered_df = df[df[['num_syntenic_genes', 'total_num_genes']].apply(tuple, axis=1).isin(valid_combinations)]
    removed_ids = set(df["genome_id"]) - set(filtered_df["genome_id"])

    filtered_df.to_csv(output_csv, index=False)

    for genome_id in removed_ids:
        for ext, dir_path in [("gff", gff_dir), ("gbk", gbk_dir), ("pdf", pdf_dir), ("", metadata_dir)]:
            file_path = os.path.join(dir_path, f"{genome_id}.{ext}" if ext else genome_id)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"Deleted: {file_path}")


def valid_gene_annotations(input_gff_dir: str, input_gbk_dir: str, required_products: tuple, sequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates gene annotations and deletes files/directories that do not contain the required gene products.
    Returns a filtered version of the provided DataFrame based on surviving genome_ids.

    Parameters:
    - input_gff_dir (str): Directory containing GFF files.
    - input_gbk_dir (str): Directory containing GBK files.
    - required_products (tuple): A tuple of required gene products to validate.
    - sequences_df (pd.DataFrame): DataFrame containing a 'genome_id' column to filter.

    Returns:
    - pd.DataFrame: Filtered DataFrame based on genome_ids that passed validation.
    """
    surviving_genome_ids = set()
    gff_files = glob.glob(os.path.join(input_gff_dir, "*.gff"))

    for gff_file in gff_files:
        extracted_products = []
        genome_id = os.path.basename(gff_file).replace(".gff", "")

        with open(gff_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue

                attributes = fields[8]
                for attr in attributes.split(";"):
                    if attr.startswith("product="):
                        product = attr.split("=")[1]
                        extracted_products.append(product)

        # Keep if all required products are present
        if all(product in extracted_products for product in required_products):
            print(f"Kept: {gff_file}")
            surviving_genome_ids.add(genome_id)
        else:
            print(f"Discarded: {gff_file}")
            os.remove(gff_file)

            gbk_file = os.path.join(input_gbk_dir, f"{genome_id}.gbk")
            if os.path.exists(gbk_file):
                os.remove(gbk_file)
                print(f"Deleted: {gbk_file}")

            genome_dir = os.path.join(input_gff_dir, genome_id)
            if os.path.exists(genome_dir):
                shutil.rmtree(genome_dir)
                print(f"Deleted directory: {genome_dir}")

    # Filter and return DataFrame
    filtered_df = sequences_df[sequences_df["genome_id"].isin(surviving_genome_ids)].copy()
    return filtered_df


##############################
### RUN FILTERING PIPELINE ###
##############################

def main(config_file):
    """Run genome design filtering pipeline."""

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    #########################
    ### 1. INITIALIZATION ###
    #########################

    ### Initialize results directory ###
    ensure_directory_exists(f'{config["results_save_dir"]}')

    ### Load sequences ###
    if config["evo_gen_seqs_fasta_file_save_location"].endswith('.fna') or config["evo_gen_seqs_fasta_file_save_location"].endswith('.fasta'):
        replace_spaces_in_fasta_headers(config["evo_gen_seqs_fasta_file_save_location"], f'{config["results_save_dir"]}/{config["initialized_seqs_fasta_file_save_location"]}')
    else:
        raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")

    if config["keep_only_up_to_first_eos"] == True:
        seq_df = load_fasta_to_df_eos_aware(f'{config["results_save_dir"]}/{config["initialized_seqs_fasta_file_save_location"]}')
        print(f"Loading {len(seq_df)} sequences trimmed after the first EOS token for nucleotide filtering...")
    elif config["keep_only_up_to_first_eos"] == False:
        seq_df = load_fasta_to_df(f'{config["results_save_dir"]}/{config["initialized_seqs_fasta_file_save_location"]}')
        print(f"Loading {len(seq_df)} sequences for nucleotide filtering...")
    
    ### Append unique identifiers to sequence IDs ###
    print(f"Appending unique sequence identifiers...")
    seq_df = append_unique_identifier(seq_df, "id_prompt", config["overwrite_sequence_ids"])
    print(f"Finished appending unique sequence identifiers.")

    ### Ensure all sequences are in same case ###
    print(f"Making all sequences uppercase...")
    seq_df['sequence'] = seq_df['sequence'].str.upper()
    print(f"Finished making all sequences uppercase.")

    ### Remove prompting tokens if necessary ###
    if config["remove_prompt"] == True:
        print(f"Removing prompting tokens...")
        seq_df = remove_prefix_from_sequences(seq_df, "sequence", config["prompt_to_remove"])
        print(f"Finished removing prompting tokens.")

    ### Prepend prompting tokens if necessary ###
    if config["prepend_prompt"] == True:
        prepend_prompt = config["prompt"]
        print(f"Prepending sequences with the prompt {prepend_prompt}...")
        seq_df['sequence'] = prepend_prompt + seq_df['sequence'].astype(str)
        print("Finished prepending prompt sequences.")

    ### Save checkpoint ###
    seq_df.to_csv(f'{config["results_save_dir"]}/{config["initialized_seqs_csv_file_save_location"]}', index=False)
    save_df_as_fasta(seq_df, f'{config["results_save_dir"]}/{config["initialized_seqs_fasta_file_save_location"]}')
    print(f"Completed initial clean-up of {len(seq_df)} sequences.")


    ###############################
    ### 2. NUCLEOTIDE FILTERING ###
    ###############################

    ### Filter by nucleotide metrics ###
    if config["nucleotide_filtering"] == True:

        ### Initialize counts ###
        filter_counts = {}
        filter_counts = pd.DataFrame([filter_counts])
        filter_counts['count_initial_before_nucleotide_metrics'] = len(seq_df)
        filtered_df = seq_df.copy()
        print(f"Initializing nucleotide filtering. Sequences to filter: {filter_counts['count_initial_before_nucleotide_metrics'].values[0]}.")

        ### Filter by nucleotide characters ###
        if config["nucleotide_character_filter"] == True:
            print("Filtering by nucleotide characters...")
            filtered_df = valid_nt_chars(seq_df)
            filter_counts['count_nt_filter'] = len(filtered_df)
            print(f"Completed nucleotide character filtering resulting in {filter_counts['count_nt_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by genome length ###
        if config["genome_length_filter"] == True:
            print(f"Filtering by genome length with threshold {config['genome_length_range']}...")
            filtered_df = valid_genome_len(filtered_df, config["genome_length_range"])
            filter_counts['count_genome_len_filter'] = len(filtered_df)
            print(f"Completed genome length filtering resulting in {filter_counts['count_genome_len_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by GC content ###
        if config["gc_content_filter"] == True:
            print(f"Filtering by GC content with threshold {config['gc_content_range']}...")
            filtered_df = valid_gc_content(filtered_df, config["gc_content_range"])
            filter_counts['count_gc_filter'] = len(filtered_df)
            print(f"Completed GC content filtering resulting in {filter_counts['count_gc_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by nucleotide homopolymer lengths ###
        if config["nucleotide_homopolymer_filter"] == True:
            print(f"Filtering by nucleotide homopolymer length with threshold {config['nucleotide_homopolymer_length_range']}...")
            filtered_df = valid_nt_homopolymer_len(filtered_df, config["nucleotide_homopolymer_length_range"])
            filter_counts['count_nt_homopolymer_filter'] = len(filtered_df)
            print(f"Completed nucleotide homopolymer filtering resulting in {filter_counts['count_nt_homopolymer_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by dinucleotide frequencies ###
        if config["dinucleotide_frequency_filter"] == True:
            print(f"Filtering by dinucleotide frequencies with threshold {config['dinucleotide_frequency_range']}...")
            filtered_df = valid_dinucleotide_content(filtered_df, config["dinucleotide_frequency_range"])
            filter_counts['count_dinucleotide_filter'] = len(filtered_df)
            print(f"Completed dinucleotide frequency filtering resulting in {filter_counts['count_dinucleotide_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by tetranucleotide usage deviation ###
        if config["tud_filter"] == True:
            print(f"Filtering by tetranucleotide usage deviation with threshold {config['tud_range']}...")
            filtered_df = valid_tud(filtered_df, config["tud_range"])
            filter_counts['count_tud_filter'] = len(filtered_df)
            print(f"Completed TUD filtering resulting in {filter_counts['count_tud_filter'].values[0]} sequences.")

        ### Save final filter counts and sequences with filter metrics ###
        filter_counts.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}', index=False)
        filtered_df.to_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_fasta_file_save_location"]}')
        print(f"Completed all nucleotide filtering. Final count: {len(filtered_df)} sequences.")


    ########################
    ### 3. ORF FILTERING ###
    ########################

    ### Filter by ORF metrics ###
    if config["orf_filtering"] == True:

        ### Re-load sequences ###
        if config["use_nucleotide_filtered_df"] == True:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}')
            filter_counts_df = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} nucleotide-filtered sequences for ORF filtering...")
        else:
            if config["evo_gen_seqs_fasta_file_save_location"].endswith('.fna') or config["evo_gen_seqs_fasta_file_save_location"].endswith('.fasta'):
                seq_df = load_fasta_to_df(config["evo_gen_seqs_fasta_file_save_location"])
                filter_counts = {}
                filter_counts_df = pd.DataFrame([filter_counts])
                print(f"Loading {len(seq_df)} sequences for ORF filtering...")
            else:
                raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")

        ### Initialize counts ###
        filter_counts_df['count_initial_before_orf_metrics'] = len(seq_df)
        print(f"Initializing ORF filtering. Sequences to filter: {filter_counts_df['count_initial_before_orf_metrics'].values[0]}.")

        ### Run Prodigal to call ORFs ###
        if config["prodigal_based_filters"] == True:
            print("Running Prodigal...")
            run_prodigal(input_sequences=f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_fasta_file_save_location"]}',
                         output_orf_file=f'{config["results_save_dir"]}/{config["prodigal_orfs_file_save_location"]}',
                         output_protein_file=f'{config["results_save_dir"]}/{config["prodigal_proteins_file_save_location"]}',
                         sequences_df=seq_df)
            print("Completed running Prodigal.")

            ### Filter by Prodigal ORF counts ###
            if config["orf_count_filter"] == True:
                print(f"Filtering by ORF counts with threshold {config['orf_count_range']}...")
                filtered_df = valid_orf_count(prodigal_orfs=f'{config["results_save_dir"]}/{config["prodigal_orfs_file_save_location"]}',
                                            orf_count_range=config["orf_count_range"],
                                            sequences_df=seq_df)
                filter_counts_df['count_orf_count_filter'] = len(filtered_df)
                print(f"Completed ORF count filtering resulting in {filter_counts_df['count_orf_count_filter'].values[0]} sequences.")
                # Save checkpoint
                filter_counts_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}', index=False)
                filtered_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}', index=False)

            ### Filter by Prodigal ORF lengths ###
            if config["orf_lengths_filter"] == True:
                print(f"Filtering by ORF lengths with threshold {config['orf_lengths_range']}...")
                filtered_df = valid_orf_lengths(prodigal_orfs=f'{config["results_save_dir"]}/{config["prodigal_orfs_file_save_location"]}',
                                                orf_length_range=config["orf_lengths_range"],
                                                sequences_df=filtered_df)
                filter_counts_df['count_orf_len_filter'] = len(filtered_df)
                print(f"Completed ORF length filtering resulting in {filter_counts_df['count_orf_len_filter'].values[0]} sequences.")
                # Save checkpoint
                filter_counts_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}', index=False)
                filtered_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}', index=False)

            ### Filter by Prodigal coding density ###
            if config["coding_density_filter"] == True:
                print(f"Filtering by coding density with threshold {config['coding_density_range']}...")
                filtered_df = valid_coding_density(sequences_df=filtered_df,
                                                coding_density_range=config['coding_density_range'])
                filter_counts_df['count_coding_density_filter'] = len(filtered_df)
                print(f"Completed coding density filtering resulting in {filter_counts_df['count_coding_density_filter'].values[0]} sequences.")
                # Save checkpoint
                filter_counts_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}', index=False)
                filtered_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}', index=False)

            ### Filter by amino acid homopolymer lengths ###
            if not filtered_df.empty:
                if config["aminoacid_homopolymer_length_filter"] == True:
                    print(f"Filtering by amino acid homopolymer lengths with threshold {config['aminoacid_homopolymer_length_range']}...")
                    filtered_df = valid_aa_homopolymer_len(prodigal_proteins=f'{config["results_save_dir"]}/{config["prodigal_proteins_file_save_location"]}', 
                                                        homopolymer_length_range=config["aminoacid_homopolymer_length_range"], 
                                                        sequences_df=filtered_df)
                    filter_counts_df['count_aa_homopolymer_len_filter'] = len(filtered_df)
                    print(f"Completed peptide homopolymer filtering resulting in {filter_counts_df['count_aa_homopolymer_len_filter'].values[0]} sequences.")
                    # Save checkpoint
                    filter_counts_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}', index=False)
                    filtered_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}', index=False)

        ### Save final filter counts and sequences with filter metrics ###
        if config["prodigal_based_filters"] == False:
            filtered_df = seq_df.copy()
        filter_counts_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}', index=False)
        filtered_df.to_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["orf_filter_seqs_fasta_file_save_location"]}')
        print(f"Completed all ORF filtering. Final count: {len(filtered_df)} sequences.")


    #############################
    ### 4. HOMOLOGY FILTERING ###
    #############################

    ### Filter by homology metrics ###
    if config["homology_filtering"] == True:

        ### Re-load sequences ###
        if config["use_orf_filtered_df"] == True and config["use_nucleotide_filtered_df_instead"] == False:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["orf_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} ORF-filtered sequences for homology filtering...")
        elif config["use_orf_filtered_df"] == False and config["use_nucleotide_filtered_df_instead"] == True:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} nucleotide-filtered sequences for homology filtering...")
        else:
            # Load sequences based on file extension
            if config["evo_gen_seqs_fasta_file_save_location"].endswith('.fna') or config["evo_gen_seqs_fasta_file_save_location"].endswith('.fasta'):
                seq_df = load_fasta_to_df(config["evo_gen_seqs_fasta_file_save_location"])
                seq_fasta = config["evo_gen_seqs_fasta_file_save_location"]
                filter_counts = {}
                filter_counts = pd.DataFrame([filter_counts])
                print(f"Loading {len(seq_df)} sequences for homology filtering...")
            else:
                raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")

        ### Run orfipy to call ORFs ###
        # Pseudo-circularize ORFs
        print(f"Pseudo-circularizing {len(seq_df)} genomes...")
        append_upstream_of_last_frame_stop(seq_fasta, f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}')
        # Call ORFs by orfipy
        print(f"Running orfipy on {len(seq_df)} genomes...")
        run_orfipy(f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}',
                    config["orfipy_threads"],
                    config["orfipy_start_codons"],
                    config["orfipy_stop_codons"],
                    config["orfipy_strand"],
                    config["orfipy_min_max_orf_lengths"][0],
                    config["orfipy_min_max_orf_lengths"][1],
                    config["results_save_dir"],
                    config["orfipy_orfs_file_save_location"], 
                    config["orfipy_tmp_proteins_file_save_location"], 
                    config["orfipy_proteins_file_save_location"])

        ### Filter by hit counts to protein database ###
        if config["protein_database_hit_count_filter"] == True:
            print(f"Filtering by hit counts to protein database {config['mmseqs_db_protein_database']}...")
            mmseqs_results_df = run_mmseqs_search_proteins(query_fasta=f'{config["results_save_dir"]}/{config["orfipy_proteins_file_save_location"]}',
                                                mmseqs_db=config["mmseqs_db_protein_database"],
                                                results_dir=f'{config["results_save_dir"]}/{config["mmseqs_protein_database_results_dir_save_location"]}',
                                                output_csv=f'{config["results_save_dir"]}/{config["mmseqs_protein_database_results_dir_save_location"]}/mmseqs2_hits.csv',
                                                descriptive_prefix='protein_database',
                                                threads=config["mmseqs_threads"],
                                                split=0,
                                                sensitivity=config["mmseqs_protein_database_sensitivity"],
                                                only_top_hits=True)
            filtered_df = valid_protein_database_hit_count(mmseqs_results_df, seq_df, 'id_prompt', config["protein_database_hit_count"])
            filter_counts['count_protein_database_hit_count_filter'] = len(filtered_df)
            print(f"Completed protein database hit count filtering resulting in {filter_counts['count_protein_database_hit_count_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}')

        ### Filter by sequence identity to training data genomes ###
        if config["training_data_sequence_identity_filter"] == True:
            print(f"Filtering by sequence identity to training data genomes in {config['training_data_genomes_fasta']}...")
            if os.path.exists(f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}'):
                shutil.rmtree(f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}')
            run_mmseqs_search_genomes(query_genomes=f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}',
                                      target_genomes=config["training_data_genomes_fasta"],
                                      query_db_dir=f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}/query_db_dir',
                                      target_db_dir=f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}/target_db_dir',
                                      tmp_dir=f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}/tmp',
                                      results_dir=f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}/results_dir',
                                      results_filename="mmseqs_results.m8",
                                      threads=config["mmseqs_threads"],
                                      sensitivity=config["mmseqs_training_data_sensitivity"])
            mmseqs_results_df = convert_m8_to_df(f'{config["results_save_dir"]}/{config["mmseqs_training_data_results_dir_save_location"]}/results_dir/mmseqs_results.m8',
                                                 'training_data')
            filtered_df = valid_mmseqs_pident(mmseqs_results_df, "training_data", config["training_data_sequence_identity_range"], filtered_df)
            filter_counts['count_training_data_sequence_identity_filter'] = len(filtered_df)
            print(f"Completed training data sequence identity filtering resulting in {filter_counts['count_training_data_sequence_identity_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}')

        ### Filter by CheckV quality ###
        if config["checkv_filter"] == True:
            print(f"Filtering by CheckV quality with threshold {config['checkv_quality_range']}...")
            run_checkv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}',
                       f'{config["results_save_dir"]}/{config["checkv_results_dir_save_location"]}',
                       config["checkv_threads"])
            filtered_df = valid_checkv_quality(f'{config["results_save_dir"]}/{config["checkv_results_dir_save_location"]}/quality_summary.tsv', config["checkv_quality_range"], filtered_df)
            filter_counts['count_checkv_quality_filter'] = len(filtered_df)
            print(f"Completed CheckV quality filtering resulting in {filter_counts['count_checkv_quality_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by sequence identity to reference genome ###
        if config["reference_genome_sequence_identity_filter"] == True:
            print(f"Filtering by sequence identity to reference genome {config['reference_genome_fasta']} with threshold {config['reference_genome_sequence_identity_range']}...")
            filtered_df = valid_reference_genome_pident(filtered_df, config["reference_genome_fasta"], config["reference_genome_sequence_identity_range"])
            filter_counts['count_seq_ident_to_reference_genome_filter'] = len(filtered_df)
            print(f"Completed reference genome sequence identity filtering resulting in {filter_counts['count_seq_ident_to_reference_genome_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by genetic architecture score ###
        if config["genetic_architecture_filter"] == True:
            print(f"Filtering by genetic architecture score with threshold {config['genetic_architecture_score_range']} using the reference genome {config['genetic_architecture_reference_genome']}...")
            filtered_df = valid_genetic_architecture_score(filtered_df, 
                                                           ga.phix174_truth_matrix_blurred_sigma5, 
                                                           ga.phix174_weight_vector, 
                                                           ga.phix174_normalization_vector_blurred_sigma5, 
                                                           config["genetic_architecture_score_range"])
            filter_counts['count_genetic_architecture_score_filter'] = len(filtered_df)
            print(f"Completed genetic architecture score filtering resulting in {filter_counts['count_genetic_architecture_score_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)
            #filtered_df.to_csv(f'{config["results_save_dir"]}/tmp_df_with_arch_scores_before_tropismfilter.csv', index=False) # Used for prompt length sweeps

        ### Filter by sequence identity to tropism protein ###
        if config["tropism_protein_sequence_identity_filter"] == True:
            print(f"Filtering by sequence identity to tropism protein {config['reference_tropism_protein']} with threshold {config['tropism_protein_sequence_identity_range']}...")
            mmseqs_results_df = run_mmseqs_search_proteins(query_fasta=f'{config["results_save_dir"]}/{config["orfipy_proteins_file_save_location"]}',
                                                           mmseqs_db=config["mmseqs_db_tropism_protein"],
                                                           results_dir=f'{config["results_save_dir"]}/{config["mmseqs_tropism_protein_results_dir_save_location"]}',
                                                           output_csv=f'{config["results_save_dir"]}/{config["mmseqs_tropism_protein_results_dir_save_location"]}/mmseqs2_hits.csv',
                                                           descriptive_prefix='tropism_protein',
                                                           threads=config["mmseqs_threads"],
                                                           split=0,
                                                           sensitivity=config["mmseqs_tropism_protein_sensitivity"],
                                                           only_top_hits=False)
            filtered_df = valid_mmseqs_pident(mmseqs_results_df, "tropism_protein", config["tropism_protein_sequence_identity_range"], filtered_df)
            filter_counts['count_tropism_protein_sequence_identity_filter'] = len(filtered_df)
            print(f"Completed tropism protein sequence identity filtering resulting in {filter_counts['count_tropism_protein_sequence_identity_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)
            save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}')

        ### Save final filter counts and sequences with filter metrics ###
        filter_counts.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}', index=False)
        filtered_df.to_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}')
        seq_fasta = f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}'
        seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}')
        print(f"Completed all homology filtering. Final count: {len(filtered_df)} sequences.")


    ####################################
    ### 5. DIVERSIFICATION FILTERING ###
    ####################################

    ### Filter to increase diversity of final candidates ###
    if config["diversification_filtering"] == True:

        ### Re-load sequences ###
        if config["use_homology_filtered_df"] == True and config["use_orf_filtered_df_instead"] == False and config["use_nucleotide_filtered_df_instead_2"] == False:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} homology-filtered sequences for diversification filtering...")
        elif config["use_homology_filtered_df"] == False and config["use_orf_filtered_df_instead"] == True and config["use_nucleotide_filtered_df_instead_2"] == False:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["orf_filter_seqs_csv_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["orf_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["orf_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} ORF-filtered sequences for diversification filtering...")
        elif config["use_homology_filtered_df"] == False and config["use_orf_filtered_df_instead"] == False and config["use_nucleotide_filtered_df_instead_2"] == True:
            seq_df = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_csv_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["nucleotide_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["nucleotide_filter_counts_file_save_location"]}')
            print(f"Loading {len(seq_df)} nucleotide-filtered sequences for homology filtering...")
        else:
            # Load sequences based on file extension
            if config["evo_gen_seqs_fasta_file_save_location"].endswith('.fna') or config["evo_gen_seqs_fasta_file_save_location"].endswith('.fasta'):
                seq_df = load_fasta_to_df(config["evo_gen_seqs_fasta_file_save_location"])
                seq_fasta = config["evo_gen_seqs_fasta_file_save_location"]
                filter_counts = {}
                filter_counts = pd.DataFrame([filter_counts])
                print(f"Loading {len(seq_df)} sequences for homology filtering...")
            else:
                raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")
        filtered_df = seq_df.copy()

        ### Filter by MMseqs clustering ###
        if config["mmseqs_clustering_filter"] == True:
            print(f"Filtering by MMseqs clustering with minimum sequence identity {config['mmseqs_clustering_min_sequence_identity']}...")
            run_mmseqs_clustering(input_fasta=seq_fasta,
                                  output_dir=f'{config["results_save_dir"]}/{config["mmseqs_clustering_results_dir_save_location"]}',
                                  min_seq_id=config["mmseqs_clustering_min_sequence_identity"])
            filtered_df = extract_mmseqs_cluster_representatives(clusters_tsv=f'{config["results_save_dir"]}/{config["mmseqs_clustering_results_dir_save_location"]}/mmseqs_results/clusters.tsv',
                                                                 input_fasta=seq_fasta,
                                                                 output_fasta=f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}',
                                                                 input_df=seq_df)
            filter_counts['count_mmseqs_clustering_filter'] = len(filtered_df)
            print(f"Completed MMseqs clustering filtering resulting in {filter_counts['count_mmseqs_clustering_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}', index=False)
            save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}')
            seq_fasta = f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}'
            seq_df = filtered_df.copy()

        ## Filter by MMseqs reference genome sequence identity removal ###
        if config["mmseqs_reference_genome_sequence_identity_remove_filter"] == True:
            print(f"Filtering by MMseqs sequence identity to reference genome {config['reference_genome_fasta']} with remove threshold outside {config['mmseqs_reference_genome_sequence_identity_keep_range']}...")
            if os.path.exists(f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}'):
                shutil.rmtree(f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}')
            run_mmseqs_search_genomes(query_genomes=seq_fasta,
                                      target_genomes=config["reference_genome_fasta"],
                                      query_db_dir=f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}/query_db_dir',
                                      target_db_dir=f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}/target_db_dir',
                                      tmp_dir=f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}/tmp',
                                      results_dir=f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}/results_dir',
                                      results_filename="mmseqs_results.m8",
                                      threads=config["mmseqs_threads"],
                                      sensitivity=config["mmseqs_reference_genome_sensitivity"])
            mmseqs_results_df = convert_m8_to_df(f'{config["results_save_dir"]}/{config["mmseqs_reference_genome_results_dir_save_location"]}/results_dir/mmseqs_results.m8',
                                                 'reference_genome')
            filtered_df = valid_mmseqs_pident(mmseqs_results_df, "reference_genome", config["mmseqs_reference_genome_sequence_identity_keep_range"], seq_df)
            filter_counts['count_mmseqs_reference_genome_sequence_identity_remove_filter'] = len(filtered_df)
            print(f"Completed MMseqs reference genome sequence identity removal filtering resulting in {filter_counts['count_mmseqs_reference_genome_sequence_identity_remove_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}', index=False)

        ### Filter by genetic architecture score removal ###
        if config["genetic_architecture_remove_filter"] == True:
            print(f"Filtering by genetic architecture score with remove threshold {config['genetic_architecture_score_range_to_remove']} using the reference genome {config['genetic_architecture_reference_genome']}...")
            filtered_df = valid_genetic_architecture_score(filtered_df,
                                                           ga.phix174_truth_matrix_blurred_sigma5, 
                                                           ga.phix174_weight_vector, 
                                                           ga.phix174_normalization_vector_blurred_sigma5,
                                                           config["genetic_architecture_score_range_to_remove"],
                                                           "remove",
                                                           config["genetic_architecture_score_mode"])
            filter_counts['count_genetic_architecture_score_remove_filter'] = len(filtered_df)
            print(f"Completed genetic architecture score removal filtering with mode {config['genetic_architecture_score_mode']}, resulting in {filter_counts['count_genetic_architecture_score_remove_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}', index=False)
            save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}')
    
        ### Save final filter counts and sequences with filter metrics ###
        filter_counts.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}', index=False)
        filtered_df.to_csv(f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}', index=False)
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}')
        print(f"Completed all diversification filtering. Final count: {len(filtered_df)} sequences.")


    #####################################################
    ### 6. GENOME VISUALIZATION AND SYNTENY FILTERING ###
    #####################################################

    ### Annotate & visualize genomes ###
    if config["genetic_architecture_visualization_and_synteny_filtering"] == True:

        ### Check that ORFs and protein annotations are predicted ###
        if config["homology_filtering"] == False or config["protein_database_hit_count_filter"] == False:
            raise ValueError("ORFs and their proteins must be predicted and annotated against a given protein database (homology filtering step) to run genome annotation visualization.")

        ### Save GFF files ###
        print("Creating gff files...")
        mmseqs_results_df = annotate_protein_hits(f'{config["results_save_dir"]}/{config["mmseqs_protein_database_results_dir_save_location"]}/mmseqs2_hits.csv',
                                                    config["protein_annotation_file"])   
        batch_create_gff_files(genomes_fasta_file=seq_fasta,
                               circular_genomes_fasta_file=f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}',
                               genomes_csv_file=filtered_df,
                               circular_orfs_fasta_file=f'{config["results_save_dir"]}/{config["orfipy_orfs_file_save_location"]}',
                               protein_database_hits_df=mmseqs_results_df,
                               output_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                               genome_id_map_output_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}')

        ### Save GBK files ###
        print("Creating gbk files...")
        batch_convert_gff_to_gbk(input_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                 output_dir=f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}')

        ### Re-load sequences ###
        if config["diversification_filtering"] == True:
            filtered_df = pd.read_csv(f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}')
            #seq_fasta = f'{config["results_save_dir"]}/{config["diversification_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}')
            print(f"Loading {len(filtered_df)} diversification-filtered sequences for genome visualization and synteny filtering...")
        elif config["diversification_filtering"] == False:
            filtered_df = pd.read_csv(f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}')
            #seq_fasta = f'{config["results_save_dir"]}/{config["homology_filter_seqs_fasta_file_save_location"]}'
            filter_counts = pd.read_csv(f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}')
            print(f"Loading {len(filtered_df)} homology-filtered sequences for genome visualization and synteny filtering...")

        ### Add genome ID mappings ###
        filtered_df.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}', index=False)
        add_genome_id_mapping(genome_id_map_csv=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}/genome_id_map.csv',
                        input_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                        output_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')

        ### Filter by average protein sequence identity ###
        if config["average_protein_sequence_identity_filter"] == True:
            print(f"Filtering by average amino acid sequence identity with threshold {config['average_protein_sequence_identity_range']}...")
            valid_average_protein_percent_identity(f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                                   f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}',
                                                   f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                                   f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                                   config["average_protein_sequence_identity_range"])
            filtered_df = pd.read_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')
            filter_counts['count_average_protein_sequence_identity_filter'] = len(filtered_df)
            print(f"Completed average amino acid sequence identity resulting in {filter_counts['count_average_protein_sequence_identity_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}', index=False)
            save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["synteny_filter_seqs_fasta_file_save_location"]}')

        ### Filter by required genes ###
        if config["required_genes_filter"] == True:
            print(f"Filtering by required genes filter with required genes {config['required_genes_list']}...")
            filtered_df = valid_gene_annotations(input_gff_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                   input_gbk_dir=f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}',
                                   required_products=config["required_genes_list"],
                                   sequences_df=filtered_df)
            filter_counts['count_required_genes_filter'] = len(filtered_df)
            print(f"Completed required genes filtering resulting in {filter_counts['count_required_genes_filter'].values[0]} sequences.")
            # Save checkpoint
            filter_counts.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_counts_file_save_location"]}', index=False)
            filtered_df.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}', index=False)
            save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["synteny_filter_seqs_fasta_file_save_location"]}')

        ### Visualize genome annotations ###
        print("Creating genetic architecture visualizations...")
        run_lovis4u_in_conda_env(config["lovis4u_conda_env"],
                                 f"python {config['genetic_architecture_visualization_script']} {config['current_config_file']}")
        move_genetic_architecture_pdfs(f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}',
                                       f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_pdf_dir_save_location"]}')
        print("Finished creating genetic architecture visualizations.")

        ### Filter by synteny to a reference genome ###
        if config["syntenic_gene_count_filter"] == True:
            print(f"Filtering by syntenic gene count with syntenic gene count range {config['syntenic_gene_count_range']}, total gene count range {config['total_gene_count_range']}, and syntenic total gene counts removed {config['syntenic_total_gene_count_remove']}...")
            count_syntenic_genes_all(root_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}',
                                     gff_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                     input_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                     output_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')
            count_total_num_genes(f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
                                  f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')
            valid_syntenic_gene_count(input_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                      output_csv=f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}',
                                      syntenic_gene_count_range= config["syntenic_gene_count_range"],
                                      total_gene_count_range= config["total_gene_count_range"],
                                      syntenic_total_gene_count_remove= config["syntenic_total_gene_count_remove"],
                                      gff_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}', 
                                      gbk_dir=f'{config["results_save_dir"]}/{config["gbk_dir_save_location"]}',
                                      pdf_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_pdf_dir_save_location"]}',
                                      metadata_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}')
            filtered_df = pd.read_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')
            filter_counts['count_syntenic_gene_count_filter'] = len(filtered_df)
            print(f"Completed syntenic gene count filtering resulting in {filter_counts['count_syntenic_gene_count_filter'].values[0]} sequences.")

        ### Save final filter counts and sequences with filter metrics ###
        filter_counts.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_counts_file_save_location"]}', index=False)
        filtered_df = pd.read_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}')
        save_df_as_fasta(filtered_df, f'{config["results_save_dir"]}/{config["synteny_filter_seqs_fasta_file_save_location"]}')
        print(f"Completed all synteny filtering. Final count: {len(filtered_df)} sequences.")


    print(f"Completed genome design filtering pipeline. Final number of candidates: {len(filtered_df)}.")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] 
    main(config_file)