import sys
import json
import yaml
import itertools
import csv
import shutil
import uuid
import os
import tempfile
import subprocess
import multiprocessing
import concurrent.futures
import time
from pathlib import Path


def run_lovis4u(input_gff_dir: str, output_dir: str) -> None:
    """
    Run lovis4u to visually annotate/compare genomes.
    
    Args:
    input_gff_dir: Directory containing gff files of genomes to compare by lovis4u.
    output_dir: Directory to save lovis4u results.
    """
    command = [
        'lovis4u', 
        '-gff', input_gff_dir,
        '-hl',
        '--set-category-colour',
        '-c', 'A4p2',
        '-o', output_dir,
        '-alip' # Short for --add-locus-id-prefix; very important for for ensuring duplicate ORF names don't cause bugs
    ]
    #Command line equivalent: lovis4u -gff gff_files -hl --set-category-colour -c A4p2 -o lovis4u_output

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print(f"Error running lovis4u: {result.stderr}")
    else:
        print(f"Command output: {result.stdout}")
    
    return result.returncode, output_dir


def process_single_genome(gff_file, query_gff_dir, visualize_against_reference_genome, reference_genome_gff, output_results_dir):
    """
    Process a single genome file with lovis4u.
    
    Args:
        gff_file: Name of the GFF file to process
        query_gff_dir: Directory containing the GFF files
        visualize_against_reference_genome: Whether to include reference genome
        reference_genome_gff: Path to reference genome GFF
        output_results_dir: Directory to save results
    
    Returns:
        Tuple of (genome_name, return_code, processing_time) where return_code is 0 for success
    """
    start_time = time.time()
    
    try:
        gff_path = os.path.join(query_gff_dir, gff_file)
        genome_name = gff_file.replace(".gff", "")  # Extract genome name without .gff
        
        # Create a process-specific temporary directory
        pid = os.getpid()
        temp_dir = Path(query_gff_dir) / f"temp_{genome_name}_{pid}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy the current .gff file and the reference file into the temporary directory
        shutil.copy(gff_path, temp_dir)
        if visualize_against_reference_genome:
            shutil.copy(reference_genome_gff, temp_dir)
        
        # Run lovis4u
        output_lovis4u_dir = os.path.join(output_results_dir, f"{genome_name}")
        return_code, _ = run_lovis4u(str(temp_dir), output_lovis4u_dir)
        
        # Cleanup: remove the temporary directory after processing
        shutil.rmtree(temp_dir)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Completed {genome_name} in {processing_time:.2f} seconds with return code {return_code}")
        return genome_name, return_code, processing_time
    
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Error processing {gff_file}: {str(e)}")
        return gff_file, 1, processing_time  # Return error code 1 for exception


def run_lovis4u_pairwise(query_gff_dir: str, visualize_against_reference_genome: bool, reference_genome_gff: str, output_results_dir: str, max_workers=None, chunk_size=10) -> None:
    """
    Run lovis4u by pairwise comparison between query genomes and a reference genome.
    Uses ThreadPoolExecutor for parallelization which may work better for I/O bound operations.

    Args:
    - query_gff_dir (str): Path to GFF files of genomes to analyze
    - visualize_against_reference_genome (bool): Whether to visualize against a reference genome
    - reference_genome_gff (str): Path to reference genome GFF file
    - output_results_dir (str): Path to output directory
    - max_workers (int): Maximum number of worker threads. None means ThreadPoolExecutor decides
    - chunk_size (int): Process files in chunks of this size
    """
    # Ensure directories exist
    os.makedirs(output_results_dir, exist_ok=True)
    
    # Ensure reference genome exists if needed
    if visualize_against_reference_genome and not os.path.isfile(reference_genome_gff):
        raise FileNotFoundError(f"Reference GFF file not found: {reference_genome_gff}")

    # Find and sort all GFF files
    gff_files = sorted(
        [f for f in os.listdir(query_gff_dir) if f.startswith("genome_") and f.endswith(".gff")],
        key=lambda x: int(x.split('_')[1].split('.')[0])  # Extracts the number after "genome_"
    )
    
    total_files = len(gff_files)
    print(f"Found {total_files} genome files to process")
    
    # Use ThreadPoolExecutor which may be better for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process in batches to allow for better progress reporting
        successful, failed = 0, 0
        total_time = 0
        processed = 0
        
        # Process files in chunks to reduce overhead and improve reporting
        for i in range(0, len(gff_files), chunk_size):
            chunk = gff_files[i:i + chunk_size]
            futures = []
            
            # Submit jobs for this chunk
            for gff_file in chunk:
                future = executor.submit(
                    process_single_genome,
                    gff_file, 
                    query_gff_dir,
                    visualize_against_reference_genome,
                    reference_genome_gff,
                    output_results_dir
                )
                futures.append(future)
            
            # Wait for all futures in this chunk to complete
            for future in concurrent.futures.as_completed(futures):
                _, return_code, processing_time = future.result()
                total_time += processing_time
                
                if return_code == 0:
                    successful += 1
                else:
                    failed += 1
                
                processed += 1
            
            # Report progress after each chunk
            print(f"Progress: {processed}/{total_files} files processed. "
                  f"Success: {successful}, Failed: {failed}. "
                  f"Average time: {total_time/processed:.2f}s per file")
    
    # Final stats
    print(f"Processing complete. Successful: {successful}, Failed: {failed}")
    if processed > 0:
        print(f"Average processing time: {total_time/processed:.2f} seconds per file")
        print(f"Total processing time: {total_time:.2f} seconds")


def main(config_file):
    """Run lovis4u in an isolated Conda environment."""

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

        # Get parallelization settings from config if available
        max_workers = config.get("n_parallel_jobs", None)
        chunk_size = config.get("chunk_size", 10)

        ### Visualize genome annotations ###
        run_lovis4u_pairwise(
            query_gff_dir=f'{config["results_save_dir"]}/{config["gff_dir_save_location"]}',
            visualize_against_reference_genome=config["use_reference_genome"],
            reference_genome_gff=config["reference_genome_gff_file_save_location"],
            output_results_dir=f'{config["results_save_dir"]}/{config["genetic_architecture_visualization_dir_save_location"]}',
            max_workers=max_workers,
            chunk_size=chunk_size
        )

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] 
    main(config_file)