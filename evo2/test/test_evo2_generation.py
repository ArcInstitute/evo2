import argparse
import csv
from importlib import resources
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

import torch

from evo2 import Evo2

def read_prompts(input_file):
    """Read prompts from input file or built-in test data.
    
    Args:
        input_file: Either a path to a file, or the name of a test data file
                   (e.g., 'prompts.csv')
    """
    # If it's a string that doesn't exist as a file path, assume it's a test data file
    if isinstance(input_file, str) and not Path(input_file).is_file():
        # This is the reliable way to get package data
        with resources.path('evo2.test.data', input_file) as data_path:
            input_file = data_path
    
    # Your existing code to read the file
    promptseqs = []
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            promptseqs.append(row[0])
    return promptseqs

def mid_point_split(*, seq, num_tokens):
    """Split sequence at midpoint for prompt and target."""
    mid_point = 2*(len(seq)//4)
    prompt = seq[:mid_point]
    target = seq[mid_point:mid_point+num_tokens]
    return prompt, target

def calculate_sequence_identity(seq1: str, seq2: str) -> Optional[float]:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None
    
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))
    return (matches / min_length) * 100

def generate_and_score(*, sequences, model, generations_per_prompt=5, n_tokens=500,
                      temperature=1.0, top_k=1, top_p=1.0):
    """Prompt with first half, generate and score on 2nd half."""
    scores = []
    prompts = []
    targets = []
    
    # Prepare all prompts and targets
    for seq in sequences:
        prompt, target = mid_point_split(seq=seq, num_tokens=n_tokens)
        prompts.extend([prompt] * generations_per_prompt)
        targets.extend([target] * generations_per_prompt)
    
    for i in range(len(prompts)):
        prompt = prompts[i]
        target = targets[i]

        with torch.inference_mode():
            generated = model.generate(
                prompt_seqs=[prompt],
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            decoded_seq = generated.sequences[0]  # Assuming generate returns list of sequences
            score = calculate_sequence_identity(decoded_seq, target)
            scores.append(score)
    
    # Reshape scores to group by original sequence
    reshaped_scores = [scores[i:i + generations_per_prompt] 
                      for i in range(0, len(scores), generations_per_prompt)]
    
    return reshaped_scores

def main():
    """
    Test sequence generation and scoring using the evo2 models
    Expected results (direct comparison w/o alignment):
    - Evo 2 40B 1m: 91.15%
    - Evo 2 7B 1m: 89.25% 
    - Evo 2 1B base: 68.0%
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Generation")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_1b_base'], default='evo2_7b',
                       help="Model to test (supports evo2_7b, evo2_40b, evo2_1b_base)")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
        
    model = Evo2(args.model_name)
    
    # Test parameters: greedy sampling of 500 tokens
    test_params = {
        'n_tokens': 500,
        'temperature': 1.0,
        'top_k': 1,
        'top_p': 1.0,
        'generations_per_prompt': 1,
    }
    
    # Read and process sequences
    sequences = read_prompts('prompts.csv')
    scores = generate_and_score(
        sequences=sequences,
        model=model,
        **test_params
    )
    
    # Calculate and validate results
    mean_score = np.mean(scores)
    print("\nTest Results:")
    print("% Matching Nucleotides:", mean_score)
    
    # Validate against expected scores
    eps = 3  # large epsilon for direct comparison, since there are numeric differences by versions
    expected_scores = {
        'evo2_40b': 91.15,
        'evo2_7b': 89.25,
        'evo2_1b_base': 68.0
    }
    
    expected_score = expected_scores[args.model_name]
    if abs(mean_score - expected_score) < eps:
        print(f"\nTest Passed! Score matches expected {expected_score}%")
    else:
        print(f"\nTest Failed: Expected {expected_score}%, got {mean_score}%")

if __name__ == "__main__":
    main()