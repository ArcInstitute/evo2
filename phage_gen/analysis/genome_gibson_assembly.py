"""
Usage:

python /path/to/genome_gibson_assembly.py

Requirements:
- Input CSV with a column 'sequence' containing genome sequences to be split into Gibson assembly fragments
- Path to save an updated CSV with Gibson assembly data
"""

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from typing import List, Dict, Any
import pandas as pd


class GibsonDesignError(Exception):
    """Custom exception for Gibson assembly design errors."""
    pass


def check_overlap_quality(sequence: str) -> Dict[str, Any]:
    """
    Evaluates overlap region quality for Gibson assembly, including GC clamps.
    
    Args:
        sequence (str): Potential overlap sequence.
        
    Returns:
        dict: Quality metrics including Tm, GC content, GC clamp evaluation, and issues.
    """
    seq = Seq(sequence)
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    
    # Evaluate GC clamp (G or C at the first and last position)
    has_gc_clamp = sequence[0] in "GC" and sequence[-1] in "GC"

    # Check for homopolymer runs in overlap region
    problems = []
    homopolymer_penalty = 0
    for base in 'ATCG':
        if base * 6 in sequence:
            problems.append(f"Contains {base}6+ homopolymer")
            homopolymer_penalty += 10  # High penalty for very long homopolymers
        if base * 4 in sequence:
            problems.append(f"Contains {base}4+ homopolymer")
            homopolymer_penalty += 5  # Smaller penalty for stretches of 4

    return {
        'tm': mt.Tm_NN(seq),
        'gc_content': gc_content,
        'problems': problems,
        'homopolymer_penalty': homopolymer_penalty,
        'has_gc_clamp': has_gc_clamp,
    }


def find_top_overlaps(
    sequence: str,
    overlap_length: int = 40,
    tm_target: float = 65,
    tm_range: tuple = (60, 75),
    gc_range: tuple = (0.4, 0.6),
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """
    Finds the top overlap junctions for a sequence based on quality metrics.
    Handles the circular nature of the sequence by evaluating overlaps that wrap around.
    
    Args:
        sequence (str): Input DNA sequence
        overlap_length (int): Desired overlap length
        tm_target (float): Target melting temperature
        tm_range (tuple): Acceptable Tm range
        gc_range (tuple): Acceptable GC content range
        top_n (int): Number of top overlap junctions to select.
    
    Returns:
        list: List of dictionaries with information about top overlap junctions.
    """
    seq_len = len(sequence)
    circular_sequence = sequence + sequence[:overlap_length]  # Simulate circularization
    candidates = []

    for i in range(seq_len):  # Evaluate overlaps at every position
        overlap = circular_sequence[i:i + overlap_length]
        metrics = check_overlap_quality(overlap)
        
        # Check Tm and GC content ranges
        if not (tm_range[0] <= metrics['tm'] <= tm_range[1]):
            continue
        if not (gc_range[0] <= metrics['gc_content'] <= gc_range[1]):
            continue
        if not metrics['has_gc_clamp']:
            continue  # Exclude overlaps without proper GC clamps
        if metrics['problems']:
            continue
        
        # Score the overlap based on closeness to target Tm and penalize homopolymers
        tm_diff = abs(metrics['tm'] - tm_target)
        gc_diff = abs(metrics['gc_content'] - 0.5)
        score = tm_diff + gc_diff * 50 + metrics['homopolymer_penalty']  # Include homopolymer penalty

        candidates.append({
            "position": i,
            "overlap": overlap,
            "tm": metrics['tm'],
            "gc_content": metrics['gc_content'],
            "has_gc_clamp": metrics['has_gc_clamp'],
            "score": score
        })

    # Sort by score and return top N candidates
    candidates = sorted(candidates, key=lambda x: x['score'])
    return candidates[:top_n]


def select_optimal_splits(
    sequence: str,
    top_candidates: List[Dict[str, Any]],
    target_distance: int = 2500
) -> List[Dict[str, Any]]:
    """
    Selects two optimal split points for a circular sequence, prioritizing balanced fragment sizes.
    
    Args:
        sequence (str): Input DNA sequence
        top_candidates (list): Top overlap candidates
        target_distance (int): Desired distance between splits
        
    Returns:
        list: Two selected splits
    """
    if len(top_candidates) < 2:
        raise GibsonDesignError("Not enough suitable overlap junctions found.")

    seq_len = len(sequence)
    best_split = top_candidates[0]
    remaining_candidates = top_candidates[1:]

    # Define a helper function to calculate fragment size imbalance
    def fragment_size_penalty(split1, split2):
        frag1_len = (split2 - split1) % seq_len
        frag2_len = seq_len - frag1_len
        return abs(frag1_len - target_distance) + abs(frag2_len - target_distance)

    # Find the second split point that minimizes fragment size imbalance
    second_split = min(
        remaining_candidates,
        key=lambda x: fragment_size_penalty(best_split["position"], x["position"])
    )

    return [best_split, second_split]


def design_circular_gibson_fragments(
    sequence: str,
    overlap_length: int = 30,
    tm_target: float = 65,
    target_distance: int = 2700,
    max_fragment_length: int = 5000,
    top_n_overlaps: int = 50
) -> Dict[str, Any]:
    """
    Designs Gibson assembly fragments for a circular sequence.
    Creates two fragments with overlaps for circular assembly.
    
    Args:
        sequence (str): Input DNA sequence
        overlap_length (int): Desired overlap length
        tm_target (float): Target melting temperature
        target_distance (int): Desired distance between splits
        max_fragment_length (int): Maximum allowed fragment length
        top_n_overlaps (int): Number of top overlap candidates to consider
        
    Returns:
        dict: Assembly design including fragments and overlaps
    """
    sequence = sequence.upper()
    seq_len = len(sequence)

    # Find the top overlap junctions
    top_overlaps = find_top_overlaps(sequence, overlap_length=overlap_length, tm_target=tm_target, top_n=top_n_overlaps)
    if not top_overlaps:
        raise GibsonDesignError("No suitable overlap junctions found.")

    # Select the two optimal split points
    optimal_splits = select_optimal_splits(sequence, top_overlaps, target_distance=target_distance)

    # Define the split points
    split1, split2 = sorted([optimal_splits[0]["position"], optimal_splits[1]["position"]])

    # Define fragments with overlaps
    fragment_1 = sequence[split1:split2] + sequence[split2:split2 + overlap_length]  # Includes overlap for Fragment 2
    fragment_2 = sequence[split2:] + sequence[:split1] + sequence[split1:split1 + overlap_length]  # Includes overlap for Fragment 1

    # Calculate fragment lengths
    fragment_1_length = len(fragment_1)
    fragment_2_length = len(fragment_2)

    # Ensure fragments do not exceed the maximum length
    if fragment_1_length > max_fragment_length or fragment_2_length > max_fragment_length:
        raise GibsonDesignError(f"Fragments exceed maximum allowed length of {max_fragment_length} bp.")

    # Ensure total length matches the sequence length (accounting for overlaps)
    assert fragment_1_length + fragment_2_length - (overlap_length * 2) == seq_len, (
        f"Fragment lengths ({fragment_1_length} + {fragment_2_length}) do not match sequence length ({seq_len})!"
    )

    # Calculate Tm difference
    overlap_tm_difference = abs(optimal_splits[0]["tm"] - optimal_splits[1]["tm"])

    return {
        "fragment_1": fragment_1,
        "fragment_2": fragment_2,
        "split_1_overlap": optimal_splits[0],
        "split_2_overlap": optimal_splits[1],
        "overlap_tm_difference": overlap_tm_difference,
        "fragment_1_length": fragment_1_length,
        "fragment_2_length": fragment_2_length
    }


def design_gibson_for_dataframe(
    input_csv: str,
    output_csv: str,
    overlap_length: int = 30,
    tm_target: float = 65,
    target_distance: int = 2700,
    max_fragment_length: int = 5000,
    top_n_overlaps: int = 50
) -> None:
    """
    Reads a CSV with DNA sequences and computes Gibson assembly design for each.
    Adds columns with fragment sequences, overlaps, and quality metrics.
    
    Args:
        input_csv (str): Path to input CSV with a column 'sequence'.
        output_csv (str): Path to save the updated CSV with Gibson assembly data.
        overlap_length (int): Desired overlap length.
        tm_target (float): Target melting temperature for overlaps.
        target_distance (int): Desired distance between split points.
        max_fragment_length (int): Maximum allowed fragment length.
        top_n_overlaps (int): Number of top overlap candidates to consider.
    """
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Ensure required column exists
    if 'sequence' not in df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column.")

    # Prepare new columns
    new_columns = [
        "gibson_fragment1_sequence", "gibson_fragment2_sequence",
        "gibson_fragment1_len", "gibson_fragment2_len",
        "gibson_overlap1_sequence", "gibson_overlap2_sequence",
        "gibson_overlap1_tm", "gibson_overlap2_tm",
        "gibson_overlap_tm_difference",
        "gibson_overlap1_gc", "gibson_overlap2_gc",
    ]
    for col in new_columns:
        df[col] = None

    # Process each sequence
    for index, row in df.iterrows():
        sequence = row['sequence'].upper()
        try:
            # Design Gibson assembly
            result = design_circular_gibson_fragments(
                sequence,
                overlap_length=overlap_length,
                tm_target=tm_target,
                target_distance=target_distance,
                max_fragment_length=max_fragment_length,
                top_n_overlaps=top_n_overlaps
            )

            # Populate DataFrame with results
            df.at[index, "gibson_fragment1_sequence"] = result["fragment_1"]
            df.at[index, "gibson_fragment2_sequence"] = result["fragment_2"]
            df.at[index, "gibson_fragment1_len"] = result["fragment_1_length"]
            df.at[index, "gibson_fragment2_len"] = result["fragment_2_length"]
            df.at[index, "gibson_overlap1_sequence"] = result["split_1_overlap"]["overlap"]
            df.at[index, "gibson_overlap2_sequence"] = result["split_2_overlap"]["overlap"]
            df.at[index, "gibson_overlap1_tm"] = result["split_1_overlap"]["tm"]
            df.at[index, "gibson_overlap2_tm"] = result["split_2_overlap"]["tm"]
            df.at[index, "gibson_overlap_tm_difference"] = result["overlap_tm_difference"]
            df.at[index, "gibson_overlap1_gc"] = result["split_1_overlap"]["gc_content"]
            df.at[index, "gibson_overlap2_gc"] = result["split_2_overlap"]["gc_content"]

        except GibsonDesignError as e:
            print(f"Error processing sequence at index {index}: {str(e)}")

    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Gibson assembly data saved to {output_csv}")


if __name__ == "__main__":
    input_csv = '/large_storage/hielab/samuelking/phage_design/generation/data/20250326_phagebatch3_finalgenome_checks/check5/qc6_final_filter_seqs.csv'
    output_csv = '/large_storage/hielab/samuelking/phage_design/generation/data/20250326_phagebatch3_finalgenome_checks/check5/qc6_final_filter_seqs_with_gibson.csv'
    design_gibson_for_dataframe(input_csv, output_csv)
