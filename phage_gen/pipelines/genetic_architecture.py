import numpy as np, pandas as pd, re, matplotlib.pyplot as plt, os, seaborn as sns, itertools, math, random
from collections import defaultdict
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed


def binary_start_stop_vector(sequence: str) -> np.array:
    """Generates a binary vector where positions of start codons (ATG) and stop codons (TAA, TAG, TGA) are marked with 1."""
    extended_sequence = sequence + sequence[:2] # Extend the sequence with the first two characters to the end
    sequence_array = np.array(list(extended_sequence))
    triplets = np.lib.stride_tricks.sliding_window_view(sequence_array, 3)
    triplets_joined = np.apply_along_axis(''.join, 1, triplets)
    start_stop_vector = np.isin(triplets_joined, ['ATG', 'TAA', 'TAG', 'TGA']).astype(int)
    return start_stop_vector


def circular_permute_vector(vector: np.array) -> np.array:
    """Circularly permutes a given vector using NumPy's roll, optimizing memory access."""
    n = len(vector)
    permuted_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        permuted_matrix[i] = np.roll(vector, i)
    return permuted_matrix.T


def binary_start_stop_matrix(sequence: str) -> np.array:
    """Generates a matrix of combined binary start and stop codon vectors for all circular permutations of a sequence."""
    start_stop_vector = binary_start_stop_vector(sequence)
    start_stop_matrix = circular_permute_vector(start_stop_vector)
    return start_stop_matrix


def pad_matrix_to_match_rows(matrix, target_rows) -> np.array:
    """Pads a matrix with zeros so that it has the desired number of rows."""
    current_rows = matrix.shape[0]
    if current_rows < target_rows:
        # Pad matrix with zeros at the bottom
        padding = ((0, target_rows - current_rows), (0, 0))
        matrix = np.pad(matrix, padding, mode='constant', constant_values=0)
    return matrix


def genetic_architecture_score(truth_matrix, sequence_to_score, weight_vector, normalization_vector) -> np.array:
    """Scores the genetic architecture of a sequence based on the truth matrix."""
    
    matrix_to_score = binary_start_stop_matrix(sequence_to_score) # Calculate start/stop matrix for sequence of interest
    
    # Get number of columns in truth_matrix and rows in matrix_to_score
    truth_cols = truth_matrix.shape[1]
    matrix_to_score_rows = matrix_to_score.shape[0]
    
    # Pad either truth_matrix or matrix_to_score based on which is smaller
    if truth_cols > matrix_to_score_rows:
        matrix_to_score = pad_matrix_to_match_rows(matrix_to_score, truth_cols)
    elif truth_cols < matrix_to_score_rows:
        truth_matrix = pad_matrix_to_match_rows(truth_matrix.T, matrix_to_score_rows).T
    
    dot_products = np.dot(truth_matrix, matrix_to_score) # Calculate dot products
    max_dot_products = np.max(dot_products, axis=1, keepdims=True) # Find max score in each row
    weighted_dot_products = np.multiply(weight_vector, max_dot_products) # Weight the dot products 
    normalized_dot_products = weighted_dot_products / normalization_vector # Normalize the weighted dot products

    return normalized_dot_products


def gaussian_row_blur(matrix, sigma) -> np.array:
    """
    Applies a 1D Gaussian blur to each row of the matrix, retaining the original non-zero values as the center/mean values in each row.
    
    Arguments:
    - matrix: The input matrix where each row is blurred independently.
    - sigma: Standard deviation to control the spread of the Gaussian distribution for each row.
    
    Returns:
    - blurred_matrix: A new matrix where each row has been blurred, retaining the original values at non-zero positions.
    """
    
    # Initialize the blurred matrix as a copy of the original
    blurred_matrix = np.copy(matrix)
    
    # Apply a 1D Gaussian blur to each row independently
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        blurred_row = gaussian_filter1d(row, sigma=sigma)
        
        # Retain the original non-zero values in the row
        non_zero_positions = np.where(row != 0)[0]
        blurred_row[non_zero_positions] = row[non_zero_positions]
        
        # Store the blurred row in the matrix
        blurred_matrix[i, :] = blurred_row
    
    return blurred_matrix


def calculate_genetic_architecture_score_parallel(sequences_df: pd.DataFrame,
                                                  truth_matrix: np.array,
                                                  weight_vector: np.array,
                                                  normalization_vector: np.array,
                                                  n_jobs=-1):
    """
    Process each sequence in parallel and calculate the genetic architecture score.
    
    Arguments:
    - sequences_df: pandas DataFrame containing sequence data.
    - truth_matrix: The truth matrix used for scoring.
    - weight_vector: The weight vector used for scoring.
    - normalization_vector: The vector used for normalizing the genetic architecture score.
    - n_jobs: Number of parallel jobs to run (default: -1, which uses all available cores).
    
    Returns:
    - A list of tuples with sequence ID and the corresponding genetic architecture score.
    """
    
    def process_single_sequence(sequence, sequence_id, truth_matrix, weight_vector, normalization_vector):
        """Helper function to calculate the genetic architecture score of a single sequence."""
        max_dot_products_normalized = genetic_architecture_score(truth_matrix, sequence, weight_vector, normalization_vector)
        return (sequence_id, max_dot_products_normalized)

    # Run in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_sequence)(row['sequence'], row['id_prompt'], truth_matrix, weight_vector, normalization_vector)
        for _, row in sequences_df.iterrows()
    )
    
    return results


def save_score(results_list: list) -> pd.DataFrame:
    # Create a DataFrame from the results list
    sequence_ids = []
    genome_scores = []
    aabkc_scores = []
    de_scores = []
    j_scores = []
    f_scores = []
    h_scores = []
    g_scores = []

    for item in results_list:
        sequence_ids.append(item[0])  # Sequence ID (description)

        # Extract values from the array and append the scalar (not the array itself)
        genome_scores.append(item[1][0, 0].item())  # Extract scalar value for Genome_score
        aabkc_scores.append(item[1][1, 0].item())  # Extract scalar value for AABKC_score
        de_scores.append(item[1][2, 0].item())  # Extract scalar value for DE_score
        j_scores.append(item[1][3, 0].item())  # Extract scalar value for J_score
        f_scores.append(item[1][4, 0].item())  # Extract scalar value for F_score
        h_scores.append(item[1][5, 0].item())  # Extract scalar value for H_score
        g_scores.append(item[1][6, 0].item())  # Extract scalar value for G_score

    df = pd.DataFrame({
        'id_prompt': sequence_ids,
        'genome_score': genome_scores,
        'aabkc_score': aabkc_scores,
        'de_score': de_scores,
        'j_score': j_scores,
        'f_score': f_scores,
        'h_score': h_scores,
        'g_score': g_scores
    })

    # Calculate product of all scores (final genetic architecture score)
    score_columns = ['genome_score', 'aabkc_score', 'de_score', 'j_score', 'f_score', 'h_score', 'g_score']
    df['genetic_architecture_score'] = df[score_columns].prod(axis=1)

    return df


### TRUTH, WEIGHT, AND NORMALIZATION VECTORS ###

vector_length = 5386 # Length of PhiX174 genome

# Define PhiX174 truth vector for whole genome architecture, with no cryptic start/stop codons
genome_indices_NC001422_1 = {
    "A_start": 3980,
    "A*_start": 4496,
    "B_start": 5074,
    "K_start": 50, 
    "C_start": 132, 
    "D_start": 389,
    "E_start": 567,
    "J_start": 847,
    "F_start": 1000,
    "G_start": 2394,
    "H_start": 2930,

    "A_stop": 133,
    "A*_stop": 133,
    "B_stop": 48,
    "K_stop": 218, 
    "C_stop": 390, 
    "D_stop": 845,
    "E_stop": 840,
    "J_stop": 961,
    "F_stop": 2281,
    "G_stop": 2919,
    "H_stop": 3914
}
phix174_true_genome_vector = np.zeros(vector_length)
for key, index in genome_indices_NC001422_1.items():
    phix174_true_genome_vector[index] = 1

# Define PhiX174 truth vectors for gene module architectures, with no cryptic start/stop codons
AABKC_indices_NC001422_1 = {
    "A_start": 3980,
    "A*_start": 4496,
    "B_start": 5074,
    "K_start": 50, 
    "C_start": 132,
    "A_stop": 133,
    "A*_stop": 133,
    "B_stop": 48,
    "K_stop": 218, 
    "C_stop": 390
}
phix174_true_AABKC_vector = np.zeros(vector_length)
for key, index in AABKC_indices_NC001422_1.items():
    phix174_true_AABKC_vector[index] = 1

DE_indices_NC001422_1 = {
    "D_start": 389,
    "E_start": 567,
    "D_stop": 845,
    "E_stop": 840
}
phix174_true_DE_vector = np.zeros(vector_length)
for key, index in DE_indices_NC001422_1.items():
    phix174_true_DE_vector[index] = 1

J_indices_NC001422_1 = {
    "J_start": 847,
    "J_stop": 961
}
phix174_true_J_vector = np.zeros(vector_length)
for key, index in J_indices_NC001422_1.items():
    phix174_true_J_vector[index] = 1

F_indices_NC001422_1 = {
    "F_start": 1000,
    "F_stop": 2281
}
phix174_true_F_vector = np.zeros(vector_length)
for key, index in F_indices_NC001422_1.items():
    phix174_true_F_vector[index] = 1

G_indices_NC001422_1 = {
    "G_start": 2394,
    "G_stop": 2919
}
phix174_true_G_vector = np.zeros(vector_length)
for key, index in G_indices_NC001422_1.items():
    phix174_true_G_vector[index] = 1

H_indices_NC001422_1 = {
    "H_start": 2930,
    "H_stop": 3914
}
phix174_true_H_vector = np.zeros(vector_length)
for key, index in H_indices_NC001422_1.items():
    phix174_true_H_vector[index] = 1

# Stack into a (7, 5386) truth matrix
phix174_truth_matrix = np.vstack([
    phix174_true_genome_vector,
    phix174_true_AABKC_vector,
    phix174_true_DE_vector,
    phix174_true_J_vector,
    phix174_true_F_vector,
    phix174_true_G_vector,
    phix174_true_H_vector
])

# Create a weight vector for scoring each gene module; let's use just the number of ORF boundaries as the assigned score
phix174_weight_vector = np.sum(phix174_truth_matrix, axis=1, keepdims=True)

# Create Gaussian blurred PhiX174 truth matrices
phix174_truth_matrix_blurred_sigma5 = gaussian_row_blur(phix174_truth_matrix, sigma=5)
phix174_truth_matrix_blurred_sigma10 = gaussian_row_blur(phix174_truth_matrix, sigma=10)
phix174_truth_matrix_blurred_sigma15 = gaussian_row_blur(phix174_truth_matrix, sigma=15)
phix174_truth_matrix_blurred_sigma20 = gaussian_row_blur(phix174_truth_matrix, sigma=20)


# Create start/stop matrix for PhiX174
fasta_file = "/large_storage/hielab/samuelking/phage_design/data/phix174_only/microviridae_genomes_NC_001422_1.fna"
record = SeqIO.read(fasta_file, "fasta")
sequence = str(record.seq)
phix174_startstop_matrix = binary_start_stop_matrix(sequence)

# Create normalization vector for PhiX174 (the genetic architecture score for PhiX174)
phix174_dot_products = np.dot(phix174_truth_matrix, phix174_startstop_matrix)
phix174_max_dot_products = np.max(phix174_dot_products, axis=1, keepdims=True)
phix174_normalization_vector = np.multiply(phix174_weight_vector, phix174_max_dot_products)

# Create normalization vector for PhiX174 after Gaussian blurring (sigma = 5)
phix174_dot_products = np.dot(phix174_truth_matrix_blurred_sigma5, phix174_startstop_matrix)
phix174_max_dot_products = np.max(phix174_dot_products, axis=1, keepdims=True)
phix174_normalization_vector_blurred_sigma5 = np.multiply(phix174_weight_vector, phix174_max_dot_products)

# Create normalization vector for PhiX174 after Gaussian blurring (sigma = 10)
phix174_dot_products = np.dot(phix174_truth_matrix_blurred_sigma10, phix174_startstop_matrix)
phix174_max_dot_products = np.max(phix174_dot_products, axis=1, keepdims=True)
phix174_normalization_vector_blurred_sigma10 = np.multiply(phix174_weight_vector, phix174_max_dot_products)

# Create normalization vector for PhiX174 after Gaussian blurring (sigma = 15)
phix174_dot_products = np.dot(phix174_truth_matrix_blurred_sigma15, phix174_startstop_matrix)
phix174_max_dot_products = np.max(phix174_dot_products, axis=1, keepdims=True)
phix174_normalization_vector_blurred_sigma15 = np.multiply(phix174_weight_vector, phix174_max_dot_products)

# Create normalization vector for PhiX174 after Gaussian blurring (sigma = 20)
phix174_dot_products = np.dot(phix174_truth_matrix_blurred_sigma20, phix174_startstop_matrix)
phix174_max_dot_products = np.max(phix174_dot_products, axis=1, keepdims=True)
phix174_normalization_vector_blurred_sigma20 = np.multiply(phix174_weight_vector, phix174_max_dot_products)

# Check that the genetic architecture score equals 1 after normalization
#print(genetic_architecture_score(phix174_truth_matrix, sequence, phix174_weight_vector, phix174_normalization_vector))
#print(genetic_architecture_score(phix174_truth_matrix_blurred_sigma5, sequence, phix174_weight_vector, phix174_normalization_vector_blurred_sigma5))
#print(genetic_architecture_score(phix174_truth_matrix_blurred_sigma10, sequence, phix174_weight_vector, phix174_normalization_vector_blurred_sigma10))
#print(genetic_architecture_score(phix174_truth_matrix_blurred_sigma15, sequence, phix174_weight_vector, phix174_normalization_vector_blurred_sigma15))
#print(genetic_architecture_score(phix174_truth_matrix_blurred_sigma20, sequence, phix174_weight_vector, phix174_normalization_vector_blurred_sigma20))