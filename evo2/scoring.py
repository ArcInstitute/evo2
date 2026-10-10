import numpy as np
from typing import List, Tuple, Union
from Bio.Seq import Seq
from tqdm import tqdm

import torch
from vortex.model.model import StripedHyena


def prepare_batch(
        seqs: List[str],
        tokenizer: object,
        prepend_bos: bool = False,
        device: str = 'cuda:0'
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes in a list of sequences, tokenizes them, and puts them in a tensor batch.
    If the sequences have differing lengths, then pad up to the maximum sequence length.
    """
    seq_lengths = [ len(seq) for seq in seqs ]
    max_seq_length = max(seq_lengths)

    input_ids = []
    for seq in seqs:
        padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
        input_ids.append(
            torch.tensor(
                ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                dtype=torch.long,
            ).to(device).unsqueeze(0)
        )
    input_ids = torch.cat(input_ids, dim=0)

    return input_ids, seq_lengths


def logits_to_logprobs(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Takes in a tensor of logits of dimension (batch, length, vocab).
    Computes the log-likelihoods using a softmax along the vocab dimension.
    Uses the `input_ids` to index into the log-likelihoods and returns the likelihood
    of the provided sequence at each position with dimension (batch, length).
    """
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    assert softmax_logprobs.shape[1] == input_ids.shape[1]

    logprobs = torch.gather(
        softmax_logprobs,       # Gather likelihoods...
        2,                      # along the vocab dimension...
        input_ids.unsqueeze(-1) # using the token ids to index.
    ).squeeze(-1)

    return logprobs


def _score_sequences(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: object,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        device: str = 'cuda:0',
) -> List[float]:
    """Helper function to score a list of sequences based on their logprobs."""
    input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device, prepend_bos=prepend_bos)
    assert len(seq_lengths) == input_ids.shape[0]

    with torch.inference_mode():
        logits, _ = model(input_ids) # (batch, length, vocab)

    logprobs = logits_to_logprobs(logits, input_ids)
    logprobs = logprobs.float().cpu().numpy()

    if reduce_method == 'sum': # PLL
        reduce_func = np.sum
    elif reduce_method == 'mean': # mean PLL
        reduce_func = np.mean
    else:
        raise ValueError(f'Invalid reduce_method {reduce_method}')

    # Without a BOS token the first nucleotide is not predicted, so a sequence of
    # length L has L - 1 logprobs. Slicing further would include padding tokens.
    n_scored = [ max(0, seq_length - 1 + int(prepend_bos)) for seq_length in seq_lengths ]

    return [
        reduce_func(logprobs[idx][:n_scored[idx]])
        for idx in range(len(seq_lengths))
    ]


def score_sequences(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: object,
        batch_size: int = None,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        device: str = 'cuda:0',
) -> List[float]:
    """
    Computes the model log-likelihood scores for sequences in `seqs`.
    Uses `reduce_method` to take the mean or sum across the likelihoods at each 
    position (default: `'mean'`).

    Returns a list of scalar scores corresponding to the reduced log-likelihoods for
    each sequence.
    """
    if batch_size is None:
        batch_size = len(seqs)

    scores = []
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch_seqs = seqs[i:i + batch_size]
        batch_scores = _score_sequences(
            batch_seqs,
            model,
            tokenizer,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
            device=device,
        )
        scores.extend(batch_scores)
    return scores


def score_sequences_rc(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: object,
        batch_size: int,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        device: str = 'cuda:0',
) -> List[float]:
    """
    Computes the model log-likelihood scores for sequences in `seqs` and for their
    reverse complements.
    Takes the mean score for the forward and reverse-complemented sequence.
    Uses `reduce_method` to take the mean or sum across the likelihoods at each 
    position (default: `'mean'`).

    Returns a list of scalar scores corresponding to the reduced log-likelihoods for
    each sequence.
    """
    scores = []
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch_seqs = seqs[i:i + batch_size]
        batch_seqs_rc = [ str(Seq(seq).reverse_complement()) for seq in batch_seqs ]
        
        batch_scores = _score_sequences(
            batch_seqs,
            model,
            tokenizer,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
            device=device,
        )
        batch_scores_rc = _score_sequences(
            batch_seqs_rc,
            model,
            tokenizer,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
            device=device,
        )
        batch_scores = (np.array(batch_scores) + np.array(batch_scores_rc)) * 0.5

        scores.extend(list(batch_scores))
    return scores


def positional_entropies(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: object,
        prepend_bos: bool = False,
        device: str = 'cuda:0',
) -> List[np.array]:
    """
    Computes the positional entropies for sequences in `seqs`.

    Returns a list of arrays, where each array is the same length as the
    corresponding sequence length. Each array contains the per-position entropy
    across the vocab dimension.
    """
    input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device, prepend_bos=prepend_bos)
    assert len(seq_lengths) == input_ids.shape[0]

    with torch.inference_mode():
        logits, _ = model(input_ids) # (batch, length, vocab)
    
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    if prepend_bos:
        softmax_logprobs = softmax_logprobs[:, 1:, :] # Remove BOS entropy.

    entropies = -torch.sum(torch.exp(softmax_logprobs) * softmax_logprobs, dim=-1)
    entropies = entropies.float().cpu().numpy()

    sequence_entropies = [
        entropies[idx][:seq_lengths[idx]] for idx in range(len(seq_lengths))
    ]
    assert all(
        len(seq) == len(entropy) for seq, entropy in zip(seqs, sequence_entropies)
    )

    return sequence_entropies


def score_perplexity_along_sequence(
        model: StripedHyena,
        seq: str,
        reverse_complement: bool = True,
        entropy: bool = False
    ) -> np.array:
    '''
    Get forward and reverse RC of dna sequence, pass both through model, and return average entropy or perplexity.
    '''
    seq_rc = str(Seq(seq).reverse_complement())
    
    entropy_forward = positional_entropies([seq], model.model, model.tokenizer)[0]

    if reverse_complement:
        entropy_reverse = positional_entropies([seq_rc], model.model, model.tokenizer)[0]
        entropy_reverse = entropy_reverse[::-1]
        
        average_entropy = (entropy_forward + entropy_reverse) / 2
    else:
        average_entropy = entropy_forward

    if entropy:
        return average_entropy
    else:
        return np.exp(average_entropy)