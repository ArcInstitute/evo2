"""
CPU-only tests for evo2.scoring that use a tiny causal stand-in model,
so they run without a GPU, vortex kernels, or model weights.

Usage: python -m pytest tests/test_scoring.py
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

# `import evo2` pulls in vortex, which needs CUDA kernels. scoring.py only uses
# StripedHyena as a type hint, so stub vortex and load the module by path.
_vortex_stub = types.ModuleType('vortex.model.model')
_vortex_stub.StripedHyena = object
_spec = importlib.util.spec_from_file_location(
    'evo2_scoring', Path(__file__).resolve().parents[1] / 'evo2' / 'scoring.py'
)
scoring = importlib.util.module_from_spec(_spec)
with mock.patch.dict(sys.modules, {
    'vortex': types.ModuleType('vortex'),
    'vortex.model': types.ModuleType('vortex.model'),
    'vortex.model.model': _vortex_stub,
}):
    _spec.loader.exec_module(scoring)

score_sequences = scoring.score_sequences


class CharTokenizer:
    eod_id = 0
    pad_id = 1

    def tokenize(self, seq):
        return [ord(c) for c in seq]


class CausalBigramModel(torch.nn.Module):
    """Next-token logits depend only on the current token, so the model is causal."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.nn.Embedding(256, 256)

    def forward(self, input_ids):
        return self.table(input_ids), None


def reference_score(seq, model, tokenizer, prepend_bos, reduce_method):
    ids = ([tokenizer.eod_id] if prepend_bos else []) + tokenizer.tokenize(seq)
    ids = torch.tensor(ids).unsqueeze(0)
    logprobs = torch.log_softmax(model(ids)[0], dim=-1)[0, :-1]
    logprobs = logprobs.gather(1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).detach().numpy()
    return getattr(np, reduce_method)(logprobs)


@pytest.mark.parametrize('prepend_bos', [False, True])
@pytest.mark.parametrize('reduce_method', ['sum', 'mean'])
def test_padding_does_not_change_scores(prepend_bos, reduce_method):
    model, tokenizer = CausalBigramModel(), CharTokenizer()
    seqs = ['ACGTAC', 'ACGTACGTACGT', 'GATTACA']

    batched = score_sequences(
        seqs, model, tokenizer, prepend_bos=prepend_bos,
        reduce_method=reduce_method, device='cpu',
    )
    unbatched = score_sequences(
        seqs, model, tokenizer, batch_size=1, prepend_bos=prepend_bos,
        reduce_method=reduce_method, device='cpu',
    )
    expected = [
        reference_score(seq, model, tokenizer, prepend_bos, reduce_method)
        for seq in seqs
    ]

    np.testing.assert_allclose(unbatched, expected, rtol=1e-5)
    np.testing.assert_allclose(batched, expected, rtol=1e-5)
