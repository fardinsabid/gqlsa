"""
Causality test for GQLSA.
Verifies that future tokens do not influence earlier positions.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention.gqlsa import GQLSA


@pytest.fixture
def attention():
    """Create a GQLSA attention module for testing."""
    model = GQLSA(
        d_model=512,
        h=8,
        g=2,
        d_k=64,
        d_v=64,
        d_c=128,
        local_window=64,
        top_k=32,
        block_size=32,
        # retrieval constants
        retrieval_M=4,
        retrieval_bucket_width=0.5,
        retrieval_c_max=16,
        bucket_count=8,
    )
    model.eval()
    return model


def test_causality(attention):
    """Test that perturbing position 5 does not affect positions 0-4."""

    x = torch.randn(2, 64, 512)
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] += 100.0

    with torch.no_grad():
        out1 = attention(x)
        out2 = attention(x_perturbed)

    max_diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()

    assert max_diff < 1e-5, f"Future token leakage detected! Max diff = {max_diff}"


def test_causality_different_lengths(attention):
    """Test causality at multiple sequence lengths."""

    for seq_len in [32, 64, 128, 256]:
        x = torch.randn(2, seq_len, 512)
        x_perturbed = x.clone()
        x_perturbed[:, 5, :] += 100.0

        with torch.no_grad():
            out1 = attention(x)
            out2 = attention(x_perturbed)

        max_diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()

        assert max_diff < 1e-5, f"Leakage at seq_len={seq_len}, diff={max_diff}"


def test_causality_via_retrieval_path(attention):
    """
    specific: perturb a token at a distant position and verify that
    earlier positions are unaffected. This exercises the content-aware
    retrieval code path — the perturbation changes the anchor summaries
    downstream of position 200, but positions 0-199 must not change.
    """

    seq_len = 256
    x = torch.randn(2, seq_len, 512)
    x_perturbed = x.clone()
    x_perturbed[:, 200, :] += 100.0

    with torch.no_grad():
        out1 = attention(x)
        out2 = attention(x_perturbed)

    max_diff = (out1[:, :200, :] - out2[:, :200, :]).abs().max().item()

    assert max_diff < 1e-5, (
        f"Distant perturbation leaked backwards through the retrieval path! "
        f"Max diff = {max_diff} at positions 0-199"
    )


def test_causality_of_block_indices(attention):
    """
    specific: verify that the selected global block indices for a
    given query are always strictly before its local window. This
    catches causality bugs in the tensor-based inverted index that a
    simple perturbation test might miss.
    """

    seq_len = 256
    x = torch.randn(1, seq_len, 512)

    with torch.no_grad():
        # Populate the block summary caches
        C_q = attention.q_compress(x)
        C_kv = attention.kv_compress(x)
        N = seq_len // attention.block_size
        attention._C_q_mean = C_q.view(1, N, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = C_kv.view(1, N, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        idx, same = attention._get_block_indices(seq_len, x.device)

    L_b = attention.local_blocks
    for q in range(N):
        for slot in range(L_b, attention.k_eff):
            block = idx[q, slot].item()
            upper = q - L_b + 1
            if block >= upper:
                # Only valid if it is the padding fallback (block 0).
                assert block == 0, (
                    f"Query {q}, global slot {slot}: "
                    f"selected block {block} is not strictly past (upper={upper})"
                )
                # Fallback is causally safe only when the same-block flag
                # matches (q == 0).
                flag = same[q, slot].item()
                expected = (q == 0)
                assert flag == expected, (
                    f"Query {q}, padding block: same-block flag {flag}, "
                    f"expected {expected}"
                )