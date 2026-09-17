"""
Attention correctness tests for GQLSA.
Verifies output shapes, validity, content-aware retrieval,
and basic causal properties.
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
        d_c=128,          # ← change to 256 to match production config
        local_window=64,
        top_k=32,
        block_size=32,
        retrieval_M=4,
        retrieval_bucket_width=0.5,
        retrieval_c_max=16,
        bucket_count=8,
    )
    model.eval()
    return model


def test_output_shape(attention):
    """Test that output shape matches input shape."""

    x = torch.randn(2, 128, 512)

    with torch.no_grad():
        output = attention(x)

    assert output.shape == x.shape


def test_output_validity(attention):
    """Test that output has no NaN or Inf values."""

    x = torch.randn(2, 128, 512)

    with torch.no_grad():
        output = attention(x)

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


@pytest.mark.parametrize("seq_len", [16, 32, 64, 128, 256, 512])
def test_multiple_sequence_lengths(attention, seq_len):
    """Test that GQLSA works with different sequence lengths."""

    x = torch.randn(1, seq_len, 512)

    with torch.no_grad():
        output = attention(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_grouped_heads_validation():
    """Test that invalid h/g ratio raises assertion."""

    with pytest.raises(AssertionError):
        GQLSA(
            d_model=512,
            h=7,  # Not divisible by g=2
            g=2,
            d_k=64,
            d_v=64,
            d_c=128,
            local_window=64,
            top_k=32,
            block_size=32,
        )


def test_content_aware_retrieval(attention):
    """
    specific: verify that block selection depends on input content.

    Uses two structured inputs whose block-level latent summaries are
    genuinely different: one where each block carries a distinct ramp
    value, and one where blocks alternate between two distinct
    clusters. Random noise cannot serve here — it has no block-level
    structure, so retrieval legitimately returns the same pattern
    for both.
    """

    seq_len = 1024
    N = seq_len // attention.block_size
    block_size = attention.block_size
    d_model = attention.d_model

    # ── Input 1: smooth ramp. Block i has value i/N. ──
    x1 = torch.zeros(1, seq_len, d_model)
    for i in range(N):
        x1[0, i * block_size:(i + 1) * block_size, :] = float(i) / N

    # ── Input 2: alternating blocks. Even blocks all +1, odd all -1. ──
    x2 = torch.zeros(1, seq_len, d_model)
    for i in range(N):
        val = 1.0 if i % 2 == 0 else -1.0
        x2[0, i * block_size:(i + 1) * block_size, :] = val

    with torch.no_grad():
        # Warm-up so _proj is created.
        _warm = torch.randn(1, 64, d_model)
        attention._C_q_mean = attention.q_compress(_warm).view(
            1, 2, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = attention.kv_compress(_warm).view(
            1, 2, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        attention._get_block_indices(64, _warm.device)

        # Input 1
        C_q1 = attention.q_compress(x1)
        C_kv1 = attention.kv_compress(x1)
        attention._C_q_mean = C_q1.view(
            1, N, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = C_kv1.view(
            1, N, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        idx1, _ = attention._get_block_indices(seq_len, x1.device)

        # Input 2
        C_q2 = attention.q_compress(x2)
        C_kv2 = attention.kv_compress(x2)
        attention._C_q_mean = C_q2.view(
            1, N, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = C_kv2.view(
            1, N, block_size, attention.d_c
        ).mean(dim=2).mean(dim=0)
        idx2, _ = attention._get_block_indices(seq_len, x2.device)

    assert not torch.equal(idx1, idx2), (
        "Block indices are identical for structurally different inputs. "
        "The retrieval is not responding to content."
    )


def test_causality(attention):
    """
    Verify that a perturbation at position 5 does not affect
    outputs at positions 0-4.
    """

    x = torch.randn(2, 128, 512)
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] += 100.0

    with torch.no_grad():
        out1 = attention(x)
        out2 = attention(x_perturbed)

    max_diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
    assert max_diff < 1e-5, (
        f"Causal leakage detected: max diff {max_diff} at positions 0-4"
    )