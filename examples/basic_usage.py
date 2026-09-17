"""
Basic usage example for GQLSA.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from attention.gqlsa import GQLSA


def main():
    # Create GQLSA attention module
    attention = GQLSA(
        d_model=4096,
        h=32,
        g=4,
        d_k=128,
        d_v=128,
        d_c=512,
        local_window=128,
        top_k=64,
        block_size=32,
        # retrieval constants
        retrieval_M=4,                 # number of random projection lines
        retrieval_bucket_width=0.5,    # width of content buckets
        retrieval_c_max=16,            # max candidates retrieved per line
        bucket_count=8,
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)
    attention.eval()
    
    # Create random input
    batch_size = 2
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 4096, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = attention(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output NaN:  {torch.isnan(output).any().item()}")
    print(f"Output Inf:  {torch.isinf(output).any().item()}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std:  {output.std().item():.4f}")
    
    # Verify causality
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] += 100.0
    
    with torch.no_grad():
        out1 = attention(x)
        out2 = attention(x_perturbed)
    
    max_diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
    print(f"\nCausality check (positions 0-4 unaffected by position 5):")
    print(f"  Max diff: {max_diff:.10f}")
    print(f"  Result: {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    
    # specific: verify content-aware retrieval is active.
    # Run two different inputs and confirm the block selection differs.
    with torch.no_grad():
        C_q1 = attention.q_compress(x)
        C_kv1 = attention.kv_compress(x)
        C_q2 = attention.q_compress(x_perturbed)
        C_kv2 = attention.kv_compress(x_perturbed)
        attention._C_q_mean = C_q1.view(batch_size, -1, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = C_kv1.view(batch_size, -1, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        idx1, _ = attention._get_block_indices(seq_len, device)
        attention._C_q_mean = C_q2.view(batch_size, -1, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        attention._C_kv_mean = C_kv2.view(batch_size, -1, attention.block_size, attention.d_c).mean(dim=2).mean(dim=0)
        idx2, _ = attention._get_block_indices(seq_len, device)
    
    content_dependent = not torch.equal(idx1, idx2)
    print(f"\nContent-dependence check (specific):")
    print(f"  Block indices differ for different inputs: {content_dependent}")
    print(f"  Result: {'PASS' if content_dependent else 'FAIL'}")


if __name__ == "__main__":
    main()