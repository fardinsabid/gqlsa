"""
Inference demo for GQLSA — shows how to use in a transformer layer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from attention.gqlsa import GQLSA, GQLSAState


class TransformerBlock(nn.Module):
    """Minimal transformer block using GQLSA."""
    
    def __init__(
        self,
        d_model,
        h,
        g,
        d_k,
        d_v,
        d_c,
        local_window,
        top_k,
        block_size,
        retrieval_M=4,
        retrieval_bucket_width=0.5,
        retrieval_c_max=16,
        bucket_count=8,
    ):
        super().__init__()
        
        self.attention = GQLSA(
            d_model=d_model,
            h=h,
            g=g,
            d_k=d_k,
            d_v=d_v,
            d_c=d_c,
            local_window=local_window,
            top_k=top_k,
            block_size=block_size,
            retrieval_M=retrieval_M,
            retrieval_bucket_width=retrieval_bucket_width,
            retrieval_c_max=retrieval_c_max,
            bucket_count=bucket_count,
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def main():
    config = {
        "d_model": 256,
        "h": 8,
        "g": 4,
        "d_k": 32,
        "d_v": 32,
        "d_c": 128,
        "local_window": 64,
        "top_k": 32,
        "block_size": 32,
        # retrieval constants
        "retrieval_M": 4,
        "retrieval_bucket_width": 0.5,
        "retrieval_c_max": 16,
        "bucket_count": 8,
    }
    
    block = TransformerBlock(**config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = block.to(device)
    block.eval()
    
    x = torch.randn(1, 128, config["d_model"], device=device)
    
    with torch.no_grad():
        output = block(x)
    
    print(f"Transformer Block Demo (GQLSA)")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Device:       {device}")
    print(f"  Valid output: {not torch.isnan(output).any().item()}")
    print(f"  Retrieval:    M={config['retrieval_M']}, "
          f"bucket_width={config['retrieval_bucket_width']}, "
          f"c_max={config['retrieval_c_max']}")

    # ── Cached inference (v1.0.1) ──
    # For generation, use GQLSA.forward_step() one token at a time.
    # Cost per token is O(k_eff · block_size) — constant in sequence length,
    # instead of the O(T) cost of re-running forward() on the growing prefix.

    print("\n" + "=" * 60)
    print("Cached generation demo")
    print("=" * 60)

    demo_len = 128
    x_demo = torch.randn(1, demo_len, config["d_model"], device=device)

    # Reference: full forward through the block
    with torch.no_grad():
        out_full = block(x_demo)

    # Cached: run the block one token at a time. The attention uses
    # forward_step(); the residual and FFN are per-token and stateless.
    attn = block.attention
    state: GQLSAState = attn.init_state(
        batch_size=1, max_seq_len=demo_len, device=device
    )

    outs = []
    with torch.no_grad():
        for i in range(demo_len):
            xi = x_demo[:, i:i+1]                             # [1, 1, d_model]
            # attention sublayer
            h_in = block.norm1(xi)
            attn_out = attn.forward_step(h_in, state, start_pos=i)
            xi = xi + attn_out
            # ffn sublayer
            xi = xi + block.ffn(block.norm2(xi))
            outs.append(xi)
    out_cached = torch.cat(outs, dim=1)

    diff = (out_full - out_cached).abs().max().item()
    print(f"Block output diff (cached vs full): {diff:.3e}")
    print(f"Result: {'PASS' if diff < 1e-4 else 'FAIL'}")
    print(f"State type: {type(state).__name__}")


if __name__ == "__main__":
    main()