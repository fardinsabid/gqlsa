"""
Inference demo for GQLSA — shows how to use in a transformer layer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from attention.gqlsa import GQLSA


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


if __name__ == "__main__":
    main()