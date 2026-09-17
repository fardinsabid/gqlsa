"""
GQLSA: Grouped-Query Latent Sparse Attention

A hardware-native attention mechanism that combines:
- Latent compression for KV cache (like MLA)
- Grouped query sharing (like GQA)
- Block sparse attention with constant selection size
- Content-aware global block retrieval (fully vectorized)

The core philosophy: "All tokens don't need compute. Compute only where it matters."

Architecture Flow:
    1. Compress input to latent space (reduces memory)
    2. Up-project to Q, K, V heads (grouped for efficiency)
    3. Select sparse blocks (local positional + global content-retrieved)
    4. Gather selected K/V blocks (batched, no loops)
    5. Compute attention (single matmul, fully vectorized)
    6. Apply causal mask (autoregressive safety)
    7. Project output back to model dimension

Key Properties:
    - Linear O(T) attention instead of quadratic O(T²)
    - KV cache size = d_c (not h × d_k)
    - Constant attention span regardless of sequence length
    - Global blocks retrieved by content, not fixed position
    - No GPU→CPU synchronization in the retrieval path
    - No Python loops or dicts in the block-selection step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQLSA(nn.Module):
    """
    Grouped-Query Latent Sparse Attention with content-aware retrieval.

    Global block selection is content-driven: each query retrieves the
    most relevant historical blocks via a random-projection inverted
    index built once per forward pass. The index is constructed and
    queried entirely on-device, with no Python loops, no dicts, and no
    CPU↔GPU synchronization. Local block selection is positional,
    unchanged from the original formulation.

    Complexity is O(N · log(N · M)) for the index build (N = T / block_size,
    M = projection lines, both effectively constant per forward), plus
    O(N · M · C_MAX · d_c) for candidate scoring. Causality is preserved:
    every selected global block is strictly before the query's local
    window.

    Args:
        d_model: Model hidden dimension.
        h: Number of query heads.
        g: Number of KV groups (h must be divisible by g).
        d_k: Key/Query head dimension.
        d_v: Value head dimension.
        d_c: Latent compression dimension.
        local_window: Local attention window (tokens).
        top_k: Global sparse tokens per query.
        block_size: Tokens per sparse block.
        retrieval_M: Number of random projection lines.
        retrieval_bucket_width: Width of content buckets.
        retrieval_c_max: Max candidates retrieved per line.

    Example:
        >>> attention = GQLSA(
        ...     d_model=4096,
        ...     h=32,
        ...     g=4,
        ...     d_k=128,
        ...     d_v=128,
        ...     d_c=512,
        ...     local_window=128,
        ...     top_k=64,
        ...     block_size=32,
        ... )
        >>> x = torch.randn(2, 512, 4096)
        >>> output = attention(x)  # [2, 512, 4096]
    """
    
    def __init__(
        self,
        d_model: int,
        h: int,
        g: int,
        d_k: int,
        d_v: int,
        d_c: int,
        local_window: int,
        top_k: int,
        block_size: int,
        retrieval_M: int = 4,
        retrieval_bucket_width: float = 0.5,
        retrieval_c_max: int = 16,
        bucket_count: int = 8,
    ):
        super().__init__()
        
        # ── Core dimensions ──
        self.d_model = d_model          # Model hidden dimension
        self.h = h                      # Number of query heads
        self.g = g                      # Number of KV groups
        self.d_k = d_k                  # Key/Query head dimension
        self.d_v = d_v                  # Value head dimension
        self.d_c = d_c                  # Latent compression dimension
        self.local_window = local_window  # Local attention window (tokens)
        self.top_k = top_k              # Global sparse tokens per query
        self.block_size = block_size    # Tokens per sparse block
        
        # ── Validation ──
        assert self.h % self.g == 0, \
            f"h ({self.h}) must be divisible by g ({self.g})"
        self.heads_per_group = self.h // self.g  # Query heads sharing each KV group
        
        # ── Block-level sparsity configuration ──
        self.local_blocks = max(1, self.local_window // self.block_size)
        self.global_blocks = max(1, self.top_k // self.block_size)
        self.k_eff = self.local_blocks + self.global_blocks  # Total blocks per query
        
        # ── Latent compression layers ──
        # Compress input to low-dimensional latent space
        self.kv_compress = nn.Linear(self.d_model, self.d_c, bias=False)
        self.q_compress = nn.Linear(self.d_model, self.d_c, bias=False)
        
        # ── Up-projection layers ──
        # Reconstruct Q/K/V from latent space
        self.q_up = nn.Linear(self.d_c, self.h * self.d_k, bias=False)
        self.k_up = nn.Linear(self.d_c, self.g * self.d_k, bias=False)
        self.v_up = nn.Linear(self.d_c, self.g * self.d_v, bias=False)
        
        # ── Output projection ──
        self.out_proj = nn.Linear(self.h * self.d_v, self.d_model, bias=False)
        
        # ── Attention scaling factor ──
        # 1/sqrt(d_k) for dot product normalization
        self.scale = self.d_k ** -0.5
        
        # ── Retrieval constants ──
        self.M = retrieval_M
        self.BUCKET_WIDTH = retrieval_bucket_width
        self.C_MAX = retrieval_c_max
        self.B = bucket_count
        
        # ── Pre-computed causal mask ──
        # Upper triangular matrix to prevent attending to future tokens
        # within the same block. Applied to same-block positions only.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.block_size, self.block_size, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,  # Not saved in state_dict (derived from block_size)
        )
    
    def _get_block_indices(self, T: int, device: torch.device):
        """
        Block-level sparse attention pattern with tensor-indexed retrieval.

        The retrieval is fully vectorized. No Python loops, no dicts,
        no .tolist() calls, and no GPU→CPU synchronization. Every step
        is a bounded tensor operation.

        Local blocks: positional, same pattern as V1.
        Global blocks: ReLU-scored retrieval over the full history.
        Padding: preserves V1's (0 == q_block) causal fix.

        Complexity:
            - Index build (sort + slot assignment): O(N · M · log(N · M))
            - Candidate gather: O(N · M · C_MAX)
            - ReLU scoring: O(N · M · C_MAX · d_c)
            All N-dependent terms have constant multipliers except the
            log factor in the sort.
        """
        N = (T + self.block_size - 1) // self.block_size
        M = self.M
        BUCKET_WIDTH = self.BUCKET_WIDTH
        C_MAX = self.C_MAX
        B = 8   # compact bucket count for the tensor index

        if not hasattr(self, '_proj') or self._proj.device != device:
            self._proj = (
                torch.randn(M, self.d_c, device=device) / (self.d_c ** 0.5)
            )

        # ── Local blocks: vectorized ──
        q_idx = torch.arange(N, device=device).view(N, 1)
        offset = torch.arange(self.local_blocks, device=device).view(1, -1)
        local_positions = (q_idx - self.local_blocks + 1 + offset).clamp(min=0)
        local_same = (local_positions == q_idx)

        # ── Project blocks and anchors, compute compact bucket ids ──
        kv_proj = self._C_kv_mean @ self._proj.T
        kv_buckets = torch.floor(kv_proj / BUCKET_WIDTH).long() % B

        anchor_positions = (
            torch.arange(N, device=device) - self.local_blocks
        ).clamp(min=0)
        anchor_vecs = self._C_kv_mean[anchor_positions]
        anchor_proj = anchor_vecs @ self._proj.T
        anchor_buckets = torch.floor(anchor_proj / BUCKET_WIDTH).long() % B

        # ── Build inverted index via sort + scatter ──
        # Every (block, line) pair becomes an entry. Group by (line, bucket).
        block_ids = torch.arange(N, device=device).view(N, 1).expand(N, M).reshape(-1)
        line_ids  = torch.arange(M, device=device).view(1, M).expand(N, M).reshape(-1)
        bucket_ids = kv_buckets.reshape(-1)
        flat_key = line_ids * B + bucket_ids

        # Sort by (key ascending, block descending) so within each bucket
        # the most recent block comes first.
        sort_key = flat_key * N + (N - 1 - block_ids)
        order = torch.argsort(sort_key)
        sorted_keys  = flat_key[order]
        sorted_block = block_ids[order]

        # Slot within each (line, bucket) group.
        group_start = torch.ones(N * M, dtype=torch.bool, device=device)
        group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
        positions = torch.arange(N * M, device=device)
        first_idx = torch.zeros_like(sorted_keys)
        first_idx[group_start] = positions[group_start]
        first_idx = torch.cummax(
            first_idx.masked_fill(~group_start, -1), dim=0
        ).values
        slot = positions - first_idx

        # Keep only the first C_MAX entries per bucket.
        keep = slot < C_MAX
        keep_keys  = sorted_keys[keep]
        keep_block = sorted_block[keep]
        keep_slot  = slot[keep]

        inverted_flat = torch.full(
            (M * B, C_MAX), -1, dtype=torch.long, device=device
        )
        inverted_flat[keep_keys, keep_slot] = keep_block
        inverted_flat = inverted_flat.reshape(-1)   # flatten to 1D: [M*B*C_MAX]

        # ── Gather candidates: [N, M, C_MAX] ──
        # Flat index into inverted_flat:
        #   (line * B + bucket) * C_MAX + slot
        line_range = torch.arange(M, device=device).view(1, M, 1).expand(N, M, C_MAX)
        slot_range = torch.arange(C_MAX, device=device).view(1, 1, C_MAX).expand(N, M, C_MAX)
        bucket_expanded = anchor_buckets.unsqueeze(-1).expand(N, M, C_MAX)

        flat_query = ((line_range * B + bucket_expanded) * C_MAX + slot_range).reshape(-1)
        candidates = inverted_flat[flat_query].view(N, M, C_MAX)
        candidates = candidates.reshape(N, M * C_MAX)

        # ── Causal filter: block j valid for query i iff j < i - L_b + 1 ──
        # Also excludes the anchor itself, which would otherwise trivially
        # win the similarity ranking (||anchor||² is always the largest
        # ReLU(dot) value). Excluding it forces the retrieval to select
        # a genuinely different historical block.
        q_block_idx = torch.arange(N, device=device).view(N, 1)
        upper_bound = q_block_idx - self.local_blocks + 1
        anchor_idx = (q_block_idx - self.local_blocks).clamp(min=0)
        valid = (candidates >= 0) & (candidates < upper_bound)
        valid = valid & (candidates != anchor_idx)

        # ── ReLU scoring against the anchor ──
        safe_cands = candidates.clamp(min=0)
        cand_vecs = self._C_kv_mean[safe_cands]
        similarity = torch.einsum('id,iwd->iw', anchor_vecs, cand_vecs)
        scores = F.relu(similarity).masked_fill(~valid, float('-inf'))

        # ── Top-G_b global blocks per query ──
        top_scores, top_pos = scores.topk(self.global_blocks, dim=-1)
        global_idx = torch.gather(safe_cands, 1, top_pos)

        # ── Fallback for queries with no candidates ──
        valid_slot = top_scores > float('-inf')
        global_idx = torch.where(
            valid_slot, global_idx, torch.zeros_like(global_idx)
        )
        q_zero = (q_block_idx == 0)
        global_same = (~valid_slot) & q_zero.expand_as(valid_slot)

        # ── Concatenate local + global ──
        all_idx = torch.cat([local_positions, global_idx], dim=-1)
        all_same = torch.cat([local_same, global_same], dim=-1)

        return all_idx, all_same
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GQLSA.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        B, T, _ = x.shape
        device = x.device
        
        N = (T + self.block_size - 1) // self.block_size
        
        # ── Pad sequence to multiple of block_size ──
        # Required for clean block partitioning
        padded_len = N * self.block_size
        if padded_len > T:
            pad = torch.zeros(B, padded_len - T, x.shape[-1], device=device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x
        
        # ── Step 1: Latent Compression ──
        # Compress to low-dimensional latent space
        # [B, T_pad, d_c] each
        C_kv = self.kv_compress(x_padded)
        C_q = self.q_compress(x_padded)
        
        # ── Step 2: Up-Projection ──
        # Reconstruct Q/K/V from latent space
        # Q: [B, T_pad, h, d_k]
        # K: [B, T_pad, g, d_k] (grouped, fewer KV heads)
        # V: [B, T_pad, g, d_v]
        Q = self.q_up(C_q).view(B, padded_len, self.h, self.d_k)
        K_all = self.k_up(C_kv).view(B, padded_len, self.g, self.d_k)
        V_all = self.v_up(C_kv).view(B, padded_len, self.g, self.d_v)
        
        # ── Step 3: Sparse Selection ──
        # Pre-compute per-block mean latents for content retrieval
        self._C_q_mean = C_q.view(B, N, self.block_size, self.d_c).mean(dim=2).mean(dim=0)
        self._C_kv_mean = C_kv.view(B, N, self.block_size, self.d_c).mean(dim=2).mean(dim=0)
        block_indices, is_same_block = self._get_block_indices(T, device)
        
        # ── Step 4: Gather Selected Blocks ──
        # Reshape to block-level tensors
        Q_blocks = Q.view(B, N, self.block_size, self.h, self.d_k)
        K_blocks = K_all.view(B, N, self.block_size, self.g, self.d_k)
        V_blocks = V_all.view(B, N, self.block_size, self.g, self.d_v)
        
        # Gather selected K/V blocks using block_indices
        # [B, N, k_eff, block_size, g, d_k]
        K_sel = K_blocks[:, block_indices]
        V_sel = V_blocks[:, block_indices]
        
        # Merge k_eff and block_size dimensions
        # [B, N, k_eff * block_size, g, d_k]
        K_sel = K_sel.reshape(B, N, self.k_eff * self.block_size, self.g, self.d_k)
        V_sel = V_sel.reshape(B, N, self.k_eff * self.block_size, self.g, self.d_v)
        
        # Expand KV for all query heads in each group
        # [B, N, k_tokens, h, d_k]
        K_sel = K_sel.repeat_interleave(self.heads_per_group, dim=3)
        V_sel = V_sel.repeat_interleave(self.heads_per_group, dim=3)
        
        # ── Step 5: Batched Attention with Causal Mask ──
        # Single vectorized matmul for all blocks
        # scores: [B, N, h, block_size, k_eff * block_size]
        scores = torch.einsum('bnqhd,bnkhd->bnhqk', Q_blocks, K_sel) * self.scale
        
        # Reshape to separate k_eff dimension for masking
        # [B, N, h, block_size, k_eff, block_size]
        scores = scores.view(B, N, self.h, self.block_size, self.k_eff, self.block_size)
        
        # Apply causal mask only to same-block positions
        # Prevents attending to future tokens within the current block
        same_block_expanded = is_same_block.view(1, N, 1, 1, self.k_eff, 1)
        causal_mask = self.causal_mask.view(1, 1, 1, self.block_size, 1, self.block_size)
        full_mask = same_block_expanded & causal_mask
        
        scores = scores.masked_fill(full_mask, float('-inf'))
        
        # Reshape back for softmax
        scores = scores.view(B, N, self.h, self.block_size, self.k_eff * self.block_size)
        
        # Softmax over key dimension
        attn = F.softmax(scores, dim=-1)
        
        # ── Step 6: Output Computation ──
        # Single vectorized matmul for all blocks
        # out: [B, N, h, block_size, d_v]
        out = torch.einsum('bnhqk,bnkhd->bnqhd', attn, V_sel)
        
        # ── Step 7: Reshape and Project ──
        # Flatten heads and blocks back to sequence
        out = out.reshape(B, padded_len, self.h * self.d_v)
        
        # Remove padding tokens
        if padded_len > T:
            out = out[:, :T]
        
        # Final output projection
        out = self.out_proj(out)
        
        return out