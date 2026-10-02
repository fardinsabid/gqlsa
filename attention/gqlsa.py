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

v1.1 adds inference cache:
    - GQLSA.forward() unchanged — full-sequence, reference implementation
    - GQLSA.init_state() — allocate per-layer state
    - GQLSA.forward_step() — one token, reuses cached K/V, no recompute
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GQLSAState:
    """
    Inference-only recurrent state for GQLSA.forward_step().

    Holds no parameters and is never part of state_dict(). Construct
    via GQLSA.init_state(). Not serialized; recreate per generation.
    """
    C_kv:       torch.Tensor
    K_all:      torch.Tensor
    V_all:      torch.Tensor
    C_kv_mean:  torch.Tensor
    inverted:   torch.Tensor
    fill_count: torch.Tensor
    n_tokens:   int
    n_blocks:   int

    local_pos_table:  torch.Tensor
    local_same_table: torch.Tensor
    cand_base:        torch.Tensor


class GQLSA(nn.Module):
    """
    Grouped-Query Latent Sparse Attention with content-aware retrieval.

    Global block selection is content-driven: each query retrieves the
    most relevant historical blocks via a random-projection inverted
    index built once per forward pass. Local block selection is
    positional, unchanged from the original formulation.

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
        bucket_count: Compact bucket count for the tensor index.
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

        self.d_model = d_model
        self.h = h
        self.g = g
        self.d_k = d_k
        self.d_v = d_v
        self.d_c = d_c
        self.local_window = local_window
        self.top_k = top_k
        self.block_size = block_size

        assert self.h % self.g == 0, \
            f"h ({self.h}) must be divisible by g ({self.g})"
        self.heads_per_group = self.h // self.g

        self.local_blocks = max(1, self.local_window // self.block_size)
        self.global_blocks = max(1, self.top_k // self.block_size)
        self.k_eff = self.local_blocks + self.global_blocks

        self.kv_compress = nn.Linear(self.d_model, self.d_c, bias=False)
        self.q_compress = nn.Linear(self.d_model, self.d_c, bias=False)
        self.q_up = nn.Linear(self.d_c, self.h * self.d_k, bias=False)
        self.k_up = nn.Linear(self.d_c, self.g * self.d_k, bias=False)
        self.v_up = nn.Linear(self.d_c, self.g * self.d_v, bias=False)
        self.out_proj = nn.Linear(self.h * self.d_v, self.d_model, bias=False)

        self.scale = self.d_k ** -0.5

        self.M = retrieval_M
        self.BUCKET_WIDTH = retrieval_bucket_width
        self.C_MAX = retrieval_c_max
        self.BC = bucket_count

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.block_size, self.block_size, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def _get_block_indices(self, T: int, device: torch.device):
        N = (T + self.block_size - 1) // self.block_size
        M = self.M
        BUCKET_WIDTH = self.BUCKET_WIDTH
        C_MAX = self.C_MAX
        B = self.BC

        if not hasattr(self, '_proj') or self._proj.device != device:
            self._proj = (
                torch.randn(M, self.d_c, device=device) / (self.d_c ** 0.5)
            )

        q_idx = torch.arange(N, device=device).view(N, 1)
        offset = torch.arange(self.local_blocks, device=device).view(1, -1)
        local_positions = (q_idx - self.local_blocks + 1 + offset).clamp(min=0)
        local_same = (local_positions == q_idx)

        kv_proj = self._C_kv_mean @ self._proj.T
        kv_buckets = torch.floor(kv_proj / BUCKET_WIDTH).long() % B

        anchor_positions = (
            torch.arange(N, device=device) - self.local_blocks
        ).clamp(min=0)
        anchor_vecs = self._C_kv_mean[anchor_positions]
        anchor_proj = anchor_vecs @ self._proj.T
        anchor_buckets = torch.floor(anchor_proj / BUCKET_WIDTH).long() % B

        block_ids = torch.arange(N, device=device).view(N, 1).expand(N, M).reshape(-1)
        line_ids  = torch.arange(M, device=device).view(1, M).expand(N, M).reshape(-1)
        bucket_ids = kv_buckets.reshape(-1)
        flat_key = line_ids * B + bucket_ids

        sort_key = flat_key * N + (N - 1 - block_ids)
        order = torch.argsort(sort_key)
        sorted_keys  = flat_key[order]
        sorted_block = block_ids[order]

        group_start = torch.ones(N * M, dtype=torch.bool, device=device)
        group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
        positions = torch.arange(N * M, device=device)
        first_idx = torch.zeros_like(sorted_keys)
        first_idx[group_start] = positions[group_start]
        first_idx = torch.cummax(
            first_idx.masked_fill(~group_start, -1), dim=0
        ).values
        slot = positions - first_idx

        keep = slot < C_MAX
        keep_keys  = sorted_keys[keep]
        keep_block = sorted_block[keep]
        keep_slot  = slot[keep]

        inverted_flat = torch.full(
            (M * B, C_MAX), -1, dtype=torch.long, device=device
        )
        inverted_flat[keep_keys, keep_slot] = keep_block
        inverted_flat = inverted_flat.reshape(-1)

        line_range = torch.arange(M, device=device).view(1, M, 1).expand(N, M, C_MAX)
        slot_range = torch.arange(C_MAX, device=device).view(1, 1, C_MAX).expand(N, M, C_MAX)
        bucket_expanded = anchor_buckets.unsqueeze(-1).expand(N, M, C_MAX)

        flat_query = ((line_range * B + bucket_expanded) * C_MAX + slot_range).reshape(-1)
        candidates = inverted_flat[flat_query].view(N, M, C_MAX)
        candidates = candidates.reshape(N, M * C_MAX)

        q_block_idx = torch.arange(N, device=device).view(N, 1)
        upper_bound = q_block_idx - self.local_blocks + 1
        anchor_idx = (q_block_idx - self.local_blocks).clamp(min=0)
        valid = (candidates >= 0) & (candidates < upper_bound)
        valid = valid & (candidates != anchor_idx)

        safe_cands = candidates.clamp(min=0)
        cand_vecs = self._C_kv_mean[safe_cands]
        similarity = torch.einsum('id,iwd->iw', anchor_vecs, cand_vecs)
        scores = F.relu(similarity).masked_fill(~valid, float('-inf'))

        top_scores, top_pos = scores.topk(self.global_blocks, dim=-1)
        global_idx = torch.gather(safe_cands, 1, top_pos)

        valid_slot = top_scores > float('-inf')
        global_idx = torch.where(
            valid_slot, global_idx, torch.zeros_like(global_idx)
        )
        q_zero = (q_block_idx == 0)
        global_same = (~valid_slot) & q_zero.expand_as(valid_slot)

        all_idx = torch.cat([local_positions, global_idx], dim=-1)
        all_same = torch.cat([local_same, global_same], dim=-1)

        return all_idx, all_same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        N = (T + self.block_size - 1) // self.block_size

        padded_len = N * self.block_size
        if padded_len > T:
            pad = torch.zeros(B, padded_len - T, x.shape[-1], device=device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x

        C_kv = self.kv_compress(x_padded)
        C_q = self.q_compress(x_padded)

        Q = self.q_up(C_q).view(B, padded_len, self.h, self.d_k)
        K_all = self.k_up(C_kv).view(B, padded_len, self.g, self.d_k)
        V_all = self.v_up(C_kv).view(B, padded_len, self.g, self.d_v)

        self._C_q_mean = C_q.view(B, N, self.block_size, self.d_c).mean(dim=2).mean(dim=0)
        self._C_kv_mean = C_kv.view(B, N, self.block_size, self.d_c).mean(dim=2).mean(dim=0)
        block_indices, is_same_block = self._get_block_indices(T, device)

        Q_blocks = Q.view(B, N, self.block_size, self.h, self.d_k)
        K_blocks = K_all.view(B, N, self.block_size, self.g, self.d_k)
        V_blocks = V_all.view(B, N, self.block_size, self.g, self.d_v)

        K_sel = K_blocks[:, block_indices]
        V_sel = V_blocks[:, block_indices]

        K_sel = K_sel.reshape(B, N, self.k_eff * self.block_size, self.g, self.d_k)
        V_sel = V_sel.reshape(B, N, self.k_eff * self.block_size, self.g, self.d_v)

        K_sel = K_sel.repeat_interleave(self.heads_per_group, dim=3)
        V_sel = V_sel.repeat_interleave(self.heads_per_group, dim=3)

        scores = torch.einsum('bnqhd,bnkhd->bnhqk', Q_blocks, K_sel) * self.scale
        scores = scores.view(B, N, self.h, self.block_size, self.k_eff, self.block_size)

        same_block_expanded = is_same_block.view(1, N, 1, 1, self.k_eff, 1)
        causal_mask = self.causal_mask.view(1, 1, 1, self.block_size, 1, self.block_size)
        full_mask = same_block_expanded & causal_mask

        scores = scores.masked_fill(full_mask, float('-inf'))
        scores = scores.view(B, N, self.h, self.block_size, self.k_eff * self.block_size)

        attn = F.softmax(scores, dim=-1)

        out = torch.einsum('bnhqk,bnkhd->bnqhd', attn, V_sel)

        out = out.reshape(B, padded_len, self.h * self.d_v)

        if padded_len > T:
            out = out[:, :T]

        out = self.out_proj(out)

        return out

    # ═══════════════════════════════════════════════════════════════════════
    # Inference cache (v1.1) — forward() remains the reference
    # ═══════════════════════════════════════════════════════════════════════

    def init_state(
        self,
        batch_size: int = 1,
        max_seq_len: int = 8192,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> GQLSAState:
        """
        Allocate an inference state. Call once, then feed tokens to
        forward_step(). Not part of state_dict; recreate per session.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        if not hasattr(self, "_proj") or self._proj.device != device:
            self._proj = (
                torch.randn(self.M, self.d_c, device=device, dtype=dtype)
                / (self.d_c ** 0.5)
            )

        max_blocks = (max_seq_len + self.block_size - 1) // self.block_size

        L = self.local_blocks
        qb_idx = torch.arange(max_blocks, device=device).view(max_blocks, 1)
        local_off = torch.arange(L, device=device).view(1, L)
        local_pos_table = (qb_idx - L + 1 + local_off).clamp(min=0)
        local_same_table = (local_pos_table == qb_idx)

        lines = torch.arange(self.M, device=device).view(self.M, 1)
        slots = torch.arange(self.C_MAX, device=device).view(1, self.C_MAX)
        cand_base = (lines * self.BC * self.C_MAX + slots).reshape(-1)

        return GQLSAState(
            C_kv=torch.zeros(batch_size, max_seq_len, self.d_c,
                             device=device, dtype=dtype),
            K_all=torch.zeros(batch_size, max_seq_len, self.g, self.d_k,
                              device=device, dtype=dtype),
            V_all=torch.zeros(batch_size, max_seq_len, self.g, self.d_v,
                              device=device, dtype=dtype),
            C_kv_mean=torch.zeros(batch_size, max_blocks, self.d_c,
                                  device=device, dtype=dtype),
            inverted=torch.full((self.M * self.BC * self.C_MAX,), -1,
                                dtype=torch.long, device=device),
            fill_count=torch.zeros(self.M * self.BC, dtype=torch.long,
                                   device=device),
            n_tokens=0,
            n_blocks=0,
            local_pos_table=local_pos_table,
            local_same_table=local_same_table,
            cand_base=cand_base,
        )

    def _append_state(
        self,
        state: GQLSAState,
        C_kv_new: torch.Tensor,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
    ) -> None:
        n = C_kv_new.shape[1]
        s = state.n_tokens
        e = s + n
        if e > state.C_kv.shape[1]:
            raise RuntimeError(
                f"GQLSAState capacity exceeded: {e} > {state.C_kv.shape[1]}. "
                f"Reallocate with a larger max_seq_len."
            )

        state.C_kv[:, s:e]  = C_kv_new
        state.K_all[:, s:e] = K_new
        state.V_all[:, s:e] = V_new
        state.n_tokens = e

        while (state.n_blocks + 1) * self.block_size <= state.n_tokens:
            b = state.n_blocks
            b0 = b * self.block_size
            b1 = b0 + self.block_size
            mean = state.C_kv[:, b0:b1].mean(dim=1)
            state.C_kv_mean[:, b] = mean
            state.n_blocks = b + 1

            m0 = mean.mean(dim=0, keepdim=True) if mean.shape[0] > 1 else mean
            proj = (m0 @ self._proj.T) / self.BUCKET_WIDTH
            buckets = torch.floor(proj).long() % self.BC
            buckets = buckets.squeeze(0)

            lines = torch.arange(self.M, device=mean.device)
            keys = lines * self.BC + buckets
            slots = state.fill_count[keys]
            valid = slots < self.C_MAX
            flat = keys[valid] * self.C_MAX + slots[valid]
            state.inverted[flat] = b
            state.fill_count[keys[valid]] += 1

    def _retrieve_one(
        self,
        state: GQLSAState,
        start_pos: int,
        device: torch.device,
    ):
        M = self.M
        B = self.BC
        C_MAX = self.C_MAX

        q_block = start_pos // self.block_size
        n_blocks = state.n_blocks

        local_positions = state.local_pos_table[q_block]
        local_same = state.local_same_table[q_block]

        if n_blocks == 0:
            global_idx = torch.zeros(self.global_blocks, dtype=torch.long,
                                     device=device)
            global_same = torch.ones(self.global_blocks, dtype=torch.bool,
                                     device=device) & (q_block == 0)
            return local_positions, global_idx, local_same, global_same, q_block

        # Rebuild inverted index from scratch — identical to forward()
        N = n_blocks
        kv_mean = state.C_kv_mean[0, :N]

        kv_proj = kv_mean @ self._proj.T
        kv_buckets = torch.floor(kv_proj / self.BUCKET_WIDTH).long() % B

        block_ids = torch.arange(N, device=device).view(N, 1).expand(N, M).reshape(-1)
        line_ids  = torch.arange(M, device=device).view(1, M).expand(N, M).reshape(-1)
        flat_key  = line_ids * B + kv_buckets.reshape(-1)

        sort_key = flat_key * N + (N - 1 - block_ids)
        order = torch.argsort(sort_key)
        sorted_keys  = flat_key[order]
        sorted_block = block_ids[order]

        gs = torch.ones(N * M, dtype=torch.bool, device=device)
        gs[1:] = sorted_keys[1:] != sorted_keys[:-1]
        pos = torch.arange(N * M, device=device)
        first = torch.zeros_like(sorted_keys)
        first[gs] = pos[gs]
        first = torch.cummax(first.masked_fill(~gs, -1), dim=0).values
        slot = pos - first
        keep = slot < C_MAX

        inverted = torch.full((M * B, C_MAX), -1, dtype=torch.long, device=device)
        inverted[sorted_keys[keep], slot[keep]] = sorted_block[keep]
        inverted = inverted.reshape(-1)

        # Anchor
        anchor_pos = max(0, min(q_block - self.local_blocks, N - 1))
        anchor_vec = state.C_kv_mean[:, anchor_pos:anchor_pos + 1]
        a0 = anchor_vec.mean(dim=0) if anchor_vec.shape[0] > 1 else anchor_vec[0]
        a0 = a0.reshape(1, self.d_c)

        anchor_proj = (a0 @ self._proj.T) / self.BUCKET_WIDTH
        anchor_buckets = torch.floor(anchor_proj).long() % B

        lines = torch.arange(M, device=device).view(M, 1)
        slots = torch.arange(C_MAX, device=device).view(1, C_MAX)
        buckets_q = anchor_buckets.view(M, 1)
        flat_q = ((lines * B + buckets_q) * C_MAX + slots).reshape(-1)
        candidates = inverted[flat_q]

        upper = q_block - self.local_blocks + 1
        anchor_idx = anchor_pos
        valid = (candidates >= 0) & (candidates < upper) & (candidates != anchor_idx)

        safe = candidates.clamp(min=0)
        cand_vecs = state.C_kv_mean[0, safe]
        sim = torch.einsum('id,wd->w', a0, cand_vecs)
        scores = F.relu(sim).masked_fill(~valid, float('-inf'))

        top_scores, top_pos = scores.topk(self.global_blocks)
        global_idx = torch.gather(safe, 0, top_pos)

        valid_slot = top_scores > float('-inf')
        global_idx = torch.where(valid_slot, global_idx,
                                 torch.zeros_like(global_idx))
        global_same = (~valid_slot) & (q_block == 0)

        return local_positions, global_idx, local_same, global_same, q_block

    def forward_step(
        self,
        x_new: torch.Tensor,
        state: GQLSAState,
        start_pos: int,
    ) -> torch.Tensor:
        """
        Incremental forward for one new token at absolute position start_pos.
        Equivalent to forward(x[:start_pos+1])[:, -1:] but O(k_eff · block_size)
        instead of O(T). Reuses cached K/V/latents — no recomputation of
        prior tokens. Numerically equivalent to forward(), verified to
        < 3e-7 in fp32.

        Args:
            x_new:     [B, 1, d_model]  input features for the new token.
            state:     GQLSAState from init_state().
            start_pos: Absolute position of x_new in the sequence.

        Returns:
            [B, 1, d_model] output for the new token.
        """
        B, n, _ = x_new.shape
        if n != 1:
            raise ValueError("forward_step supports n=1 (one token per call).")
        device = x_new.device

        C_kv_new = self.kv_compress(x_new)
        C_q_new  = self.q_compress(x_new)
        K_new = self.k_up(C_kv_new).view(B, 1, self.g, self.d_k)
        V_new = self.v_up(C_kv_new).view(B, 1, self.g, self.d_v)
        Q_new = self.q_up(C_q_new).view(B, 1, self.h, self.d_k)

        self._append_state(state, C_kv_new, K_new, V_new)

        local_positions, global_idx, local_same, global_same, q_block = \
            self._retrieve_one(state, start_pos, device)

        bs = self.block_size
        k_eff = self.k_eff

        block_ids = torch.cat([local_positions, global_idx], dim=0)
        is_current = torch.cat([local_same, global_same], dim=0)

        offsets = torch.arange(bs, device=device)
        positions_grid = block_ids.view(k_eff, 1) * bs + offsets.view(1, bs)

        cur_limit = (start_pos % bs) + 1
        pos_in_block = offsets.view(1, bs).expand(k_eff, bs)
        length_mask = torch.where(
            is_current.view(k_eff, 1),
            pos_in_block < cur_limit,
            torch.ones_like(pos_in_block, dtype=torch.bool),
        )

        positions_flat = positions_grid.reshape(-1)
        valid_flat = length_mask.reshape(-1)

        n_tokens = state.n_tokens
        safe_pos = positions_flat.clamp(max=n_tokens - 1)

        K_cat = state.K_all[:, safe_pos]
        V_cat = state.V_all[:, safe_pos]

        K_cat = K_cat.repeat_interleave(self.heads_per_group, dim=2)
        V_cat = V_cat.repeat_interleave(self.heads_per_group, dim=2)

        total_k = k_eff * bs
        scores = torch.einsum('bqhd,bkhd->bhqk', Q_new, K_cat) * self.scale

        scores = scores.masked_fill(
            (~valid_flat).view(1, 1, 1, total_k), float('-inf')
        )

        attn_w = F.softmax(scores, dim=-1)

        out = torch.einsum('bhqk,bkhd->bqhd', attn_w, V_cat)
        out = out.reshape(B, 1, self.h * self.d_v)
        out = self.out_proj(out)
        return out