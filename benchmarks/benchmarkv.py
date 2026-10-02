"""
GQLSA Complete Unified Benchmark
Tests: Causality, Speed, Memory, Quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import requests


# ============================================
# BASELINES
# ============================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        
        self.q_proj = nn.Linear(d_model, h * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, h * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, h * d_v, bias=False)
        self.out_proj = nn.Linear(h * d_v, d_model, bias=False)
        
        self.scale = d_k ** -0.5
    
    def forward(self, x):
        B, T, _ = x.shape
        
        Q = self.q_proj(x).view(B, T, self.h, self.d_k)
        K = self.k_proj(x).view(B, T, self.h, self.d_k)
        V = self.v_proj(x).view(B, T, self.h, self.d_v)
        
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
        out = out.reshape(B, T, -1)
        
        return self.out_proj(out)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, h, g, d_k, d_v):
        super().__init__()
        self.h = h
        self.g = g
        self.d_k = d_k
        self.d_v = d_v
        
        assert h % g == 0
        
        self.q_proj = nn.Linear(d_model, h * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, g * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, g * d_v, bias=False)
        self.out_proj = nn.Linear(h * d_v, d_model, bias=False)
        
        self.scale = d_k ** -0.5
    
    def forward(self, x):
        B, T, _ = x.shape
        
        Q = self.q_proj(x).view(B, T, self.h, self.d_k)
        K = self.k_proj(x).view(B, T, self.g, self.d_k)
        V = self.v_proj(x).view(B, T, self.g, self.d_v)
        
        K = K.repeat_interleave(self.h // self.g, dim=2)
        V = V.repeat_interleave(self.h // self.g, dim=2)
        
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
        out = out.reshape(B, T, -1)
        
        return self.out_proj(out)


class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, d_c):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_c = d_c
        
        self.kv_compress = nn.Linear(d_model, d_c, bias=False)
        self.q_compress = nn.Linear(d_model, d_c, bias=False)
        
        self.q_up = nn.Linear(d_c, h * d_k, bias=False)
        self.k_up = nn.Linear(d_c, h * d_k, bias=False)
        self.v_up = nn.Linear(d_c, h * d_v, bias=False)
        
        self.out_proj = nn.Linear(h * d_v, d_model, bias=False)
        
        self.scale = d_k ** -0.5
    
    def forward(self, x):
        B, T, _ = x.shape
        
        c_kv = self.kv_compress(x)
        c_q = self.q_compress(x)
        
        Q = self.q_up(c_q).view(B, T, self.h, self.d_k)
        K = self.k_up(c_kv).view(B, T, self.h, self.d_k)
        V = self.v_up(c_kv).view(B, T, self.h, self.d_v)
        
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bkhd->bqhd', attn, V)
        out = out.reshape(B, T, -1)
        
        return self.out_proj(out)


# ============================================
# GQLSA (vectorized content-aware retrieval)
# Adapted from model/attention/gqlsa.py for the benchmark's
# kwargs-style constructor. Logic identical to codebase version.
# ============================================

class GQLSA(nn.Module):
    def __init__(self, d_model, h, g, d_k, d_v, d_c,
                 local_window, top_k, block_size,
                 retrieval_M=4, retrieval_bucket_width=0.5, retrieval_c_max=16):
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

        assert self.h % self.g == 0
        self.heads_per_group = self.h // self.g

        self.local_blocks = max(1, self.local_window // self.block_size)
        self.global_blocks = max(1, self.top_k // self.block_size)
        self.k_eff = self.local_blocks + self.global_blocks

        self.M = retrieval_M
        self.BUCKET_WIDTH = retrieval_bucket_width
        self.C_MAX = retrieval_c_max

        self.kv_compress = nn.Linear(self.d_model, self.d_c, bias=False)
        self.q_compress = nn.Linear(self.d_model, self.d_c, bias=False)

        self.q_up = nn.Linear(self.d_c, self.h * self.d_k, bias=False)
        self.k_up = nn.Linear(self.d_c, self.g * self.d_k, bias=False)
        self.v_up = nn.Linear(self.d_c, self.g * self.d_v, bias=False)

        self.out_proj = nn.Linear(self.h * self.d_v, self.d_model, bias=False)
        self.scale = self.d_k ** -0.5

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.block_size, self.block_size, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def _get_block_indices(self, T, device):
        N = (T + self.block_size - 1) // self.block_size
        M = self.M
        BUCKET_WIDTH = self.BUCKET_WIDTH
        C_MAX = self.C_MAX
        B = 8

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

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        N = (T + self.block_size - 1) // self.block_size

        padded_len = N * self.block_size
        if padded_len > T:
            pad = torch.zeros(B, padded_len - T, x.shape[-1],
                              device=device, dtype=x.dtype)
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

        return self.out_proj(out)


# ============================================
# MINI TRANSFORMER FOR QUALITY TEST
# ============================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, attention, expansion=4):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, attention_type, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(256, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, attention_type(**config)) 
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.lm_head(x)


# ============================================
# BENCHMARK FUNCTIONS
# ============================================

def benchmark_speed(model, x, warmup=10, repeat=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.time()
            model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000)
    
    return sum(times) / len(times)


def benchmark_memory(model, x):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def test_causality(model, T=64, d_model=512):
    """
    Output at position t should NOT depend on input at position t+1.
    """
    model.eval()
    device = next(model.parameters()).device
    
    x = torch.randn(2, T, d_model, device=device)
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] += 100.0
    
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x_perturbed)
    
    max_diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
    
    return max_diff < 1e-5


def benchmark_cache(attn, seq_len, device, max_new=None):
    """
    Compare forward() on the growing prefix vs forward_step() on the same
    GQLSA instance. Returns (t_full, t_cached, speedup, max_diff) where
    t_* are seconds and max_diff verifies output equivalence.

    Uses the real attention.gqlsa.GQLSA — the cache is a v1.0.1 feature
    of the codebase module, not the inline benchmark copy.
    """
    if max_new is None:
        max_new = seq_len

    x = torch.randn(1, seq_len, attn.d_model, device=device)

    # ── baseline: forward() on growing prefix ──
    with torch.no_grad():
        for i in range(min(32, max_new)):
            _ = attn(x[:, :i+1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(max_new):
            _ = attn(x[:, :i+1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_full = time.time() - t0

    # ── cached: forward_step() ──
    state = attn.init_state(batch_size=1, max_seq_len=seq_len + 8, device=device)
    with torch.no_grad():
        for i in range(min(32, max_new)):
            _ = attn.forward_step(x[:, i:i+1], state, i)
        state = attn.init_state(batch_size=1, max_seq_len=seq_len + 8, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(max_new):
            _ = attn.forward_step(x[:, i:i+1], state, i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cached = time.time() - t0

    # ── correctness ──
    with torch.no_grad():
        out_full = attn(x)
    state = attn.init_state(batch_size=1, max_seq_len=seq_len + 8, device=device)
    outs = []
    with torch.no_grad():
        for i in range(seq_len):
            outs.append(attn.forward_step(x[:, i:i+1], state, i))
    out_step = torch.cat(outs, dim=1)
    diff = (out_full - out_step).abs().max().item()

    return t_full, t_cached, t_full / t_cached, diff


def train_and_evaluate(model, data, optimizer, steps=300, batch_size=8, seq_len=128):
    losses = []
    model.train()
    
    for step in range(steps):
        idx = random.randint(0, data.shape[0] - batch_size * seq_len - 1)
        x = data[idx:idx + batch_size * seq_len].view(batch_size, seq_len)
        y = data[idx + 1:idx + batch_size * seq_len + 1].view(batch_size, seq_len)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


# ============================================
# MAIN BENCHMARK
# ============================================

print("=" * 70)
print("GQLSA UNIFIED BENCHMARK")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

config = {
    'd_model': 4096,
    'h': 32,
    'g': 4,
    'd_k': 128,
    'd_v': 128,
    'd_c': 512,
    'local_window': 128,
    'top_k': 64,
    'block_size': 32,
}

# ═══════════════════════════════════
# TEST 1: CAUSALITY
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("TEST 1: CAUSALITY CHECK")
print(f"{'='*70}")

models = {
    'MHA': MultiHeadAttention(config['d_model'], config['h'], config['d_k'], config['d_v']).to(device),
    'GQA': GroupedQueryAttention(config['d_model'], config['h'], config['g'], config['d_k'], config['d_v']).to(device),
    'MLA': MultiLatentAttention(config['d_model'], config['h'], config['d_k'], config['d_v'], config['d_c']).to(device),
    'GQLSA': GQLSA(
        config['d_model'], config['h'], config['g'],
        config['d_k'], config['d_v'], config['d_c'],
        config['local_window'], config['top_k'], config['block_size']
    ).to(device),
}

for name, model in models.items():
    is_causal = test_causality(model, T=64, d_model=config['d_model'])
    status = "✅ PASS" if is_causal else "❌ FAIL — LEAKAGE DETECTED"
    print(f"  {name:<10} {status}")

# ═══════════════════════════════════
# TEST 2: SPEED
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("TEST 2: SPEED BENCHMARK")
print(f"{'='*70}")

seq_lengths = [512, 1024, 2048, 4096]
speed_results = {}

for T in seq_lengths:
    print(f"\n  Sequence Length: {T}")
    x = torch.randn(1, T, config['d_model'], device=device)
    speed_results[T] = {}
    
    for name, model in models.items():
        model.eval()
        try:
            avg_ms = benchmark_speed(model, x)
            speed_results[T][name] = avg_ms
            print(f"    {name:<10} {avg_ms:>10.2f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"    {name:<10} {'OOM':>10}")
            speed_results[T][name] = float('inf')
        torch.cuda.empty_cache()

# ═══════════════════════════════════
# TEST 3: MEMORY
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("TEST 3: MEMORY BENCHMARK")
print(f"{'='*70}")

mem_results = {}

for T in seq_lengths:
    print(f"\n  Sequence Length: {T}")
    x = torch.randn(1, T, config['d_model'], device=device)
    mem_results[T] = {}
    
    for name, model in models.items():
        model.eval()
        try:
            mem_mb = benchmark_memory(model, x)
            mem_results[T][name] = mem_mb
            print(f"    {name:<10} {mem_mb:>10.2f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"    {name:<10} {'OOM':>10}")
            mem_results[T][name] = float('inf')
        torch.cuda.empty_cache()

# ═══════════════════════════════════
# TEST 4: QUALITY
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("TEST 4: QUALITY (WikiText-2)")
print(f"{'='*70}")

print("\nDownloading WikiText-2...")
url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
response = requests.get(url)
text = response.text

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long).to(device)

print(f"Vocabulary size: {vocab_size}")
print(f"Data size: {encoded.shape[0]:,} tokens")

quality_config = {
    'd_model': 256, 'h': 8, 'g': 4, 'd_k': 32, 'd_v': 32, 'd_c': 128,
    'local_window': 64, 'top_k': 32, 'block_size': 32,
}

quality_models = {
    'MHA': MiniLanguageModel(vocab_size, 256, 4, MultiHeadAttention, {
        'd_model': 256, 'h': 8, 'd_k': 32, 'd_v': 32
    }).to(device),
    'GQA': MiniLanguageModel(vocab_size, 256, 4, GroupedQueryAttention, {
        'd_model': 256, 'h': 8, 'g': 4, 'd_k': 32, 'd_v': 32
    }).to(device),
    'MLA': MiniLanguageModel(vocab_size, 256, 4, MultiLatentAttention, {
        'd_model': 256, 'h': 8, 'd_k': 32, 'd_v': 32, 'd_c': 128
    }).to(device),
    'GQLSA': MiniLanguageModel(vocab_size, 256, 4, GQLSA, quality_config).to(device),
}

quality_results = {}

for name, model in quality_models.items():
    print(f"\n  Training {name}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = train_and_evaluate(model, encoded, optimizer, steps=300, batch_size=8, seq_len=128)
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, 2000 - 128, 128):
            x = encoded[i:i+128].unsqueeze(0)
            y = encoded[i+1:i+129].unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            total_loss += loss.item() * 128
            total_tokens += 128
    
    ppl = math.exp(total_loss / total_tokens)
    quality_results[name] = {'final_loss': losses[-1], 'perplexity': ppl}
    
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"    Perplexity: {ppl:.2f}")

# ═══════════════════════════════════
# TEST 5: INFERENCE CACHE
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("TEST 5: INFERENCE CACHE (forward_step vs forward)")
print(f"{'='*70}")
print("\n  Note: only GQLSA has a cache. This test compares GQLSA against")
print("  itself — forward() on the growing prefix vs forward_step() per token.")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from attention.gqlsa import GQLSA as GQLSA_Real

cache_model = GQLSA_Real(
    d_model=config['d_model'],
    h=config['h'], g=config['g'],
    d_k=config['d_k'], d_v=config['d_v'], d_c=config['d_c'],
    local_window=config['local_window'],
    top_k=config['top_k'],
    block_size=config['block_size'],
).to(device).eval()

cache_results = {}
print(f"\n  {'T':<8} {'forward()':<16} {'forward_step()':<18} {'speedup':<10} {'max diff'}")
print("  " + "-" * 62)
for T in [128, 256, 512, 1024]:
    t_full, t_cached, speedup, diff = benchmark_cache(
        cache_model, T, device, max_new=T
    )
    cache_results[T] = {'full': t_full, 'cached': t_cached, 'speedup': speedup, 'diff': diff}
    print(f"  {T:<8} {t_full*1000:>8.2f} ms    {t_cached*1000:>8.2f} ms      "
          f"{speedup:>5.2f}×     {diff:.2e}")

# ═══════════════════════════════════
# SUMMARY
# ═══════════════════════════════════
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n{'Model':<10} {'Speed(ms)':<12} {'Mem(MB)':<12} {'PPL':<10} {'Causal':<10}")
print("-" * 55)

for name in ['MHA', 'GQA', 'MLA', 'GQLSA']:
    speed = speed_results[4096].get(name, float('inf'))
    mem = mem_results[4096].get(name, float('inf'))
    ppl = quality_results.get(name, {}).get('perplexity', float('inf'))
    causal = "✅" if test_causality(models[name], T=64, d_model=config['d_model']) else "❌"
    
    speed_str = f"{speed:.2f}" if speed != float('inf') else "OOM"
    mem_str = f"{mem:.2f}" if mem != float('inf') else "OOM"
    ppl_str = f"{ppl:.2f}" if ppl != float('inf') else "—"
    
    print(f"{name:<10} {speed_str:<12} {mem_str:<12} {ppl_str:<10} {causal:<10}")

print(f"\n{'='*70}")
print("INFERENCE CACHE SUMMARY")
print(f"{'='*70}")
print(f"\n  {'T':<8} {'forward()':<16} {'forward_step()':<18} {'speedup'}")
print("  " + "-" * 55)
for T, r in cache_results.items():
    print(f"  {T:<8} {r['full']*1000:>8.2f} ms    {r['cached']*1000:>8.2f} ms      {r['speedup']:>5.2f}×")

print("\n" + "=" * 70)
print("✅ Benchmark complete!")
print("=" * 70)