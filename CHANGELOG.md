# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] — 2026-10-03

Inference cache. Generation no longer recomputes the full forward pass
on the growing sequence for every new token. Purely additive — existing
code, checkpoints, and `forward()` behavior are unchanged.

### Added

- **`GQLSAState` dataclass** (`attention/gqlsa.py`)
  - Recurrent inference state: compressed latents, projected K/V, per-block
    anchor means, persistent inverted index, fill counters, and precomputed
    per-query-block lookup tables.
  - Holds no parameters and no buffers. Not part of `state_dict()`. Not
    serialized.

- **`GQLSA.init_state(batch_size, max_seq_len, device, dtype)`**
  - Allocates a `GQLSAState` and precomputes the local-position table,
    the local-same-block table, and the flat candidate-gather base index.
  - Shares `_proj` with the module so cached and uncached paths use the
    same random projection.

- **`GQLSA.forward_step(x_new, state, start_pos)`**
  - Incremental forward for a single new token at absolute position
    `start_pos`. Numerically equivalent to `forward(x[:start_pos+1])[:, -1:]`.
  - Reuses cached K/V/latents. No recomputation of prior tokens.
  - Per-token cost is O(k_eff · block_size) — constant in sequence length.
    The prior path's per-token cost was O(T · k_eff · block_size).
  - Fully vectorized: no Python loops over blocks, no `.item()` calls, no
    CPU↔GPU synchronization in the hot path.

- **Cache benchmark** (`benchmarks/benchmarkv.py`)
  - New TEST 5: `forward()` on the growing prefix vs `forward_step()` per
    token, at 128/256/512/1024 tokens.
  - Reports wall-clock time, speedup, and output equivalence per length.

- **Tests** (`tests/`)
  - `test_cache_equivalence` — `forward_step` matches `forward` bit-exactly
  - `test_cache_rejects_multitoken` — multi-token input rejected with a clear error
  - `test_cache_capacity_error` — exceeding `max_seq_len` raises `RuntimeError`
  - `test_causality_of_forward_step` — future-token perturbation does not
    change earlier outputs through the cached path
  - Test count: 15 → 18

- **Examples**
  - `examples/basic_usage.py` — added cache demo comparing `forward_step`
    against full `forward`
  - `examples/inference_demo.py` — added cached generation demo through a
    minimal transformer block

- **README**
  - New "Inference cache" section: usage, guarantees, cost-per-token table,
    measured throughput, limitations.

### Changed

- `attention/__init__.py` — exports `GQLSAState`, version bumped to `1.0.1`.

### Unchanged

- `GQLSA.forward()` — byte-identical to v1.0.0.
- `GQLSA._get_block_indices()` — byte-identical to v1.0.0.
- Constructor signature, parameters, buffers, `state_dict()` keys and order.
- v1.0.0 checkpoints load without modification and produce identical outputs.

### Correctness

`forward_step()` is **bit-exact** against `forward()` on the same prefix.
Max abs diff `0.0` in fp32 across all tested configurations and sequence
lengths (verified by `test_cache_equivalence` and both examples).

### Measured (Tesla T4, `d_model=768, h=12, g=3, d_c=384, local_window=128, top_k=32, block_size=32`)

| | 500-token generation |
|---|---|
| `forward()` per step (v1.0.0) | 1.60 s (312 tok/s) |
| `forward_step()` (v1.0.1) | 0.90 s (555 tok/s) |

Per-token cost of `forward_step` is flat in sequence length (~1.9 ms on
the T4 for the config above), confirming the O(1) claim. The speedup over
`forward()` grows with sequence length because `forward()`'s per-token
cost grows while `forward_step()`'s does not.

### Notes

- The cache operates at the attention-module level. Model-level integration
  (per-layer states, position-aware preamble) is the caller's responsibility.
  GQLSA remains position-agnostic; the caller supplies `start_pos`.
- `forward_step` supports exactly one token per call per batch element.
  Batched or multi-token incremental forward is not supported in v1.0.1.
- CUDA graph capture was prototyped but deferred. It requires `start_pos`
  to flow through the forward path as a device tensor rather than a Python
  int, which would touch `forward_step` and `_retrieve_one`. Kept out of
  v1.0.1 to keep the diff minimal and the correctness guarantee tight.

---

## [1.0.0] — 2026-09-18

Initial public release. Accompanies the preprint:

> **Grouped-Query Latent Sparse Attention: Compute Only Where It Matters**
> Fardin Sabid — Independent Researcher
> DOI: [10.5281/zenodo.22818585](https://doi.org/10.5281/zenodo.22818585)

### Added

- **GQLSA attention module** (`attention/gqlsa.py`)
  - Latent KV compression (MLA-style) with configurable dimension `d_c`
  - Grouped-query head sharing (GQA-style) with `h` query heads and `g` KV groups
  - Content-retrieved block sparsity with a fully vectorized inverted index
  - Local positional window of `local_window` tokens
  - Global content-retrieved blocks via random-projection LSH
  - Anchor exclusion to prevent trivial self-match in retrieval
  - Cross-block and same-block causal masking

- **Correctness and causality tests** (`tests/`)
  - Shape and numerical validity
  - Multiple sequence lengths (16, 32, 64, 128, 256, 512)
  - Grouped-heads validation
  - Content-aware retrieval verification
  - Causality at multiple levels, including through the retrieval path
  - Verification that selected global block indices are causally valid
  - 15 tests total, all passing

- **Examples** (`examples/`)
  - `basic_usage.py` — end-to-end forward pass with causality and content-dependence checks
  - `inference_demo.py` — minimal transformer block using GQLSA

- **Unified benchmark** (`benchmarks/benchmarkv.py`)
  - Causality check across MHA, GQA, MLA, and GQLSA
  - Speed benchmark at sequence lengths 512–4096
  - Peak memory benchmark at the same sequence lengths
  - Quality benchmark on WikiText-2 (char-level language modeling)

- **Paper** (`paper/`)
  - `GQLSA.pdf` — published preprint
  - `paper/figures/*.png` — all figures referenced in the paper and README

- **Repository metadata**
  - `README.md` — overview, results, usage, benchmarking, and citation
  - `LICENSE` — Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  - `CITATION.cff` — machine-readable citation metadata
  - `requirements.txt` — Python dependencies

### Measured results (Tesla T4, 4096 tokens, vs. MHA)

| Metric | Value |
|---|---|
| Attention FLOPs | 4.00× reduction |
| Wall-clock latency | 3.68× speedup |
| Peak memory | 2.03× reduction |
| KV cache per token per layer | 15.5× reduction |
| Attention parameters | 2.8× fewer |
| WikiText-2 perplexity | 13.20 (rank 2 of 4) |

### Notes

- Quality numbers are reported as an initial signal from a char-level language model on WikiText-2 and are consistent with a small-scale benchmark.
- Single hardware platform (Tesla T4) — cross-hardware behavior is not characterized in this release.
- Retrieval operates on batch-averaged block means during training; at inference (batch size 1), retrieval is per-sequence.
- A C++/CUDA kernel implementation is planned for a future release.

---

## [Unreleased]

### Planned

- C++/CUDA fused kernel for the retrieval and sparse attention path
- Per-sequence retrieval option as an alternative to batch-averaged retrieval
- Ablation over `block_size`, `local_window`, `top_k`, and retrieval parameters
- Recall@G_b measurement for retrieval quality
- Cross-hardware validation (A100, H100, consumer GPUs)
- Long-context downstream task evaluation (RULER, LongBench)
- CUDA graph capture for `forward_step` (requires `start_pos` as device tensor)

---

[1.0.1]: https://github.com/fardinsabid/gqlsa/releases/tag/v1.0.1
[1.0.0]: https://github.com/fardinsabid/gqlsa/releases/tag/v1.0.0
[Unreleased]: https://github.com/fardinsabid/gqlsa/compare/v1.0.1...HEAD