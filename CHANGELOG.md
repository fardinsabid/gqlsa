# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

---

[1.0.0]: https://github.com/fardinsabid/gqlsa/releases/tag/v1.0.0
[Unreleased]: https://github.com/fardinsabid/gqlsa/compare/v1.0.0...HEAD