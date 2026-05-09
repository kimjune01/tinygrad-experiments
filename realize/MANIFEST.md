# Integration Manifest

| Artifact | Location | What it hosts |
|----------|----------|---------------|
| Experiment repo | `~/Documents/tinygrad-realize-experiment/` | Prework, validation, benchmarks |
| tinygrad fork | `~/Documents/tinygrad/` | Target codebase (origin: tinygrad/tinygrad, fork: kimjune01/tinygrad) |
| Hypothesis graph | `~/Documents/tinygrad/HYPOTHESIS_GRAPH.md` | Investigation provenance (H₁₂) |
| Prior investigations | `~/Documents/tinygrad-matvec-experiment/` | Matvec codegen (H₁ₘ) |
| | `~/Documents/tinygrad-linecount-experiment/` | Line budget (onnx refactor) |

## Experiment repo files

| File | Purpose | Dependencies |
|------|---------|-------------|
| `shapes.py` | Test matrix: quant types × model sizes | None |
| `extract.py` | Confirms lazy dequant chains exist in target | tinygrad, shapes.py |
| `reference.py` | Baseline: realize=0 vs realize=1 performance | tinygrad, shapes.py |
| `propose.py` | Documents the one-character fix | None |
| `compat.py` | Output equivalence: lazy vs realized weights | tinygrad, shapes.py |
| `bench.py` | Performance delta measurement | tinygrad, shapes.py |
| `validate.py` | Runs extract + compat + bench | All above |
| `TRANSFORM.md` | Maps fix onto target codebase | — |
| `MANIFEST.md` | This file | — |
| `RESULTS_WINDOWS_CUDA.md` | Cross-platform bench replication (RTX 5000 Ada, Win11) + Windows setup gotchas | — |

## tinygrad file to touch

| File | Change | Risk |
|------|--------|------|
| `tinygrad/llm/model.py` line 323 | `getenv("REALIZE", 0)` → `getenv("REALIZE", 1)` | Low — existing code path |

## Remotes

| Repo | Remote | Branch |
|------|--------|--------|
| tinygrad | origin: `tinygrad/tinygrad` | master |
| tinygrad | fork: `kimjune01/tinygrad` | (realize-default branch TBD) |
