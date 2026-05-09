# Integration Manifest

| Artifact | Location | What it hosts |
|----------|----------|---------------|
| Experiment repo | `~/documents/tinygrad-matvec-experiment/` | Prototype, validation, prework |
| tinygrad fork | `~/documents/tinygrad/` | Target codebase (origin: tinygrad/tinygrad, fork: kimjune01/tinygrad) |
| Hypothesis graph | `~/documents/tinygrad-matvec-experiment/HYPOTHESIS_GRAPH.md` | Investigation provenance |
| Prior investigation | `~/documents/tinygrad/HYPOTHESIS_GRAPH.md` | Op-level findings (predecessor) |
| Dumped Metal kernels | `~/documents/tinygrad-matvec-experiment/kernels/` | Generated Metal shaders from extract.py |

## Experiment repo files

| File | Purpose | Dependencies |
|------|---------|-------------|
| `shapes.py` | Test matrix: 14 shapes (matvec, GEMM, edge cases) | None |
| `reference.py` | numpy/BLAS ground truth: bandwidth + optimal inner axis | numpy, shapes.py |
| `propose.py` | Stride-cost function: derives loop ordering from strides | numpy, shapes.py |
| `validate.py` | Checks propose vs reference for all shapes | propose.py, reference.py |
| `extract.py` | Dumps tinygrad's actual Metal kernel + loop ordering | tinygrad, shapes.py |
| `compat.py` | Proves loop reordering doesn't change numerical output | numpy, shapes.py |
| `TRANSFORM.md` | Transformation design: how propose.py maps to rangeify.py | — |
| `HYPOTHESIS_GRAPH.md` | Full investigation with references | — |
| `MANIFEST.md` | This file | — |

## tinygrad files to touch (Option A from TRANSFORM.md)

| File | Change | Risk |
|------|--------|------|
| `tinygrad/schedule/rangeify.py` | Stride-aware range ordering in `add_ranges_to_store` | High — affects all kernels |
| `tinygrad/codegen/opt/heuristic.py` | (Option B fallback) Matvec-specific range swap | Low — only matvec path |

## Remotes

| Repo | Remote | Branch |
|------|--------|--------|
| tinygrad | origin: `tinygrad/tinygrad` | master (shallow clone, 1 commit) |
| tinygrad | fork: `kimjune01/tinygrad` | (working branch TBD) |
