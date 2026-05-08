# tinygrad abduction engine

## What this is

An alternative kernel schedule optimizer for tinygrad. Replaces BEAM's grid search with theory-guided optimization using abduction (observe → theorize → test).

## Key files

- `ABDUCTION.md` — what abduction means in this project
- `BASELINE.md` — measured BEAM search behavior (the thing we're beating)
- `BEAM_CONTEXT.md` — tinygrad project's attitude toward BEAM from issues/PRs
- `HYPOTHESIS_GRAPH.md` — root hypothesis and dependencies with falsification conditions
- `bench_beam_baseline.py` — instrumented BEAM benchmark
- `baseline_results.json` — raw baseline data

## tinygrad source

The tinygrad repo is cloned at `~/Documents/tinygrad-for-abduction`. Do not use `~/Documents/tinygrad` — that directory has a parallel session.

BEAM search lives in `tinygrad/codegen/opt/search.py`. Heuristics live in `tinygrad/codegen/opt/heuristic.py`. The optimizer entry point is `tinygrad/codegen/opt/postrange.py`.

## Ambiguity heuristic

Lines of code. Less is better. The engine should be smaller than the code it replaces.

## Methodology

Abduction: perturb, observe, classify the trajectory shape, follow the edge. See `ABDUCTION.md` for the full explanation.

## Benchmarking

```bash
BEAM=2 python3 bench_beam_baseline.py              # full suite
WORKLOAD=gemm_256 python3 bench_beam_baseline.py    # single workload
```

Results go to `baseline_results.json`.
