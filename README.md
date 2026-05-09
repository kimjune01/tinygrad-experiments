# tinygrad experiments

Investigation artifacts from tinygrad kernel optimization work. Hypothesis graphs, benchmarks, evidence. The reasoning scaffolding behind upstream PRs — none of this belongs in tinygrad itself.

## Experiments

| Directory | Investigation | Key artifact |
|---|---|---|
| `beam/` | BEAM heuristic replacement — theory-guided search vs grid search | [HYPOTHESIS_GRAPH.md](beam/HYPOTHESIS_GRAPH.md) |
| `linecount/` | Onnx proto parser refactor — line count reduction | [HYPOTHESIS_GRAPH.md](linecount/HYPOTHESIS_GRAPH.md) |
| `matvec/` | Matvec loop ordering — Metal kernel analysis | [HYPOTHESIS_GRAPH.md](matvec/HYPOTHESIS_GRAPH.md) |
| `or/` | Operations research — scheduling overhead, instruction scheduling, bank conflicts | [hypothesis-graph.md](or/hypothesis-graph.md) |
| `pareto-frontier/` | CPython dispatch overhead — JIT compilation, trie dispatch, Cython rewrite | [HYPOTHESIS_GRAPH.md](pareto-frontier/HYPOTHESIS_GRAPH.md) |
| `realize/` | Realize performance — Windows CUDA replication | [HYPOTHESIS_GRAPH.md](realize/HYPOTHESIS_GRAPH.md) |
