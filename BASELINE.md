# BEAM Baseline

## Setup

- **Platform:** Apple Silicon, Metal backend
- **tinygrad:** HEAD as of 2026-05-07
- **BEAM=2**, IGNORE_BEAM_CACHE=1, CNT=4
- **Action pool:** 193 actions (48 UPCAST, 44 LOCAL, 30 THREAD, 24 GROUPTOP, 15 UNROLL, 12 GROUP, 10 TC, 10 SWAP)

## Results

```
workload       heur      beam     b/h   rounds  candidates  timed  search   yield
gemm_1024     1.88ms    1.71ms   0.91x    13       173       165    3.3s     95%
gemm_256      1.23ms    0.87ms   0.70x    12       141       132    2.2s     94%
add_4096      1.30ms    1.34ms   1.03x    13       114       105    2.5s     92%
mul_sum       2.36ms    1.48ms   0.63x    16       180       165    3.1s     92%
relu_4096     1.40ms    0.99ms   0.71x     9       109       100    1.8s     92%
exp_2048      1.06ms    1.13ms   1.07x     9       106        97    2.0s     92%
sum_4096      1.84ms    1.46ms   0.79x    15       253       229    3.2s     91%
permute       1.02ms    1.00ms   0.97x     9       152       148    1.9s     97%
softmax       1.41ms    1.49ms   1.05x    15       251       235    3.3s     94%
layernorm     2.07ms    1.61ms   0.78x    14       269       255    3.5s     95%
matvec        1.87ms    1.41ms   0.76x    15       163       146    4.6s     90%
```

- **b/h**: beam time / heuristic time. Below 1.0 = BEAM helped. Above 1.0 = BEAM regressed.
- **yield**: timed / candidates. How much of the search space reached hardware timing.

## Findings

### 1. No baseline candidate

`beam = [(s, float("inf"))]` — the starting scheduler is assigned time = infinity, never measured. `get_kernel_actions(si, include_0=False)` excludes the identity action. BEAM never knows how fast "do nothing" is.

**Consequence:** BEAM cannot fall back to the heuristic. If every transformation is worse than no transformation, BEAM picks the least-bad transformation and ships a regression. This explains the three regressions: add_4096 (1.03x), exp_2048 (1.07x), softmax (1.05x).

`search.py:129, 150`

### 2. Proxy measurement distorts rankings

`allow_test_size` shrinks the global size to max 65536 threads and scales the time linearly. A kernel that is bandwidth-bound at full size may be compute-bound at 1/16th size — different bottleneck, different optimal schedule.

**Consequence:** BEAM selects schedules optimized for a workload that doesn't match production. The linear scaling assumption breaks for ops where cache behavior, occupancy, or memory coalescing dominate. These are exactly the ops where BEAM regresses.

`search.py:28-36, 42-44`

### 3. No pruning — 92-97% yield

Almost everything that compiles gets timed on hardware. The only filters are:
- Deduplication via `seen_libs` (same compiled binary)
- 1000x compute ops ceiling (pathological blowups)

No structural reasoning eliminates candidates before the expensive compile+time step. BEAM doesn't ask "is this kernel memory-bound?" before trying 48 UPCAST variants that only help compute-bound kernels.

`search.py:150-171`

### 4. Shallow search, plateau exit

Exit condition: `beam[0][1] - opts[0][1] < min_progress` (default: 10ns). BEAM exits when one more round doesn't improve by at least 10ns. This means "I plateaued" — not "I found the optimum."

Compute kernels converge in 2-4 rounds. The search never asks whether the plateau is a local minimum or a global one. It can't — that would require a theory about the optimization landscape, which requires abduction.

`search.py:182`

### 5. No triage across kernels

Each workload produces 5-7 kernels. Of these, 3-4 are trivial (copy, setup, constant init — 7us, 2 candidates, 1 round). BEAM spends the same per-kernel startup cost on a 7us copy as on the 1ms compute kernel that dominates runtime.

No mechanism identifies which kernels matter before searching. The total search budget (2-5s wall time) is spread evenly, not allocated by impact.

### 6. Early stop discards diagnostic information

`if early_stop is not None and early_stop < min(tms): break` — if a candidate is 3x slower than the current best, timing stops after one sample. The data that could explain *why* it's slow (high variance? warming up? consistent?) is never collected.

BEAM observes a single number (min time) and discards it if it's bad. An abduction engine would want to know: is this candidate slow because of memory stalls, register spilling, bank conflicts, or low occupancy? Even a bad candidate's failure mode is evidence about the kernel's structure.

`search.py:55-56`

## What BEAM actually is

BEAM is grid search over a hyperparameter space with beam pruning. The 193 actions are a hardcoded grid of tile sizes and parallelism knobs — not program rewrites, not compiler transformations. BEAM applies them one at a time, times each combination on hardware, and keeps the top-k. It's iterative coordinate descent with beam width > 1.

It is not a good grid search. A competent grid search would:
- Include the origin (no transformation) as a candidate
- Use the actual workload, not a proxy at 1/16th scale
- Allocate budget proportional to kernel impact on total runtime
- Distinguish plateau from local minimum before stopping

BEAM does none of these. A random search with a baseline fallback would avoid the regressions.

## What this means for the hypothesis

The six findings decompose into two categories:

**Search errors** (findings 1, 2): BEAM can produce worse results than doing nothing because it has no baseline and uses proxy measurements. These are bugs fixable without abduction.

**Missing abduction** (findings 3, 4, 5, 6): BEAM wastes trials because it can't reason about what it's seeing. It can't prune (3), can't distinguish plateau from optimum (4), can't triage kernels by importance (5), and discards the diagnostic signal in bad results (6). These require the Observation → Theory edge that BEAM doesn't have.

The bar for abduction is low. The comparison isn't "abduction vs. a sophisticated compiler optimizer." It's "abduction vs. grid search that can't find the origin." Any technique that classifies a kernel as memory-bound vs. compute-bound before searching eliminates half the grid on the first step — something grid search can never do regardless of budget.
