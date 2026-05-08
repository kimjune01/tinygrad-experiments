# tinygrad abduction engine

Replace BEAM's grid search with theory-guided optimization. Find better kernel schedules in fewer trials by reasoning about *why* a kernel is slow before deciding what to try next.

## Goal

Beat BEAM on tinygrad's benchmark suite: better final schedules, fewer trials, zero regressions. Then obsolete the hand-coded heuristics.

## Context

tinygrad optimizes GPU kernels with two systems that both fall short:

**Heuristics** (`hand_coded_optimizations`) encode human theories about kernel structure — "matvec needs different tiling than square GEMM." They're fast but frozen. Every new kernel shape requires a new rule. When they misfire, performance degrades and BEAM can't compensate.

**BEAM** (`codegen/opt/search.py`) is grid search over 193 hyperparameter combinations with beam pruning. It applies transformations, times each on hardware, keeps the top-k. It has no model of what it's optimizing. It regresses on 3/11 benchmark workloads because it can't fall back to baseline. It doesn't know why anything is fast.

Both exist because neither is sufficient alone. The abduction engine replaces both by forming theories from measurement and testing only the deductions.

See [BEAM_CONTEXT.md](BEAM_CONTEXT.md) for the project's attitude toward BEAM, [BASELINE.md](BASELINE.md) for measured search behavior, and [ABDUCTION.md](ABDUCTION.md) for what abduction means here.

## Setup

Clone tinygrad to a separate directory (there is a parallel session using `~/Documents/tinygrad`):

```bash
git clone https://github.com/tinygrad/tinygrad.git ~/Documents/tinygrad-for-abduction
cd ~/Documents/tinygrad-for-abduction
python3 -m pip install -e .
```

Run the baseline benchmark:

```bash
cd ~/Documents/tinygrad-abduction-engine
BEAM=2 python3 bench_beam_baseline.py
```

## Ambiguity heuristic

Lines of code. Less is better. The abduction engine should be smaller than the heuristic it replaces, not larger. If the engine needs more code than `hand_coded_optimizations` + `beam_search` combined, the abstraction is wrong.

## Methodology

Abduction itself. Perturb, observe, classify the trajectory, follow the edge.

1. **Observe** what BEAM observes (kernel timings, hardware counters where available)
2. **Abduce** a theory about the bottleneck (memory-bound, compute-bound, occupancy-limited)
3. **Deduce** which transformations the theory predicts will help
4. **Experiment** with only those transformations
5. **Classify** the result (convergent, divergent, oscillatory, chaotic)
6. **Repeat** until the hypothesis graph converges

Each cycle narrows the search. The branching factor at each step is 2-3 targeted experiments, not 193 blind ones.

See [HYPOTHESIS_GRAPH.md](HYPOTHESIS_GRAPH.md) for the full hypothesis tree and falsification conditions.

## Files

| File | What |
|------|------|
| [HYPOTHESIS_GRAPH.md](HYPOTHESIS_GRAPH.md) | Root hypothesis (H0) and dependencies, with falsification conditions |
| [ABDUCTION.md](ABDUCTION.md) | What abduction means in context — for future Claude instances |
| [BASELINE.md](BASELINE.md) | BEAM's measured search behavior: 6 findings, line references |
| [BEAM_CONTEXT.md](BEAM_CONTEXT.md) | tinygrad's attitude toward BEAM from issues/PRs |
| [bench_beam_baseline.py](bench_beam_baseline.py) | Instrumented BEAM benchmark |
| [baseline_results.json](baseline_results.json) | Raw baseline data |
