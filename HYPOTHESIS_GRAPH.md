# Hypothesis Graph: Abduction Engine for BEAM

## Starting observation

BEAM search perturbs (applies schedule transformations) and observes (times kernels on hardware). It loops Experiment → Observation without ever forming a theory. The branching factor is the full combinatorial space of `(TC × UPCAST × UNROLL × LOCAL × GROUP)` because every perturbation is equally plausible.

A system that closes the triangle — Observation → Theory → Experiment — would collapse the branching factor before timing anything.

```
         Theory
        ╱      ╲
abduction    deduction
     ╱            ╲
Observation ——→ Experiment
           induction
```

BEAM today runs only the bottom edge.

---

## H0: Abduction outperforms BEAM

**Status:** ALIVE

**Claim:** A system that forms theories about *why* a kernel is slow and tests them will find better schedules in fewer trials than BEAM's black-box search.

BEAM treats schedule optimization as enumeration: generate candidates, time them, keep the best. It cannot separate figure (which transformation caused a speedup) from ground (everything else that changed). Without this separation, it cannot form theories, so every perturbation is equally plausible and the branching factor is the full combinatorial space.

**Evidence:**
- BEAM's search is in `tinygrad/codegen/opt/search.py`. It generates candidate schedules, compiles and times each, keeps the top-k, repeats. No causal attribution step between rounds.
- The search space is the cross-product of optimizer knobs. No structure is exploited to reduce it.
- BEAM is inconsistent on the PERCEIVE benchmark: improves some ops (gemm_256: 0.65x, permute: 0.63x), regresses others (relu: 1.63x). A system with theories would not regress — it would at least fall back to the heuristic baseline.

**What would disprove this:**
- If BEAM's branching factor is already low enough that exhaustive search converges in reasonable time for all kernel shapes → the problem is search budget, not missing abduction.
- If adding more BEAM rounds (higher `amt`) closes all gaps → brute force suffices.

---

## H2: Observation → Theory requires causal attribution, not just ranking

**Status:** ALIVE

**Claim:** The abductive step is: given two schedules with different runtimes, identify *which transformation* caused the difference. This is a causal inference problem (Pearl's do-calculus), not a sorting problem.

BEAM currently ranks schedules by runtime. Ranking tells you *what* is fast, not *why*. The theory needs to be about the transformation, not the schedule.

**Analogy to the modes-of-reason table:**

| Mode | BEAM today | Abduction engine |
|------|-----------|-----------------|
| Observation | Wall-clock time per schedule | Wall-clock time + hardware counters (cache misses, occupancy, stalls) |
| Theory | (missing) | "This kernel is memory-bound; UPCAST doesn't help because ALU is not the bottleneck" |
| Experiment | Try all knob combinations | Test only the deductions from the current theory |

**What would disprove this:**
- If schedule performance is dominated by interaction effects (no single transformation is attributable) → causal attribution is intractable and ensemble ranking is the best you can do.

---

## H3: Theory → Experiment collapses the search space

**Status:** ALIVE (contingent on H2)

**Claim:** A theory about *why* a kernel is slow produces a small set of targeted experiments instead of the full combinatorial space.

Example theory chain:
1. Observe: kernel has low ALU utilization, high memory stall cycles
2. Abduce: kernel is memory-bound
3. Deduce: increasing UPCAST (more ALU work per thread) won't help; increasing LOCAL (more threads to hide latency) might; adding shared memory tiling (GROUP) to reduce global memory traffic should help
4. Experiment: test only the LOCAL and GROUP knobs, skip UPCAST and TC
5. Observe: GROUP=2 improved throughput 1.4x, LOCAL change was neutral
6. Abduce: the bottleneck was global memory bandwidth, not latency
7. Deduce: further GROUP increases should help until shared memory capacity is saturated
8. Experiment: test GROUP=4

Each cycle narrows. The branching factor at each step is 2-3, not the full knob space.

**What would disprove this:**
- If the theory is wrong often enough that the pruned search misses the optimum more than brute force does → abduction adds overhead without improving outcomes.
- If the cost of forming a theory (hardware counter reads, causal analysis) exceeds the cost saved by pruning → net negative.

---

## H4: The memo table should cache theories, not just winning schedules

**Status:** ALIVE (contingent on H2, H3)

**Claim:** BEAM's disk cache stores `applied_opts` — the winning schedule transformations. An abduction engine should cache the *theory* (e.g., "memory-bound, GROUP helps") alongside the winning schedule.

**Why this matters:**
- When a new kernel arrives with similar structure but different shape, the cached theory transfers. The cached schedule does not — it's shape-specific.
- Theory transfer reduces cold-cold cost: instead of searching from scratch, start from the theory and deduce which experiments to run for the new shape.
- The cache key shifts from `ast.key` (exact match) to structural features of the kernel (memory-bound vs compute-bound, reduction depth, live variable count).

**What would disprove this:**
- If kernel performance is so shape-sensitive that no theory transfers across shapes → caching theories has no advantage over caching schedules.
- If the structural features that determine the theory are already captured by `ast.key` → the current cache is sufficient.

---

## H5: Abduction obsoletes heuristics

**Status:** ALIVE (contingent on H0)

**Claim:** tinygrad maintains two optimization paths: hand-coded heuristics (`hand_coded_optimizations`) and BEAM search. They exist because neither is sufficient alone. Heuristics encode human theories about kernel structure ("matvec needs different tiling than square GEMM") but can't adapt. BEAM adapts but can't reason. Every new kernel shape that the heuristic doesn't cover requires a human to write a new rule.

An abduction engine that forms theories from observations does what heuristics do (classify the kernel, select a strategy) but derives the theory from measurement instead of hardcoding it. If it works, the heuristic codepath is dead code.

**Evidence from the repo:**
- The heuristic has shape-specific branches that accumulate over time: CPU matvec (#15599, #15616), group heuristic (#13677), tensor core heuristic (#3197). Each is a manually encoded theory about one kernel shape.
- When the heuristic misapplies, BEAM can't compensate: matvec lift = 0.8x (heuristic *hurts*), mul_sum lift = 0.5x. The heuristic assumes a structure the kernel doesn't have, and BEAM has no mechanism to override it.
- geohot rejects heuristic PRs that are too shape-specific (#15599: "looks vaguely AI", #15616: closed). He wants general solutions, not more special cases.

**What would disprove this:**
- If the heuristic's theories are cheap to maintain and the abduction engine's theories are expensive to derive → heuristics win on amortized cost even if abduction is more general.
- If kernel shapes cluster into a small fixed set (GEMM, elementwise, reduction, scan) and the heuristic already covers all of them → no new theories are needed, and the maintenance cost of heuristics is bounded.

---

## Dependencies

```
H0 (abduction outperforms BEAM)
 ├→ H2 (causal attribution is the mechanism)
 │   └→ H3 (theories collapse search)
 │       └→ H4 (cache theories, not schedules)
 ├→ H5 (abduction obsoletes heuristics)
 └→ falsified if BEAM + more budget matches abduction's quality
```

H0 is the root claim. If BEAM with sufficient budget matches the abduction engine's schedule quality, the rest is unnecessary — the problem was just search budget, not missing reasoning. H5 is a corollary: if abduction can form the theories that heuristics encode, the heuristic codepath is redundant.

## Open questions

1. **What hardware counters are available on Metal?** GPU profiling on Apple Silicon is limited compared to NVIDIA (nsight) or AMD (rocprof). If the observation channel is too narrow (wall-clock only), the abductive step may not have enough signal.
2. **What does BEAM's actual search tree look like?** Before building an abduction engine, instrument BEAM to log the full search tree: which candidates were generated, which survived, what the timing distribution looks like. This is the baseline measurement.
3. **Is there prior art on theory-guided compiler autotuning?** The ML-for-compilers literature (TVM/Ansor, Halide autoscheduler) uses learned cost models, which are implicit theories. How do they compare to explicit causal theories?
