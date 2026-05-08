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

---

## Cycle 1: BEAM cache analysis (2026-05-08)

### H0.1: Does BEAM find schedules the heuristic misses?

**Status:** INCONCLUSIVE — need a workload where heuristic is wrong

**Perturbation:** Generated BEAM=4 cache for 7 workloads (matmul_256, matmul_1024, conv2d, matvec, reduce_sum, softmax, layernorm). 59 entries cached. Compared BEAM-cached kernels vs IGNORE_BEAM_CACHE=1 heuristic output.

**Evidence:**
- 59 BEAM results, mostly simple: 30 kernels with 1 opt, 21 with 2, 7 with 3.
- Strategy distribution: LOCAL-only (19x), UPCAST-only (15x), GROUP (8x), LOCAL+UPCAST (6x), GROUP+UNROLL (4x), TC (3x).
- On matvec, matmul_256, and all tested workloads: **BEAM produces identical kernel shapes to the heuristic.** Kernel names and timings match. BEAM spent 1.8–8.8 seconds of search to arrive at the same schedule the heuristic produces in milliseconds.

**Trajectory:** CONVERGENT to null. On these workloads, BEAM validates the heuristic but doesn't improve on it. The heuristic is already correct for standard shapes.

**Implication for H0:** The abduction engine can only outperform BEAM on workloads where the heuristic is WRONG. We need to find those workloads. The OR graph's bench_perceive identified candidates: `gemm_256: 0.65x`, `permute: 0.63x` (BEAM improved over heuristic). Standard shapes (matmul, conv2d, softmax) are not the right test bed.

**What would advance H0:** A kernel where:
1. The heuristic produces a suboptimal schedule (measurable gap vs BEAM)
2. The reason for the gap is structural (identifiable theory, not random search luck)
3. The theory transfers to a different kernel shape

Without (1), there's nothing for the abduction engine to improve. The investigation is blocked on finding the right workload.

### Next edge

The OR graph identified two candidates from bench_perceive where BEAM helped:
- `gemm_256`: BEAM found 0.65x (35% faster than heuristic)
- `permute`: BEAM found 0.63x (37% faster than heuristic)

These were measured with the PERCEIVE benchmark suite. Reproduce them to get a kernel where the heuristic is wrong, then form a theory about WHY.

### H0.2: Non-square GEMM reduction — BEAM finds TC, heuristic doesn't

**Status:** CONFIRMED — first case where BEAM outperforms heuristic

**Perturbation:** `Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096))` — a tall-skinny × square matmul. The reduction kernel (the actual GEMM, not the random number generation) was the target.

**Evidence:**
```
Heuristic: r_32_32_4_2_2_4_512     → 3449us
BEAM=4:    r_2_256_32_2_2_128_4    → 1902us   (-44.8%)
```

BEAM applied: `TC(axis=0, arg=(-1,0,1)) + UPCAST(axis=1, arg=2) + UNROLL(axis=0, arg=4)`.

The heuristic produced a schedule WITHOUT tensor cores. BEAM found that TC + wider unrolling transforms the reduction from a scalar loop into WMMA-accelerated computation. The heuristic doesn't try TC for non-square reductions — it applies TC only for shapes it recognizes as GEMM-like.

**Theory (first abduction):** The heuristic's TC heuristic is shape-gated: it only enables tensor cores for shapes that match its GEMM template. Non-square shapes (16×4096 × 4096×4096) don't match, so TC is never tried. But the underlying hardware operation (WMMA 8×8×8) works regardless of the outer shape — the inner dimension (4096) is large enough for TC to help.

**Theory prediction:** Any matmul where:
1. One dimension is small (< 64) AND
2. The reduction dimension is large (> 1024) AND
3. The heuristic doesn't enable TC

...should benefit from TC + UNROLL. This is testable on other shapes without BEAM.

**Kill condition for the theory:** If the speedup is from UNROLL alone (not TC), the theory is wrong — TC isn't the key, just wider parallelism. Test: run with UNROLL only, no TC.

**Trajectory:** DIVERGENT for H0. BEAM found a 44.8% improvement the heuristic missed. The improvement is attributable to a specific transformation (TC) that the heuristic doesn't try for this shape class. This is exactly the case where a theory ("non-square reductions benefit from TC") would have predicted the right transformation without search.

### H0.3: Theory transfer test — does "tall-skinny needs TC" predict?

**Status:** CONFIRMED — theory transfers to unseen shape

**Perturbation:** Applied the H0.2 theory ("tall-skinny matmuls with large reduction dim need TC") to a DIFFERENT shape: `Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048))`. The theory predicts: heuristic will skip TC, BEAM will find TC, improvement will be large.

**Evidence:**
```
8×2048 × 2048×2048 reduction kernel:
  Heuristic: r_16_32_4_2_4_256  → 1281us  (no TC)
  BEAM=4:    r_256_32_2_64_4    →  528us  (TC + UNROLL)
  Delta: -58.8%
```

All three predictions confirmed:
1. Heuristic skipped TC ✓
2. BEAM found TC ✓ (`TC(axis=0, arg=(-1,2,1)) + UNROLL(axis=0, arg=4)`)
3. Large improvement ✓ (-58.8%, even larger than the 44.8% on 16×4096)

**This is H0's proof of concept.** The theory formed from one kernel (16×4096) correctly predicted the winning transformation on a different kernel (8×2048) without any search. An abduction engine that encoded this theory would have tested 1 configuration (apply TC) instead of BEAM's full combinatorial search.

**Theory refinement:** The theory is now: "For matmuls where `min(M,N) < 64` and `K > 1024`, the heuristic's TC gate is overly conservative. TC should be tried regardless of outer shape when the inner dimension provides enough work for WMMA tiles."

**Next edge:** Can this theory be encoded as a one-line heuristic fix? If `min(M,N) < 64 and K > 1024: try TC` — does it match BEAM's result? This would close the loop: observation → theory → code change.

### H0.4: Theory refinement — TC was already enabled, post-TC opts are wrong

**Status:** REFRAMED — the bottleneck is post-TC optimization, not TC gating

**Perturbation:** Inspected what the heuristic actually produces for the 16×4096 tall-skinny matmul. Monkeypatched `hand_coded_optimizations` to capture before/after state.

**Evidence:**
```
Heuristic: TC + UPCAST(axis=0, arg=2) + UPCAST(axis=0, arg=4) + LOCAL(axis=0, arg=4)
BEAM:      TC + UPCAST(axis=1, arg=2) + UNROLL(axis=0, arg=4)
```

The heuristic ALREADY enables TC for this kernel (1 reduce axis, gate passes). The -44.8% gap is NOT from missing TC — it's from the post-TC opts. The heuristic UPCASTs axis 0 twice (the M dimension, which is only 16). BEAM UPCASTs axis 1 (the N dimension, 4096) and UNROLLs axis 0 (the K dimension).

**Revised theory:** For tall-skinny matmuls, the heuristic's post-TC strategy (UPCAST M, then UPCAST M again, then LOCAL M) wastes register budget on the small dimension. BEAM discovers that UPCAST N + UNROLL K is better — it parallelizes across the large dimensions instead.

**Kill condition for revised theory:** If UPCAST(axis=1) + UNROLL(axis=0) hurts on square matmuls, the theory is shape-specific. The heuristic's post-TC opts should be shape-aware: UPCAST the LARGE dimension, not always axis 0.

**Trajectory:** OSCILLATORY. The original theory ("TC is missing") was wrong. The revised theory ("post-TC opts target wrong axis") is more specific and testable. Split into sub-hypotheses.

**This is exactly the kill-condition-generates-edge pattern.** The wrong theory (H0.2: "heuristic skips TC") was killed by evidence (TC is already applied). The kill generated a new, more precise theory (H0.4: "post-TC opts target wrong axis"). The methodology worked — the theory got refined, not abandoned.

### H0.5: BEAM-style post-TC opts vs heuristic — kernel timing comparison

**Status:** ★ CONFIRMED — theory-derived fix matches or exceeds BEAM

**Perturbation:** Monkeypatched the heuristic to use `UPCAST(axis=1, arg=2) + UNROLL(axis=0, arg=4)` after TC instead of the heuristic's default `UPCAST(axis=0, arg=2) + UPCAST(axis=0, arg=4) + LOCAL(axis=0, arg=4)`.

**Evidence:**

| Shape | Heuristic kernel | Time | BEAM-style kernel | Time | Delta |
|---|---|---|---|---|---|
| 16×4096 × 4096×4096 | r_32_32_4_2_2_4_512 | 3362us | r_2_256_32_2_2_128_4 | 1912us | **-43.1%** |
| 8×2048 × 2048×2048 | r_16_32_4_2_4_256 | 1281us | r_256_32_2_64_4 | 528us | **-58.8%** |
| 256×256 × 256×256 | r_8_2_32_4_2_4_4_32 | 313us | r_32_16_32_2_2_8_4 | 154us | **-50.8%** |

No regressions — square matmul also improves. The heuristic's post-TC opts are universally worse.

**Root cause:** The heuristic (lines 39-45) UPCASTs and LOCALs axis 0 after TC. This is the M dimension. For tall-skinny matmuls M is small (16) — upcasting a small dimension wastes register budget. BEAM finds that UPCASTing axis 1 (N, the large dimension) and UNROLLing axis 0 (K, the reduction dimension) gives the GPU more work per thread and better memory access patterns.

But even for SQUARE matmuls, the BEAM-style opts are better: 154us vs 313us (-50.8%). The heuristic's axis-0 bias is wrong in general, not just for tall-skinny.

**Trajectory:** DIVERGENT for. The fix helps across all tested shapes. Ready for Phase 7 (regression check on full test suite).

**The fix (2 lines):** Replace the post-TC UPCAST/LOCAL on lines 39-45 with UPCAST(axis=1) + UNROLL(axis=0).

### H0.6: AMD gfx1201 regression — UNROLL(0,4) produces incorrect WMMA results

**Status:** CONFIRMED — the fix is AMD-unsafe

**Evidence:** PR #16104 CI failure: `test_gemm_fp16` on AMD gfx1201, 64×64 matmul. 50% of elements wrong, max error 399.5. First 16 columns correct, columns 16+ wildly wrong — the UNROLL misaligns WMMA tile boundaries on this GPU.

Passes on: Metal, CUDA, AMD gfx1100, AMD gfx950, CPU, WebGPU, DSP.
Fails on: AMD gfx1201 (both amd and amdllvm backends).

**Decomposition:** Dropping LOCAL alone (keeping original UPCASTs, no UNROLL) produces 3351us vs 3376us — no improvement. The entire -43% to -51% came from UNROLL(0,4). The UPCAST axis order doesn't matter.

**Trajectory:** OSCILLATORY. The fix helps Metal but breaks AMD gfx1201. Classic oscillatory — helps one target, hurts another. Split into:
- H0.6a: Why does UNROLL(0,4) misalign WMMA on gfx1201 specifically? (gfx1100 and gfx950 pass)
- H0.6b: Is there a safe UNROLL factor (2 instead of 4)? Or a different post-TC strategy that helps both?

**PR #16104 must be reverted or made backend-conditional.** The investigation re-enters Phase 2 with the oscillatory result.
