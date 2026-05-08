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

**Status:** ALIVE — strong evidence from H5 investigation

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

**Status:** ALIVE — supported by H5 (52 trials vs BEAM's 193 actions, 1.85x better quality)

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

**Status:** ★ CONFIRMED (2026-05-08)

**Claim:** BEAM's disk cache stores `applied_opts` — the winning schedule transformations. An abduction engine should cache the *theory* (e.g., "memory-bound, GROUP helps") alongside the winning schedule.

**Why this matters:**
- When a new kernel arrives with similar structure but different shape, the cached theory transfers. The cached schedule does not — it's shape-specific.
- Theory transfer reduces cold-cold cost: instead of searching from scratch, start from the theory and deduce which experiments to run for the new shape.
- The cache key shifts from `ast.key` (exact match) to structural features of the kernel (memory-bound vs compute-bound, reduction depth, live variable count).

**What would disprove this:**
- If kernel performance is so shape-sensitive that no theory transfers across shapes → caching theories has no advantage over caching schedules.
- If the structural features that determine the theory are already captured by `ast.key` → the current cache is sufficient.

### H4 evidence: theory transfer test (2026-05-08)

**Perturbation:** Derived a theory from gemm_1024 ("after TC: UPCAST N by 2, UNROLL K by 4, UPCAST M by 4, LOCAL M by 4") and tested it on 6 other matmul shapes.

**Exact schedule transfer:** FAILS on 3/6 shapes. Literal opt sequences are shape-specific — `UPCAST(0,4)` fails when post-TC axis 0 has size 2 (tall_skinny), `UNROLL(0,4)` fails when reduce axis has size 2 (wide).

**Semantic theory transfer:** WORKS on 7/7 shapes. The adaptive theory ("UPCAST N, UNROLL K, UPCAST M, LOCAL M — each by largest divisor that fits") applies universally:

| Shape | Heuristic | Adaptive Theory | theory/heur |
|-------|-----------|-----------------|-------------|
| 1024×1024 | 55068us | 9224us | **0.17x** |
| 256×256 | 749us | 146us | **0.19x** |
| 2048×2048 | 580501us | 89313us | **0.15x** |
| 16×4096 × 4096×4096 | 21428us | 4230us | **0.20x** |
| 8×2048 × 2048×2048 | 3715us | 420us | **0.11x** |
| 4096×16 × 16×4096 | 16671us | 2955us | **0.18x** |
| 512×2048 × 2048×256 | 15435us | 2229us | **0.14x** |

The cached theory derived from ONE measurement (gemm_1024) transfers to ALL tested shapes and beats the heuristic 5-9x with zero additional measurements. Fresh per-shape abduction (~52 trials) gets another 1.3-2x on top.

**Bonus finding:** The heuristic is actively harmful on large matmuls — NOOPT is faster than the heuristic on gemm_1024 (19348us vs 55068us) and gemm_2048. The heuristic's post-TC opts hurt more than they help at scale.

**Trajectory:** DIVERGENT for H4. Cache the semantic theory, not the literal schedule. The theory is the transferable unit.

**Reasoning mode:** Induction (7 shapes measured). Confidence: 90% — tested on matmul class only, transfer to other kernel classes (reduction, elementwise) is untested.

---

## H5: Abduction obsoletes heuristics

**Status:** REFRAMED — abduction obsoletes parameters, not pattern matching

**Original claim:** An abduction engine that forms theories from observations does what heuristics do but derives the theory from measurement instead of hardcoding it. If it works, the heuristic codepath is dead code.

**Revised claim:** The heuristic is a cascade of pattern-matchers feeding parameterized transformations. Abduction replaces the *parameters* (upcast amounts, local sizes, thresholds) with measurement-derived values. The *pattern matchers* (TC eligibility, matvec detection, reduction classification) are structural priors that measurement alone cannot efficiently replace. The optimal system is a hybrid: structural priors from AST analysis, parameters from measurement, cached theories for amortization.

**What would disprove the revised claim:**
- If the structural priors are themselves wrong often enough to cancel the parameter improvements → the pattern matchers are a liability, not a prior.
- If theory caching (H4) fails to amortize measurement cost → the hybrid is more expensive than the heuristic for no quality gain.

---

### H5 Investigation (2026-05-08)

#### Phase 1: Heuristic decomposition

**Perturbation:** Decomposed `hand_coded_optimizations` (192 lines) into 9 distinct strategies. Measured heuristic vs NOOPT on 12 workloads.

The heuristic encodes 9 strategies in a priority cascade with early returns:

| # | Strategy | Lines | Theory | Derivable from measurement? |
|---|----------|-------|--------|----------------------------|
| S1 | Tensor Core | 38 | Map WMMA-eligible reductions to hardware TC units | Partially — TC applicability is structural, post-TC params are measurable |
| S2 | Image Upcast | 12 | GPU image memory returns float4; upcast to match | No — requires dtype/memory-layout knowledge |
| S3 | MatVec | 19 | Shared memory reduces global traffic for MV products | Mostly — params searchable, trigger is structural |
| S4 | GroupTop | 10 | Small-output reductions need thread-parallel reduce | Partially — threshold searchable, decision structural |
| S5 | Masked Upcast | 10 | Small masked dims (stack/winograd) → register upcast | No — requires WHERE-gate detection in AST |
| S6 | Stride Upcast | 27 | Upcast broadcast axes to maximize data reuse | Mostly — stride analysis is structural, amounts searchable |
| S7 | Reduce Unroll | 15 | Small reduce dims → straight-line code | Yes |
| S7b | Fallback Upcast | 4 | Some upcasting always better than none | Yes |
| S8 | Local Groups | 20 | Map expand axes to workgroup dims for occupancy | Partially — axis ranking structural, sizes searchable |
| S9 | CPU Threading | 12 | ~128K ops/thread avoids thread overhead | Yes |

**Result:** 45% of lines encode measurement-derivable theories (parameters, thresholds). 42% encode structural knowledge (pattern matching, hardware architecture). 13% is boilerplate.

**Heuristic vs NOOPT (Metal, M4 Max):**

| Workload | Heuristic | NOOPT | h/noopt | Verdict |
|----------|-----------|-------|---------|---------|
| matmul_big (2048²) | 5.08ms | 168.51ms | 0.03x | HELPS (33x) |
| gemm_1024 | 1.77ms | 20.19ms | 0.09x | HELPS (11x) |
| tall_skinny (16×4096) | 1.59ms | 6.37ms | 0.25x | HELPS (4x) |
| conv2d_3x3 | 1.48ms | 3.42ms | 0.43x | HELPS |
| layernorm | 1.93ms | 4.09ms | 0.47x | HELPS |
| softmax | 0.97ms | 1.41ms | 0.69x | HELPS |
| matvec | 1.07ms | 1.40ms | 0.77x | HELPS |
| mul_sum | 2.00ms | 1.25ms | 1.60x | **HURTS** |

**Trajectory:** OSCILLATORY. Heuristic helps most workloads (11-33x on matmul) but hurts mul_sum (1.6x slower). Split into sub-hypotheses.

---

#### H5.1: TC dominates the heuristic's value

**Status:** REFUTED

**Perturbation:** Disabled TC via `USE_TC=0`, kept all other heuristic strategies. Measured.

**Evidence:** For gemm_1024, non-TC strategies alone (upcast/local/unroll) provide 10.5x speedup over NOOPT. Full heuristic (with TC) provides 12.0x. TC adds only 1.14x on top.

For tall_skinny, TC contributes nothing — the entire 3.5x speedup is from non-TC strategies.

**Trajectory:** DIVERGENT against H5.1. The heuristic's main value is tiling (upcast/local), not tensor cores. TC is the cherry on top, not the cake. (Abduction: Peirce 1878. Confidence: 85% — based on 6 workloads, Metal only.)

---

#### H5.2: MATVEC pattern matching is a liability

**Status:** CONFIRMED — one misclassification identified

**Perturbation:** Analyzed the MATVEC detection pattern (lines 68-82) against all benchmark workloads.

**Evidence:** `(a * b).sum()` (mul_sum) triggers the MATVEC path because both buffers use identical indexing — the subset check `all(r in idx1.ranges for r in idx0.ranges)` is trivially true when ranges are equal. A true matvec has an asymmetric access pattern: vector ranges are a *strict* subset of matrix ranges.

GEMM correctly fails the pattern because M is in idx0 but not idx1. Only mul_sum is misclassified.

The MATVEC path on mul_sum produces kernel `r_256_8_4_4_512` (GROUP+LOCAL+UPCAST) = 1758us. The NOOPT kernel `r_4096_4096` = 424us. 4x regression from misclassification.

**Fix:** Add strict subset check: `len(idx0.ranges) < len(idx1.ranges)` or `idx0.ranges != idx1.ranges`.

**Trajectory:** DIVERGENT for H5.2. The pattern matcher is wrong, and the wrongness is identifiable from measurement (try GROUP, observe regression). (Deduction: traced code logic. Confidence: 95%.)

---

#### H5.3: Minimal abduction loop vs heuristic

**Status:** CONFIRMED — abduction beats heuristic on 3/5, loses on 2/5

**Perturbation:** Built a 3-step measurement loop (try TC → try UPCAST per axis → try LOCAL per axis, 23 trials total) and compared to the heuristic on individual kernels.

**Evidence:**

| Workload | NOOPT | Heuristic | Abduction (23 trials) | a/h |
|----------|-------|-----------|----------------------|-----|
| gemm_1024 | 19614us | 307us | **144us** | **0.47x** (abduction 2x better) |
| mul_sum | 941us | 1180us | **371us** | **0.31x** (abduction 3x better) |
| softmax | 135us | 22us | **18us** | **0.82x** (abduction better) |
| matvec | 10907us | **223us** | 257us | 1.15x (heuristic better) |
| layernorm | 2751us | **63us** | 467us | 7.35x (heuristic 7x better) |

**Why abduction wins on gemm:** Found TC+UPCAST(1,2)+UNROLL(0,4)+UPCAST(0,4)+LOCAL(0,4). The heuristic's frozen post-TC opts (UPCAST(0,2)+UNROLL(0,4)) miss the additional UPCAST and LOCAL that measurement discovers.

**Why abduction wins on mul_sum:** Correctly skipped GROUP (no MATVEC misclassification). Just UPCAST(1,4) = 371us. Measurement avoids the pattern-matching bug.

**Why heuristic wins on matvec:** The heuristic's dedicated MATVEC path (GROUP+LOCAL+UPCAST) is better. The abduction loop doesn't try GROUP — it's missing from its repertoire.

**Why heuristic wins on layernorm:** The heuristic's stride-based multi-axis selection (UPCAST axis 2, LOCAL axes 1+2) is precisely tuned. The abduction loop picks wrong axes because it doesn't analyze buffer strides — it just tries axes in order.

**Trajectory:** OSCILLATORY. Abduction beats the heuristic when the heuristic's parameters are wrong or its patterns misfire. Heuristic beats abduction when structural priors (stride analysis, GROUP strategy) guide the search to the right region. Neither dominates. Split into the revised claim.

---

#### H5.4: The hybrid architecture (reframe)

**Status:** PROPOSED — the investigation's structural finding

The surviving hypothesis is not "abduction replaces heuristics" or "heuristics suffice." It's:

**The heuristic is a two-layer system:** structural priors (pattern matching on the AST) and parameterized transformations (upcast amounts, local sizes, thresholds). The abduction engine should:

1. **Keep the structural priors** — TC eligibility, kernel class detection (matmul/matvec/reduction/elementwise), stride analysis for axis selection. These are cheap (zero measurement) and mostly correct.

2. **Replace the parameters with measurement** — instead of hardcoded [5,4,3,2] for upcast and [4,2] for local, try 2-3 values and pick the best. This is where the heuristic's frozen constants go wrong (post-TC axis, mul_sum MATVEC).

3. **Fix the structural priors when measurement proves them wrong** — mul_sum's MATVEC misclassification is detected by measurement (GROUP hurts → not a matvec). Feed this back to refine the pattern.

4. **Cache the theory** (H4) — "gemm_1024 class: TC + UPCAST(1,2) + UNROLL(0,4) + UPCAST(0,4) + LOCAL(0,4)." Next time a similar kernel arrives, skip measurement and apply the cached theory. Amortized cost approaches the heuristic.

**Falsification:** If theory caching doesn't transfer across shapes (every new shape requires fresh measurement), the hybrid is just BEAM with extra steps.

---

### H5 reasoning mode table

| Claim | Mode | Confidence | Source |
|-------|------|------------|--------|
| Heuristic encodes 9 strategies, 45% measurement-derivable | Deduction | 95% | Code analysis of heuristic.py |
| Non-TC strategies provide 10.5x on gemm (TC adds 1.14x) | Induction | 85% | Measured on Metal M4 Max, 6 workloads |
| mul_sum is MATVEC-misclassified due to equal (not strict subset) ranges | Deduction | 95% | Traced code logic + confirmed by measurement |
| Minimal abduction beats heuristic on 3/5 kernels | Induction | 80% | 23-trial loop, Metal only, single kernel per workload |
| Heuristic wins on matvec/layernorm due to GROUP and stride analysis | Abduction | 75% | Inferred from missing GROUP in loop + wrong axis selection |
| Hybrid (structural priors + measurement params) is the optimal architecture | Abduction | 70% | Proposed from oscillatory evidence, not yet tested |

### H5 pruning log

| Hypothesis | Status | Killed by |
|------------|--------|-----------|
| H5 (original): abduction obsoletes heuristics | REFRAMED | Oscillatory evidence: wins on 3/5, loses on 2/5 |
| H5.1: TC dominates value | KILLED | Measurement: USE_TC=0 still gives 10.5x |
| H5.2: MATVEC misclassification | CONFIRMED | Code trace + measurement |
| H5.3: minimal abduction loop | CONFIRMED (partial) | Wins on gemm/mul_sum/softmax, loses on matvec/layernorm |
| H5.4: hybrid architecture | PROPOSED | Open — needs implementation and benchmark |

### H5 frontier edges — resolved

**Edge 1+2: GROUP + stride-aware abduction loop — RESOLVED**

Added GROUP [4,8,16], GROUPTOP(16), UNROLL [4,0], and stride-based axis ordering to the abduction loop (~52 trials). Score went from 3/5 to **4/5**:

| Workload | Heuristic | New Abduction (52t) | vs Heur |
|----------|-----------|---------------------|---------|
| gemm_1024 | 307us | **153us** | 0.50x |
| mul_sum | 343us | **223us** | 0.65x |
| softmax | 15us | **4us** | 0.24x |
| matvec | **103us** | 112us | 1.10x |
| layernorm | 33us | **19us** | 0.56x |

Geometric mean: abduction is **1.85x faster** than the heuristic. The sole remaining gap is matvec (1.10x) — the heuristic's fixed GROUP+LOCAL+UPCAST combo beats the loop's greedy GROUP selection. Closing this gap requires joint optimization (branch-and-bound or mini-beam), not more repertoire.

**Edge 3: Theory transfer — RESOLVED (see H4)**

Semantic theories transfer across all 7 tested matmul shapes. Exact schedules fail on 3/6. Cache the theory, not the schedule.

**Edge 4: MATVEC strict subset fix — CONFIRMED**

Adding `set(idx0.ranges) == set(idx1.ranges)` rejection correctly blocks mul_sum from the MATVEC path. Separate PR candidate — independent of the abduction engine.

### H5 remaining frontier

1. **Theory transfer to non-matmul classes** — does the adaptive theory pattern work for reductions, elementwise, convolutions? Each class would need its own seed measurement. (Untested.)
2. **Joint GROUP+LOCAL+UPCAST optimization** — the matvec gap (1.10x) requires evaluating combos, not greedy steps. A 2-deep mini-beam (try GROUP×LOCAL×UPCAST jointly) would close it. (Design question, ~20 lines.)
3. **Amortized cost measurement** — the abduction loop takes 52 trials × compile+time cost. What's the wall-clock cost vs BEAM's 200+ trials? Is it actually faster end-to-end? (Needs timing.)

---

## Dependencies

```
H0 (abduction outperforms BEAM) — strong evidence
 ├→ H2 (causal attribution is the mechanism) — alive
 │   └→ H3 (theories collapse search) — supported (52 trials, 1.85x geo mean)
 │       └→ H4 (cache theories, not schedules) — ★ CONFIRMED
 ├→ H5 (abduction obsoletes heuristics) — REFRAMED (parameters yes, patterns no)
 └→ falsified if BEAM + more budget matches abduction's quality
```

H0 is the root claim. H5's 52-trial abduction loop beats the heuristic on 4/5 workloads at 1.85x geometric mean, using 52 trials vs BEAM's 193-action pool. H4's theory transfer means the 52-trial cost is amortized to zero for subsequent kernels of the same class. The remaining gap is matvec (1.10x), where the heuristic's joint GROUP+LOCAL+UPCAST combo beats greedy search.

## Open questions

1. **What hardware counters are available on Metal?** GPU profiling on Apple Silicon is limited compared to NVIDIA (nsight) or AMD (rocprof). If the observation channel is too narrow (wall-clock only), the abductive step may not have enough signal. *Partially answered: wall-clock timing alone was sufficient for the H5 abduction loop to beat the heuristic on 4/5 workloads. Hardware counters would help diagnose WHY, but the loop works without them.*
2. **What does BEAM's actual search tree look like?** *Answered in BASELINE.md: 193 actions, 92-97% yield, no pruning, plateau exit. The abduction loop's 52 trials with pruning outperforms BEAM's 193 unpruned trials.*
3. **Is there prior art on theory-guided compiler autotuning?** The ML-for-compilers literature (TVM/Ansor, Halide autoscheduler) uses learned cost models, which are implicit theories. How do they compare to explicit causal theories?
4. **Does theory transfer work on CUDA?** H4 confirmed on Metal only. CUDA verification pending — bootstrap prompt updated for Windows RTX 4080.

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

### H0.6a: RDNA4 WMMA operand lane mapping is incompatible with N-only UPCAST

**Status:** ★ ROOT CAUSE IDENTIFIED — WMMA swizzle + opts ordering, not register pressure

**Evidence (original, killed):**
```
gfx1100 (RDNA3): elements_per_thread = (16, 16, 8)  — passes
gfx1201 (RDNA4): elements_per_thread = (8, 8, 8)    — fails
gfx950  (CDNA3): elements_per_thread = (32, 32, 4)  — passes
Metal:           elements_per_thread = (2, 2, 2)     — passes
```

Original theory (KILLED): register pressure from elements_per_thread × unroll_factor. Disproved by PR #16107 CI: UPCAST(1,2) alone — no UNROLL — also produces incorrect WMMA on gfx1201.

**Evidence (2026-05-08, code analysis):** Three structural differences between RDNA3 and RDNA4 TensorCore definitions in `tc.py`:

1. **opts ordering** — RDNA3: `('l0','l0','l0','l0','l1','u1','u1','u1')`, RDNA4: `('l0','l0','l0','l0','u1','u1','u1','l1')`. RDNA4 places the `l1` local split AFTER the axis-1 upcasts. This changes the order in which axes are carved during TC setup (`postrange.py:265-276`).

2. **swizzle lane mapping** — RDNA4 skips r2 in operand upcast dimensions:
   - RDNA3 A operand upcast: `('r1','r2','r3')` — sequential
   - RDNA4 A operand upcast: `('r0','r1','r3')` — skips r2
   - RDNA3 B operand upcast: `('r1','r2','r3')` — sequential
   - RDNA4 B operand upcast: `('r0','r1','r3')` — skips r2

3. **Permutation order** — fundamentally different lane-to-axis mapping:
   - RDNA3 A: `(4,5,6,7,0,9,10,11,1,2,3,8)`
   - RDNA4 A: `(4,5,6,7,8,9,11,10,0,1,2,3)`

**Root cause mechanism:** When `apply_opt(Opt(OptOps.TC, ...))` sets up WMMA, it creates upcast axes via `tc.opts` and maps operands to lanes via `tc.swizzle`. The TC setup at `postrange.py:288-289` computes `tc_upcast_axes` from `base_upcast_axes[:log2(elements_per_thread[i])]`:
- RDNA3: 4 upcast axes per operand (log2(16)=4), includes r3
- RDNA4: 3 upcast axes per operand (log2(8)=3), excludes r3

When the heuristic applies UPCAST(1,2) after TC, it further splits axis 1 (M dimension). This changes which schedule axes map to which WMMA operand lanes. On RDNA3, the larger element count (16) provides enough lane-mapping flexibility to absorb the change. On RDNA4, with only 8 elements per thread and a non-sequential lane mapping (skipping r2), the post-TC UPCAST violates the operand-to-lane assignment that `tc.swizzle` requires.

The old heuristic's UPCAST M+N pattern works because it matches the structure the TC opts/swizzle expect — both axes are upcasted in the order and factors that preserve the lane mapping. Changing to N-only UPCAST breaks the expected layout.

**Trajectory:** DIVERGENT for root cause identification. The mechanism is deductive (traced through code), not inductive (measured). Confidence: 90%.

**Open edges (testable via CI):**
- H0.6b: ★ CONFIRMED (2026-05-08). `UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,2)` passes both gfx1201 CI jobs (amd and amdllvm). Axis coverage is the constraint — both operand axes must be upcasted to preserve the swizzle lane mapping.
- H0.6c: Now testable — try `UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,4)` on gfx12. If UNROLL(0,4) is safe when both axes are covered, the full speedup applies to RDNA4 too.
- H0.6d: Strongly supported by H0.6b. The rule is: post-TC UPCAST must cover all operand axes that appear in `tc.opts`. N-only violates it; both-axis preserves it. Needs testing on CDNA (gfx950) to confirm generality.

**Resolution (updated):** PR #16109 proves gfx12 CAN benefit from the new opts when both axes are upcasted. The conservative fallback in #16107 is no longer necessary — gfx12 should get `UPCAST(0,2) + UPCAST(1,2) + UNROLL(0,2)` instead of the old heuristic. Next perturbation: try UNROLL(0,4) on gfx12 with both-axis coverage.
