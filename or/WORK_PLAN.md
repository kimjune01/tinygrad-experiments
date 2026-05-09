# Work Plan

## Root hypothesis

Applying the OR lens produces compiler optimizations that exceed PyTorch performance on standard workloads, in fewer lines than the compiler community's ad-hoc versions.

## Baseline (2026-05-07)

`or-baseline` branch: master + PR #16072 (matvec) + PR #16094 (contiguous-prune).

Two workloads already beat PyTorch: gemm_1024 (0.84x), add_4096 (0.82x).
Remaining gap is dominated by fusion: layernorm (3.98x), softmax (2.43x), matvec (2.45x), sum (2.24x).

## Dependency graph

```
H1 (CPL + APRP scheduling) ──prerequisite──→ H5 (algebraic fusion)
                            ──prerequisite──→ H4 (fused dequant)
H2 (XOR-swizzle) ──────────── independent
```

## Phase 1: H1 — Instruction scheduling

**Target file:** `tinygrad/codegen/late/linearizer.py` (~53 lines)

**Step 1a: CPL priority (~30 lines)**
Replace flat `{LOAD:-1, ALU:0, STORE:+1}` with critical path length.
Benchmark: bench_perceive, focus on mul_sum (lift 0.9x) and sum_4096 (lift 0.8x) where the heuristic currently hurts.

**Step 1b: LUC tie-breaking (~10 lines)**
When CPL ties, prefer the node that closes the most live ranges.
Benchmark: same workloads, check register pressure via DEBUG=4.

**Step 1c: APRP ceiling (~40-60 lines)**
Two-pass: RP-minimizing pass finds the APRP bound, CPL pass respects it.
Benchmark: check relu_4096 and matvec — these are where greedy scheduling is most likely to cross an occupancy boundary.
Kill metric: if 1c regresses any workload that 1a improved, the APRP table is wrong for Metal.

**Gate:** bench_perceive with H1 applied. Must not regress gemm_1024 (already 0.84x). Must improve at least one of: mul_sum, sum_4096, matvec.

## Phase 2: H2 — Bank conflict avoidance (parallel with Phase 1)

**Target file:** new UOp rewrite rule, likely in `tinygrad/codegen/late/` or `tinygrad/schedule/`

**Step 2a: XOR-swizzle rewrite (~30 lines)**
Pattern-match STORE/LOAD to DEFINE_LOCAL in GEMM-shaped kernels.
Insert `idx XOR (((idx >> (MBase + SShift)) & mask) << MBase)`.
Benchmark: bench_perceive gemm_1024 and gemm_256.

**Step 2b: Conflict detection heuristic (~10 lines)**
Only apply swizzle when stride % 32 == 0 (bank conflict present).
Benchmark: test_opt_gemm correctness gate.

**Gate:** gemm_1024 must not regress (currently 0.84x — already beats PyTorch). gemm_256 (1.71x) is the primary improvement target.

## Phase 3: H5 — Algebraic fusion (after Phase 1)

**Target file:** new PatternMatcher rules, likely in `tinygrad/schedule/` or `tinygrad/codegen/`

**Step 3a: Flashlight online softmax (~80-100 lines)**
Detect reduce-elementwise-reduce where elementwise is exp (homomorphism).
Rewrite to single-pass with O(1) running state (max, sum).
Benchmark: bench_perceive softmax (2.43x → target <1.5x).
Correctness gate: test_softmax_fusion.py MUST pass.

**Step 3b: RedFuser layernorm decomposition (~60-80 lines)**
Detect mean-variance-normalize pattern.
Rewrite to single-pass tracking sum_x and sum_x² (separable).
Benchmark: bench_perceive layernorm (3.98x → target <1.5x).

**Step 3c: Moderate fusion cost model (~40-60 lines)**
Before fusing, estimate register pressure of fused kernel.
If fused pressure exceeds APRP ceiling (from H1), keep separate kernels.
This is the SpaceFusion++ "moderate fusion" idea.

**Gate:** bench_perceive softmax and layernorm must beat PyTorch (<1.0x). If they match but don't beat, the OR lens achieved parity but not the ambitious target. If they regress vs 3-kernel baseline, the algebraic decomposition didn't achieve O(1) state — recheck.

**Decisive test (from PERCEIVE_ANALYSIS claim 4):** kernel count drops from 3 to 1 AND runtime improves. PCONTIG=99 proved kernel count alone doesn't help. Algebraic decomposition must show qualitative improvement.

## Phase 4: H4 — Fused dequantization (after Phase 1)

**Target:** UOp rewrite rule fusing LOAD(int4) → BITUNPACK → SCALE → CAST into GEMM.

**Step 4a: Dequant pattern matcher (~50-80 lines)**
Detect quantized weight load → dequant arithmetic → matmul.
Fuse dequant into matmul kernel.
APRP check (from H1): reject fusion if register pressure exceeds ceiling.

**Step 4b: Offline weight layout (~30-50 lines)**
Ensure GGUF loading produces layouts where scale factors align with tile boundaries.

**Gate:** LLM inference tokens/sec improvement. No existing benchmark — need to build one or use the LLaMA runner directly.

## Measurement protocol

- **Oracle:** `python3 test/speed/bench_perceive.py` on Metal
- **Correctness gates:** test_softmax_fusion.py, test_opt_gemm.py, test_linearizer.py
- **Trajectory classification (from /investigate):**
  - Divergent improvement on all workloads → proceed
  - Oscillatory (helps some, hurts others) → split, re-enter hypothesis graph
  - No improvement → diagnosis was wrong, re-enter hypothesis graph
- **Ambiguity heuristic:** when two implementations perform similarly, prefer fewer lines

## Platform note

Current benchmarks are Metal on M5 Max. PyTorch MPS is its least mature backend — gemm_1024 and add_4096 beat MPS but this may not hold against cuBLAS on CUDA. CUDA Windows machine available for cross-validation. The fusion gap (H5) is backend-agnostic and the more generalizable result.

## Success criteria

- **Table stakes:** match PyTorch on all 11 bench_perceive workloads (v_torch ≤ 1.0x)
- **Ambitious:** beat PyTorch on ≥8 of 11 workloads
- **Cross-platform:** verify on CUDA before claiming generality
- **Root hypothesis validated:** total added lines < 300, improvement attributable to OR-derived techniques

## Status (2026-05-07)

### Hypothesis results

| Hypothesis | Status | Finding |
|---|---|---|
| H1.1a: CPL scheduling | **CONFIRMED** | 23% matvec kernel speedup (1 instruction reorder). Neutral elsewhere. |
| H1.1b: LUC tie-breaking | DEFERRED | CPL alone doesn't regress — low urgency |
| H1.1c: APRP ceiling | DEFERRED | Kill condition didn't fire for scheduling |
| H2: XOR-swizzle | **KILLED (Metal)** | No shared memory in GEMM — simdgroup WMMA bypasses it |
| H4: Fused dequant | PENDING | Waiting on CUDA results |
| H5: Algebraic fusion | **REFRAMED** | Fusion gap is Python scheduling overhead, not kernel quality at bench_perceive scale |

### Key reframe

**tinygrad's GPU kernels already beat PyTorch MPS on every workload.** The bench_perceive gap is 98% Python scheduling overhead (graph_rewrite pattern matching). This is a known bottleneck (issue #13488) under active optimization by the tinygrad team.

The OR hypotheses correctly identify kernel-level optimizations, but the kernel is not the bottleneck at bench_perceive tensor sizes. At LLM scale (large graphs, many tokens), kernel execution dominates and the OR hypotheses become relevant.

### Benchmark improvements

bench_perceive now reports GPU kernel time (via GlobalCounters.time_sum_s at DEBUG=2) alongside wall time. Softmax/layernorm tensor sizes increased so GPU stays above idle frequency. Branch: `or-cpl-bench` on kimjune01/tinygrad fork.

### Pending

- CUDA cross-validation (Windows machine, in progress)
- H5 at LLM scale (where kernel time dominates)
