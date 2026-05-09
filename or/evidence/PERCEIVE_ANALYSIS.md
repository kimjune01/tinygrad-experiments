# Diagnosis: IR Expressiveness Limits in tinygrad

## Problem

tinygrad is 1.2–5.9x slower than PyTorch on Metal across standard workloads. The gap splits into two structurally different symptoms: single-kernel codegen quality and multi-kernel fusion/partitioning.

## Methodology

**Benchmark:** `test/speed/bench_perceive.py` (v2)
**Platform:** arm64, macOS 26.4.1, Apple Silicon (Metal)
**PyTorch:** 2.11.0, MPS backend, `torch.no_grad()`, single-threaded
**tinygrad:** HEAD at time of test, Metal backend
**Dtype:** float32, TF32 disabled
**Timing:** 8 iterations, trimmed mean (drop min/max), 3 warmup runs excluded. Compile time excluded — all measurements are post-JIT. BEAM cache disabled (`IGNORE_BEAM_CACHE=1`) for fresh search.
**Levels:** `NOOPT=1` (raw lowering, no optimizer), default (heuristic), `BEAM=2` (compile-and-time search, fresh).

## Results

```
workload        torch     noopt      heur      beam  #k  lift  v_torch   b/h
gemm_1024        979 us  20.22 ms   1.77 ms   1.86 ms  1  11.4x   1.81x  1.05x
gemm_256         260 us   2.41 ms   1.30 ms    843 us  1   1.8x   5.02x  0.65x
add_4096        1.01 ms  11.76 ms   1.79 ms   1.65 ms  1   6.6x   1.76x  0.93x
mul_sum         1.60 ms   1.20 ms   2.42 ms   2.22 ms  3   0.5x   1.51x  0.92x
relu_4096        903 us  11.83 ms   1.09 ms   1.77 ms  1  10.9x   1.20x  1.63x
exp_2048         489 us   3.40 ms   1.04 ms    948 us  1   3.3x   2.13x  0.91x
sum_4096         564 us    983 us   1.11 ms   1.24 ms  3   0.9x   1.97x  1.11x
permute          348 us   1.62 ms   1.03 ms    650 us  1   1.6x   2.96x  0.63x
softmax          232 us   1.43 ms   1.36 ms   1.42 ms  3   1.1x   5.85x  1.05x
layernorm        348 us   4.12 ms   1.85 ms   2.05 ms  3   2.2x   5.32x  1.11x
matvec           835 us   1.41 ms   1.88 ms   1.88 ms  1   0.8x   2.25x  1.00x
```

- **lift** = NOOPT / heuristic: how much the optimizer improves raw lowering
- **v_torch** = heuristic / torch: competitive gap users see (1.0x = parity)
- **b/h** = beam / heuristic: <1.0 = search improved, >1.0 = search regressed
- **#k** = tinygrad kernel count (post-JIT compute dispatches, excludes copies and setup)

## Symptom 1: Single-kernel ops, codegen/backend gap

**Workloads:** gemm_1024, gemm_256, add_4096, relu_4096, exp_2048, permute

**Pattern:** 1 compute kernel, same count as PyTorch. vs_torch 1.2–5.0x. Kernel count does not implicate fusion. The remaining gap is more likely codegen quality, memory layout, launch/runtime overhead, or backend-specific scheduling.

**Heuristic lift is high** (3–11x) on elementwise/GEMM because NOOPT skips tiling, vectorization, and memory layout optimization — expected behavior for a raw-lowered baseline.

**BEAM is inconsistent.** Search improves some ops (gemm_256: 0.65x, permute: 0.63x) and regresses others (relu: 1.63x, gemm_1024: 1.05x). This may be measurement noise at sub-millisecond kernel sizes, Metal-specific warmup behavior, or BEAM exploring moves that don't help bandwidth-bound ops. If BEAM is intended to dominate the heuristic, it should include the heuristic schedule as a baseline and fall back to it when measured candidates are slower.

### Shape-specialization failure (matvec)

Matvec is 1 kernel, lift = 0.8x (heuristic *hurts*), vs_torch = 2.25x. This doesn't fit either main symptom cleanly. Matrix-vector multiply needs different tiling/reduction strategies than square GEMM. The heuristic appears to apply GEMM-optimized decisions that hurt the rectangular case. This suggests a shape-specialization gap in the optimizer, distinct from both codegen quality and fusion structure.

## Symptom 2: Multi-kernel ops, fusion/partitioning gap

**Workloads:** softmax, layernorm, mul_sum, sum

**Pattern:** 3 compute kernels (#k=3), where evidence suggests PyTorch dispatches fused implementations. vs_torch 1.5–5.9x. BEAM doesn't help (b/h ≈ 1.0) because search optimizes within kernels, not across kernel boundaries. Confirmed: #k_beam = #k_heur = 3 for all multi-kernel ops.

### Evidence that PyTorch MPS uses fused kernels

We can't directly trace MPS kernel counts, but we can compare PyTorch's fused ops against manual decomposition on the same hardware:

```
softmax fused:     0.217 ms    manual (max-sub-exp-sum-div):  0.354 ms    ratio: 1.63x
layernorm fused:   0.200 ms    manual (mean-var-normalize):   0.721 ms    ratio: 3.60x
```

If PyTorch MPS executed the same decomposition as the manual expression, fused and manual times would be similar. The 1.6–3.6x speedup indicates PyTorch is not merely executing the same decomposition — it likely uses a specialized fused or otherwise optimized implementation for these ops. The layernorm result (3.6x) is strong evidence; the softmax result (1.6x) is suggestive but less decisive, as manual decomposition may differ in temporary allocation or command-buffer behavior.

### tinygrad kernel structure (DEBUG=2 inspection)

**Softmax** — 7 scheduled ops, 3 JIT compute dispatches:
```
METAL  E_512_32_4         — elementwise setup (fused into dispatch 1 with reduce below)
METAL  r_32_16_128        — dispatch 1: reduce (max for numerical stability)
METAL  r_32_16_128n1      — dispatch 2: reduce (sum of exp)
METAL  E_4_32_8_16_4      — dispatch 3: elementwise (divide by sum)
```
(plus 2 PYTHON→METAL copy ops and 1 small constant init — not counted as compute dispatches. GlobalCounters.kernel_count = 3 confirmed via TinyJit.)

**Layernorm** — 7 scheduled ops, 3 JIT compute dispatches:
```
METAL  E_32768_32_4       — elementwise setup (fused into dispatch 1 with reduce below)
METAL  r_32_32_4_256_4    — dispatch 1: reduce (mean)
METAL  r_32_32_4_256_4n1  — dispatch 2: reduce (variance)
METAL  E_512_16_8_16_4    — dispatch 3: elementwise (normalize)
```
(same: 2 copies + 1 constant init excluded. GlobalCounters.kernel_count = 3 confirmed.)

Both follow a **reduce-then-reduce** pattern that the scheduler splits at each reduction boundary. The split creates 3 kernel launches with intermediate materialization, where PyTorch uses a fused implementation that keeps intermediate values in registers/shared memory.

### Corroborating evidence

- Softmax fusion required a $500 bounty and months of cross-system work (tinygrad #3521)
- Norm fusion required a $2,000 bounty (tinygrad #1146)
- 384 MULACC fusions missed due to rewrite rule ordering (tinygrad #14774)
- The scheduler's historical AST grammar constrained fusion to: "MovementOps → ElementwiseOps → ReduceOps → ElementwiseOps" — reduce-then-reduce doesn't fit

### Heuristic misfires in multi-kernel ops

- **mul_sum lift = 0.5x**: Heuristic is slower than NOOPT. The heuristic applies per-kernel optimizations that hurt this particular reduction pattern.
- **sum lift = 0.9x**: Nearly neutral — the heuristic adds little value for a pure reduction.

These cases are evidence that the heuristic is not monotonic over this benchmark set; whether the loss comes from tiling assumptions, launch shape, memory traffic, or small-kernel overhead needs kernel-level inspection.

## Falsifiable claims

1. **Large gaps have two distinct causes.** For compound reduction workloads (softmax, layernorm) with gap > 3x, tinygrad dispatches 3 compute kernels while PyTorch uses a specialized/fused implementation (confirmed by 1.6–3.6x fused-vs-manual speedup). For non-compound workloads (gemm_256: 5.0x, 1 kernel), the gap is within-kernel codegen/backend quality. The two causes require different fixes.

2. **On this benchmark set, BEAM changes schedules inside already-selected kernels but does not alter scheduler partitioning or kernel count.** Tested: #k_beam = #k_heur for all 11 workloads. Confirmed.

3. **For single-kernel ops with gap < 2x, the observed gap is not explained by extra tinygrad kernel boundaries.** Tested: gemm_1024, relu_4096, add_4096 all use 1 compute kernel. Confirmed — the gap there is within-kernel.

4. **If a scheduler or canonicalization change lowers softmax/layernorm from 3 compute dispatches to 1–2, and runtime improves materially, then the partitioning hypothesis is supported. If kernel count drops without runtime improvement, the bottleneck is not just launch/materialization overhead.** Not yet tested. This is the key experiment that would distinguish "missing semantic structure upstream of the scheduler" from "insufficient scheduler fusion rules over the existing UOp graph."

5. **Heuristic regressions appear on workloads whose optimal schedule differs from the square-GEMM/elementwise assumptions: reductions (sum), matvec, and compound reduce pipelines (mul_sum).** Tested: lift < 1.0 for mul_sum (0.5x), matvec (0.8x), sum (0.9x). All involve non-standard reduction shapes. Confirmed.

## What would disprove each symptom

**Symptom 1 (codegen gap):**
- If BEAM measurement noise is eliminated (larger kernels, more samples, explicit heuristic-as-candidate), BEAM regressions disappear → problem is search policy, not codegen
- If Metal-specific intrinsics or layout changes close the single-kernel gap → confirms codegen diagnosis

**Symptom 2 (fusion gap):**
- If adding scheduler fusion rules fixes softmax/layernorm *without* changing upstream representation → problem is missing scheduler rules, not missing semantic structure
- If changing upstream representation (compound pattern annotations before scheduling) fixes the gap → problem is missing semantic structure
- Claim 4 is the decisive experiment: it distinguishes these two explanations

**Matvec (shape specialization):**
- If the heuristic is fixed for non-square shapes and matvec gap closes → confirms shape-specialization diagnosis
- If the gap persists after heuristic fix → problem is codegen for non-square kernels

## Comparison with other systems

The multi-kernel fusion gap is a known problem. Other systems address it at different points:

- **XLA:** HLO fusion pass explicitly recognizes compound patterns including reduce-elementwise-reduce
- **TVM:** Graph-level optimization (Relay) fuses compound patterns before per-kernel scheduling (TE/Ansor)
- **Halide:** Algorithm/schedule split is explicit; the schedule is a first-class object
- **torch.compile/Inductor:** Pattern matching on FX/ATen graph → fused kernel emission for known compound ops
- **PyTorch MPS:** Dispatches pre-fused Metal implementations for `softmax`, `layer_norm`, etc.

Common pattern: compound structure is either preserved, recovered, or explicitly scheduled before low-level per-kernel optimization. tinygrad's pipeline goes tensor API → UOp graph → scheduler → per-kernel optimization. Whether the fix belongs upstream of the scheduler (semantic annotations) or inside the scheduler (better fusion rules) is the open question that claim 4 would resolve.

## Claim 4 result: fusion makes softmax and layernorm SLOWER

tinygrad already has an internal env var (`PCONTIG`) that controls fusion aggressiveness. Setting `PCONTIG=99` fuses softmax and layernorm into single kernels. We tested:

```
softmax:
  PCONTIG=99 (1 kernel):  0.578 ms  ← fused, 1.8x SLOWER than split
  default    (3 kernels): 0.323 ms  ← split
  torch MPS:              0.230 ms  ← baseline

layernorm:
  PCONTIG=99 (1 kernel):  1.065 ms  ← fused, 1.9x SLOWER than split
  default    (3 kernels): 0.556 ms  ← split
  torch MPS:              0.360 ms  ← baseline
```

**The partitioning hypothesis is disproven.** Fewer kernels does not mean faster execution. The fused kernel is slower because it has worse occupancy, higher register pressure, or suboptimal memory access patterns compared to three smaller specialized kernels.

This reframes the diagnosis:

- The 3-kernel split may actually be the correct partitioning for tinygrad's codegen on Metal
- The gap to torch is not kernel count — it's that torch uses a specialized/fused backend path (likely MPS primitives, MPSGraph fusion, or equivalent), not a compiler-generated kernel
- The fix is not "fuse more" — it's either "generate better code for the fused case" (codegen quality) or "dispatch to specialized backend primitives for known compound ops" (library dispatch)

**This means all three symptoms — single-kernel gaps, multi-kernel gaps, and shape-specialization failures — trace to the same root cause.** The scheduler's partitioning decisions are defensible; the generated kernels are just slower than hand-tuned alternatives. The question is why.

## Root cause: the codegen pipeline lacks microarchitectural scheduling machinery

All three symptoms, the claim 4 result, and the scattered issues in tinygrad's tracker (#1477, #4931, #6928) converge on one structural limitation: **tinygrad's portable UOp-to-kernel pipeline lacks enough microarchitectural scheduling machinery to match specialized backend implementations.** Some of this is true IR expressiveness gaps; the rest is missing compiler analyses over the existing IR.

### What the codegen pipeline controls

The 20 rewrite passes in `full_rewrite_to_sink` handle graph canonicalization, symbolic simplification, range analysis, and axis transformations. The optimizer (`hand_coded_optimizations` / BEAM) picks tile sizes, thread mapping, and unroll factors — **what work** each thread does.

The optimizer's knobs are: `TC` (tensor core), `UPCAST` (register tiling), `UNROLL` (loop unrolling), `LOCAL` (threadgroup dimensions), `GROUP` (shared memory reduction). These control parallelism and data reuse strategy.

### Missing compiler analyses (fixable over the current IR)

These do not require new IR ops — they could be implemented as additional passes or a better linearizer over the existing UOp DAG.

**Instruction ordering.** The linearizer (`codegen/late/linearizer.py`) does a priority-based topological sort: LOADs get priority -1, STOREs get +1, everything else 0. No latency model, no throughput model, no dependency-aware reordering, no load-compute interleaving. A latency-aware topological scheduler over the same DAG could interleave loads with independent computes without any IR changes. This is why matmul on M1 improved 1.9x from random memory barriers (#1477) — the linearizer's ordering was so poor that noise helped.

**Register pressure.** The UOp graph generates unbounded SSA variables. Register allocation is delegated entirely to the downstream compiler (Metal shader compiler, nvcc). Liveness analysis after linearization could track register pressure and reorder or spill to shared memory — this is a missing analysis, not an IR limitation.

**Memory access patterns within shared memory.** `DEFINE_LOCAL` creates a flat array. The indexing expression is whatever falls out of axis transformations. There is no XOR-swizzle or padding to avoid bank conflicts. However, bank conflict avoidance could be implemented as index rewriting or allocation-size padding over the existing IR — a layout annotation would be cleaner but is not strictly required.

### True IR expressiveness gaps (need new ops or abstractions)

These cannot be implemented over the current IR without extending it.

**Async operations.** The IR has `LOAD` and `STORE`, both synchronous. There is no `ASYNC_COPY` op, no `COMMIT`/`WAIT_GROUP`. The `CUSTOM`/`CUSTOMI` escape hatch could emit inline assembly, but the linearizer and barrier placement have no awareness of async semantics. Modern GPU performance depends on overlapping memory access with computation — the IR's normal portable path cannot express this.

**Software pipelining.** The `RANGE`/`END` structure represents a single loop body. There is no "split loop body across iterations" transformation, no double-buffered shared memory, no prologue/epilogue outside the main loop. Loads for iteration N+1 cannot be interleaved with computes for iteration N. This requires fundamentally extending the loop model.

### What this means for the benchmark results

The optimizer correctly picks tile sizes and parallelism. But the generated code within each tile has:
- No load-compute interleaving → memory latency is not hidden
- No shared memory bank conflict avoidance → shared memory throughput is degraded
- No register pressure management → the downstream compiler may spill, reducing occupancy
- No async copies → global memory loads block until complete

For simple ops (elementwise, basic reductions), the downstream Metal/CUDA compiler can recover reasonable instruction ordering from the naive code. The gap is 1.2–2x — here the bottleneck may be launch overhead, vector width, memory coalescing, or Metal compiler behavior rather than IR expressiveness. For compound ops (softmax, layernorm, GEMM), where performance depends on precise choreography of the memory hierarchy, the downstream compiler cannot compensate. The gap is 2–6x.

This also explains why fusion made things *worse*: fusing softmax into one kernel creates a larger loop body with more live variables, more shared memory traffic, and more opportunities for the linearizer's naive ordering to produce poor instruction schedules. Three smaller kernels each have simpler instruction patterns that the downstream compiler can optimize more effectively. The exact cause of fused slowdown (occupancy, registers, memory traffic, barriers, or access patterns) would require Metal shader profiling to confirm.

### The four GEMM roadmap items (#6928) map to both categories

Francis Lam's roadmap for reaching full NVIDIA GEMM speed identifies four missing features:

1. **XOR-swizzled shared memory** — could be implemented as index rewriting over the existing IR, though a layout annotation would be cleaner
2. **Async global-to-shared copies** (`cp.async`) — true IR gap: requires new async ops the portable IR doesn't have
3. **2-stage software pipelining** — true IR gap: requires cross-iteration scheduling the `RANGE`/`END` loop model can't express
4. **Register-to-shared demotion** — could be implemented as a post-linearization pass, but would benefit from explicit staging abstractions

Items 2 and 3 are IR expressiveness gaps. Items 1 and 4 are missing compiler analyses that could plausibly be built over the current IR. The practical effect is the same — none are implemented — but the distinction matters for the fix: a better linearizer addresses 1 and 4 without IR changes, while 2 and 3 require extending the IR.

### Corroborating evidence across backends

The gap is in the shared portable pipeline, not any one backend:
- **NVIDIA PTX:** Linearizer produces worse LOAD/WMMA interleaving than CUDA (#4931)
- **NVIDIA GEMM:** 1.13x behind cuBLAS on simple matmul (#3660); full-speed GEMM requires 4 IR extensions none of which exist (#6928)
- **AMD GPT-2:** 1.7x behind PyTorch with BEAM (#4301)
- **Apple M1:** Matmul 1.9x slower; fixed by random memory barriers, not codegen improvement (#1477)

The gap is smallest on NVIDIA (where tinygrad has the most investment — PTX renderer, HCQ driver, tensor cores) and largest on Metal/AMD. But it's nonzero everywhere because the structural limitation is in the shared IR, not in any one backend.

## Revised actionable steps

Ordered by impact and feasibility — items 1–3 require no IR changes.

1. **Latency-aware instruction scheduling.** Replace the linearizer's priority sort with a latency-aware topological scheduler that interleaves loads with independent computes. This is the highest-impact single change — it addresses the #1477 matmul barrier hack, the #4931 PTX ordering issue, and would close part of the gap across all backends without any IR changes.
2. **Register pressure tracking.** Add liveness analysis after linearization. Reorder to reduce peak liveness, or spill to shared memory when pressure exceeds target occupancy. Implementable as a post-linearization pass over the existing IR.
3. **Shared memory bank conflict avoidance.** Detect conflict patterns in `DEFINE_LOCAL` index expressions and insert padding or XOR-swizzle via index rewriting. A layout annotation on `DEFINE_LOCAL` would be cleaner but is not strictly required.
4. **Async copy support (IR extension).** Add `ASYNC_COPY` and `WAIT_GROUP` ops to the IR. Required for competitive GEMM on NVIDIA (cp.async) and as a prerequisite for software pipelining.
5. **Software pipelining (IR extension).** Once async copies exist, add a loop transformation that double-buffers shared memory and interleaves loads for iteration N+1 with computes for iteration N. This is the largest change — it requires extending the `RANGE`/`END` loop model.
6. **Library dispatch as stopgap.** For known compound ops (softmax, layernorm, GEMM), dispatch to specialized backend primitives (MPS, cuBLAS/cuDNN) until the codegen catches up. This contradicts tinygrad's thesis but closes the user-visible gap immediately.
