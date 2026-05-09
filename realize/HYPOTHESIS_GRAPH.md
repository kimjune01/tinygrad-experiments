# Hypothesis Graph: tinygrad performance investigation

Each node: hypothesis (null), perturbation, e-value trajectory shape, kill condition, edge to next node.

Framework: [The Hypothesis Graph](https://june.kim/the-hypothesis-graph). Classification: [Evidence has a trajectory](https://june.kim/evidence-has-a-trajectory). Reasoning modes: [Modes of Reason](https://june.kim/modes-of-reason).

---

## H₀: tinygrad matches PyTorch

- **Null:** performance ratio ≈ 1.0
- **Perturbation:** benchmark 11 ops on Metal (fp32, post-JIT, trimmed mean, 8 iterations)
- **Trajectory:** ratios across workloads: 1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 3.0, 5.0, 5.0, 5.3, 5.9. Monotonically accumulating evidence against the null.
- **Shape:** **divergent**. Kill H₀.
- **Edge →** why?

## H₁: the bottleneck is graph encoding

- **Null:** the encoding is adequate; downstream optimization is the issue
- **Perturbation:** measure NOOPT / heuristic / BEAM across workloads
- **Trajectory:** optimizer lift varies wildly — 0.5x, 0.8x, 0.9x, 1.1x, 1.6x, 1.8x, 2.2x, 3.3x, 6.6x, 10.9x, 11.4x. BEAM regresses on some ops, improves on others. Evidence for the hypothesis waxes and wanes depending on the workload.
- **Shape:** **oscillatory**. The hypothesis fits elementwise/GEMM (high lift) but not compound ops (low lift, BEAM neutral). Two modes visible.
- **Kill condition:** oscillation means the hypothesis is too coarse. The two modes need separate explanations.
- **Edge →** split into H₁ₐ (multi-kernel: fusion?) and H₁ᵦ (single-kernel: codegen?)

## H₁ₐ: multi-kernel ops are slow because of fusion gaps

- **Null:** kernel count doesn't explain the gap
- **Perturbation:** PCONTIG=99 forces softmax/layernorm into 1 kernel. Also: torch fused-vs-manual timing on same hardware.
- **Trajectory:** softmax 3→1 kernel: 0.323ms → 0.578ms (worse). Layernorm: 0.556ms → 1.065ms (worse). The e-value for "fusion helps" was accumulating, then reversed sharply. Meanwhile torch fused-vs-manual: 1.6x softmax, 3.6x layernorm — torch's specialized path is real.
- **Shape:** **divergent against the hypothesis**. Evidence grows monotonically that fusion hurts with current codegen. Kill H₁ₐ.
- **Kill condition:** fewer kernels ≠ faster. The scheduler was right to split. Torch is fast not because it fuses — because it dispatches to a specialized implementation.
- **Edge →** the gap persists across single-kernel and multi-kernel ops alike. Something common to both.

## H₁ᵦ: within-kernel codegen quality

- **Null:** generated code is adequate
- **Perturbation:** read the linearizer source code, enumerate what it does and doesn't do
- **Trajectory:** no latency model (evidence grows). No register tracking (grows). No load-compute interleaving (grows). Random memory barriers improved matmul 1.9x — tinygrad#1477 (grows sharply). PTX ordering worse than CUDA — tinygrad#4931 (grows). Francis Lam's 4 missing GEMM features, none implemented — tinygrad#6928 (grows). Each sample monotonically accumulates.
- **Shape:** **divergent**. Evidence against "codegen is adequate" grows with every sample. No reversals.
- **Edge →** is this fixable within the current IR?

## H₂: the IR can't express what competitive kernels need

- **Null:** the IR is expressive enough; the compiler just needs better passes
- **Perturbation:** enumerate IR ops vs GEMM requirements from tinygrad#6928
- **Trajectory:** bank conflict avoidance — could be index rewriting, no new ops needed (e-value flat). Register pressure — could be post-linearization liveness analysis (flat). Instruction scheduling — better linearizer over existing DAG (flat). Async copies (cp.async) — truly missing, no IR op (e-value jumps). Software pipelining — truly missing, loop model can't express it (jumps again). Then stabilizes — no further missing features found.
- **Shape:** **convergent**. The trajectory rises on async/pipelining, then settles. Partial confirmation: some gaps are IR limitations, others are missing analyses over the existing IR.
- **Kill condition:** none — hypothesis partially confirmed, partially refined.
- **Edge →** two open experiments remain.

---

## Graph state

| Node | Status | Shape |
|------|--------|-------|
| H₀ | killed | divergent |
| H₁ | refined | oscillatory |
| H₁ₐ | killed | divergent (against) |
| H₁ᵦ | confirmed | divergent |
| H₂ | partial | convergent |
| H₃ₐ | confirmed | divergent |
| H₃ᵦ | confirmed | divergent |
| H₃ᵧ | confirmed | divergent |
| H₄ₐ | confirmed (softened) | convergent |
| H₄ᵦ | refuted | divergent (against) |
| H₅ₐ | confirmed | convergent |
| H₅ᵦ | confirmed (softened) | convergent |
| H₅ᵧ | confirmed | convergent |
| H₆ | partial | convergent |
| H₇ | **open** | — |
| H₈ | **open** | — |
| H₉ | **confirmed** | convergent |
| H₁₀ | **open** | — |
| H₁₁ | refined | oscillatory |
| H₁₁ₐ | confirmed | convergent |
| H₁₂ | **confirmed** | **divergent** |
| H₁₃ | **refined** | convergent (against) |
| H₁₄ | **open** | — |
| H₁₅ | **confirmed** | divergent |
| H₁₆ | **confirmed safe** | convergent |
| H₁₇ | **confirmed (benign)** | convergent |
| H₁₈ | **confirmed** | divergent |

## Frontier edge 1: TESTED — linearizer reordering

**Experiment:** Change LOAD priority from -1 to 0 in `codegen/late/linearizer.py` line 29. This interleaves loads with compute instead of clustering all loads first.

**Result (vs_torch, original → patched):**
```
gemm_1024    1.81x → 0.89x   ✓ beats torch
permute      2.96x → 1.39x   ✓ 2.1x faster
add_4096     1.76x → 1.26x   ✓ 1.4x faster
mul_sum      1.51x → 1.07x   ✓ 1.4x faster
layernorm    5.32x → 3.70x   ✓ 1.4x faster
gemm_256     5.02x → 3.93x   ✓ 1.3x faster
exp_2048     2.13x → 1.89x   ✓ 1.1x faster
relu_4096    1.20x → 1.14x   ✓ slight
softmax      5.85x → 5.31x   ✓ slight
sum_4096     1.97x → 2.06x   ✗ slight regression
matvec       2.25x → 2.37x   ✗ slight regression
```

**Median vs_torch: 2.13x → 1.89x.** Nine of eleven workloads improved. gemm_1024 crossed parity.

**Classification:** **divergent** — strong evidence that instruction scheduling is a major component of the gap. A one-character change to the linearizer's priority table closed 20-50% of the gap on most workloads. The missing-analysis explanation dominates over the IR-limitation explanation for single-kernel ops.

**Confidence update:** elementwise gap explanation moves from 60% to 85%. The linearizer is a proven bottleneck. A proper latency-aware scheduler (not just priority 0 instead of -1) would likely close more.

**What this doesn't explain:** softmax (5.31x) and layernorm (3.70x) improved but remain far from torch. These compound ops need more than instruction reordering — they need either specialized dispatch or IR extensions for the fused implementations torch uses.

## H₃ₐ: launch/scheduling overhead dominates compound ops

- **Null:** GPU kernel compute is the bottleneck
- **Perturbation:** DEBUG=2 per-kernel GPU timing vs wall-clock for softmax and layernorm
- **Trajectory:** Softmax GPU compute = 48 μs (1.7% of 2800 μs wall time). Layernorm GPU compute = 607 μs (13.6% of 4476 μs). 86–98% of time is host-side: Python scheduling (~600 μs/kernel), command buffer encoding, synchronization.
- **Shape:** **divergent**. Every measurement monotonically confirms: kernel compute is not the bottleneck for compound ops.
- **Kill condition:** none — confirmed. Even infinitely fast kernels save only 48 μs out of 2800 μs for softmax.
- **Caveat:** DEBUG=2 adds instrumentation overhead; benchmark without DEBUG measured 1.36ms not 2.8ms. Directional conclusion holds but absolute numbers should use non-instrumented timing.
- **Provenance:** subagent (opus), codex round 1 validated

## H₃ᵦ: reduction kernels are scheduling-inert

- **Null:** reduction kernels have reorderable instructions like GEMM
- **Perturbation:** instruction count and dependency analysis on generated kernel source
- **Trajectory:** Max-reduction loop body: 4 instructions, 0 independent — strict chain (load → compare → select → store). Sum-exp loop: 6 instructions, 0 independent — strict chain (load → sub → mul → exp2 → add → store). GEMM loop: 58 instructions, 24 independent (8 loads + 16 WMMAs).
- **Shape:** **divergent**. Reduction kernels have zero instruction-level parallelism per iteration. There is literally nothing for a scheduler to reorder.
- **Kill condition:** none — confirmed. Explains why linearizer fix (LOAD priority -1→0) improved GEMM 2x but barely touched softmax/layernorm.
- **Provenance:** subagent (opus), no codex round needed (structural analysis, not empirical claim)

## H₃ᵧ: PyTorch dispatches vendor primitives

- **Null:** PyTorch generates similar multi-kernel code
- **Perturbation:** read PyTorch MPS source code on GitHub
- **Trajectory:**
  - Softmax: single call to `[mpsGraph softMaxWithTensor:axis:name:]` — Apple's closed-source MPSGraph primitive. Likely single-pass online softmax or hardware-tuned Metal shader. Data stays in registers/shared memory, zero intermediate global writes.
  - Layernorm: hand-written Metal kernel (`LayerNorm.metal`) — fused two-pass within one dispatch, uses `simd_sum` for warp-level reduction, 4-wide vectorized reads, `metal::precise::rsqrt`.
  - Both: 1 kernel dispatch vs tinygrad's 3.
- **Shape:** **divergent**. Source code confirms PyTorch uses fundamentally different implementations for both ops.
- **Kill condition:** none — confirmed. No per-kernel compiler optimization in tinygrad's current architecture can match a vendor primitive that fuses the entire operation into a single dispatch with register-local intermediates.
- **Provenance:** subagent (opus), codex round 1 validated. Sources: [SoftMax.mm](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/SoftMax.mm), [LayerNorm.metal](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/LayerNorm.metal)

## Three complementary layers

The three hypotheses are not competing — they explain the same gap at different levels:

1. **H₃ᵦ** (bottom): the kernels themselves can't benefit from scheduling — no ILP in reductions
2. **H₃ₐ** (middle): kernel compute is 2–14% of wall time anyway — host overhead dominates
3. **H₃ᵧ** (top): torch bypasses both problems — vendor primitives do the entire op in one dispatch with an optimized algorithm

This means the compound-op gap has two independent fixes:
- **Reduce host overhead** (scheduling, command buffer, Python glue) — helps all multi-kernel ops
- **Dispatch to vendor primitives** for known compound patterns — eliminates the gap for softmax/layernorm specifically

The linearizer fix is irrelevant for compound ops because it operates at layer 1 (kernel instruction ordering) while the bottleneck is at layers 2 and 3.

## H₄ₐ: host overhead is Metal sync + Python realize path

- **Null:** Python scheduling is the dominant host overhead
- **Perturbation:** profile JIT-warmed softmax: JIT dispatch path, Metal command buffer round-trip, GPU compute
- **Trajectory:** JIT-warmed wall time ~600 μs. GPU compute ~55 μs. Python JIT dispatch ~80 μs. Residual ~300-600 μs attributed to Metal command buffer lifecycle and sync. Non-JIT: ~1.2ms (JIT eliminates ~600 μs of scheduling).
- **Shape:** **convergent** — evidence rises (overhead is real) then stabilizes (specific source of residual needs further isolation).
- **Codex correction:** "Metal sync barrier" is too specific. The residual could include command encoding, allocation, cache effects, or tinygrad runtime bookkeeping. The tighter claim: host-side overhead around launching/synchronizing small GPU work dominates, with GPU compute only ~55 μs.
- **Provenance:** subagent (opus), codex round 1 softened attribution

## H₄ᵦ: single-kernel gap is NOT kernel quality — REFUTED the codegen hypothesis

- **Null:** the remaining single-kernel gap (1.4-5x vs torch) is codegen quality
- **Perturbation:** dump generated kernels with DEBUG=4, profile GPU-only time via Metal timestamps, compare against torch wall time, test TC=0 for small GEMM, profile with cProfile
- **Trajectory:** tinygrad GPU kernel times are competitive with or faster than torch for all four tested workloads:
  - gemm_256: 67 μs GPU (506 GFLOPS), uses WMMA appropriately
  - exp_2048: 100 μs GPU, hardware exp2 intrinsic, float4 vectorized
  - permute: 28 μs GPU, achieves 286 GB/s despite naive access pattern
  - matvec: ~260 μs GPU, memory-bound, close to theoretical
  - All gaps come from Python-side realize overhead: 600-1000 μs per call
  - With TinyJit, all drop to 200-300 μs wall time
- **Shape:** **divergent against the hypothesis.** Every sample confirms: GPU kernels are not the bottleneck for these workloads at these sizes.
- **Kill condition:** the earlier diagnosis ("codegen quality is the root cause across all workloads") was too broad. For single-kernel ops at these sizes, codegen is competitive. The gap is eager realize overhead.
- **Codex correction:** torch GPU times were inferred from wall time, not device-profiled — comparison is uneven. Kernel quality may still matter for other sizes, reductions, fusion boundaries, or larger models. TinyJit changes more than scheduling (graph execution, launch behavior), so "unified_rewrite explains the gap" overclaims.
- **Provenance:** subagent (opus), codex round 1 narrowed scope

## Revised diagnosis

The investigation has converged on a two-part finding:

**Single-kernel ops (gemm, add, relu, exp, permute):** Generated Metal kernels are competitive. The wall-time gap is dominated by Python-side eager realize overhead (~600-1000 μs vs torch's C++ dispatch at ~10-25 μs). TinyJit closes most of the gap. The linearizer improvement (LOAD priority -1→0) helped GEMM specifically because it has high ILP — but the bigger win is just using JIT. The earlier "codegen quality" diagnosis was wrong for these workloads at these sizes.

**Compound-op ops (softmax, layernorm):** Three complementary layers:
1. Reduction kernels have zero ILP — nothing for a scheduler to improve (H₃ᵦ)
2. GPU compute is ~55 μs, dominated by host overhead per dispatch (H₃ₐ/H₄ₐ)
3. Torch uses vendor primitives (MPSGraph softmax, hand-written LayerNorm.metal) that fuse the entire operation into one dispatch with register-local intermediates (H₃ᵧ)

**Common thread:** tinygrad's eager realize path is the bottleneck for small-to-medium workloads on Metal. For large workloads (big GEMM, full model inference), GPU compute time dominates and kernel quality matters more — that's where the linearizer fix and IR extensions become relevant.

## H₅ₐ: realize overhead is known, accepted as architectural cost

- **Null:** realize overhead is an unrecognized bug
- **Perturbation:** search tinygrad issue tracker for scheduling overhead discussions
- **Trajectory:** issue #7698 (open): "Tensor(numpy).realize() takes 0.85ms to schedule." Schedule cache merged Dec 2025 (PR #13529). TinyJit bypasses Python entirely on replay. A 29% scheduling speedup PR (#15491) was rejected as "AI SLOP." Moving to compiled language: never proposed. Pure Python is an explicit design value.
- **Shape:** **convergent.** The overhead is known and deliberately accepted. The project amortizes it (SCACHE, TinyJit) rather than eliminating it. No open effort to close the gap for novel-graph workloads.
- **Codex:** not needed — factual findings from issue tracker, no empirical claims to overclaim.
- **Provenance:** subagent (opus), issue #7698, PR #13529, PR #15491

## H₅ᵦ: large-workload kernel quality is already competitive

- **Null:** codegen quality matters more at larger scale
- **Perturbation:** benchmark gemm_4096 and softmax_256x8192 with JIT, default vs patched linearizer, vs torch
- **Trajectory:** gemm_4096: tinygrad 9.15ms vs torch 9.41ms — roughly competitive (2.8% delta, within noise). Linearizer patch: no material change. Softmax_256x8192: linearizer patch hurts (median 0.49→0.79ms, bimodal). Torch wins at 0.35ms median but all sub-millisecond.
- **Shape:** **convergent.** For large square GEMM, kernel quality appears competitive. The linearizer fix is workload-dependent — helps high-ILP kernels, hurts serial-chain reductions. At large scale, JIT amortizes realize overhead and the remaining gap is single-digit percentages.
- **Codex corrections:** 9.15 vs 9.41ms is within noise — "roughly competitive," not "beats." The linearizer patch is a knob, not a universal fix. "Competitive" only tested on one large GEMM shape — overbroad as a general statement.
- **Provenance:** subagent (opus), codex round 1 softened

## H₅ᵧ: vendor dispatch is philosophically rejected

- **Null:** tinygrad could dispatch to MPSGraph/MPS primitives as a stopgap
- **Perturbation:** search codebase and issue tracker for vendor library usage/discussion
- **Trajectory:** zero vendor library calls in the entire codebase. geohot (issue #429): "While we could make things faster today with a BLAS library, this gives up one of the long term gains possible with tinygrad." The compiler alternative is PCONTIG — fuse multi-reduce into single kernels. Actively developed (PRs #12884, #12899, #14266) but not stable enough to enable by default.
- **Shape:** **convergent.** Clear architectural position: compiler replaces vendor libraries. The compound-op gap is accepted as temporary cost of the thesis. PCONTIG is the intended fix but current fused codegen regresses (our PCONTIG=99 experiment at depth 2).
- **Codex corrections:** PCONTIG is one mechanism, not the full alternative. PCONTIG=99 regression is shape/config-specific, not proof that fused codegen is broadly broken.
- **Provenance:** subagent (opus), codex round 1 softened. Source: issue #429, PR #13529

## Depth 5 summary: the frontier is narrowing

| Symptom | Root cause | Status |
|---------|-----------|--------|
| Single-kernel gap (small/medium) | Python realize overhead (~600-1000 μs) | Known, accepted, amortized by JIT/SCACHE |
| Single-kernel gap (large) | None — kernel quality is competitive | Closed |
| Compound-op gap | 3 layers: no ILP in reductions + host overhead per dispatch + vendor primitive vs compiler | Known, PCONTIG is the compiler path |
| Linearizer scheduling | Naive priority sort, helps GEMM, hurts reductions | Workload-dependent knob, not universal fix |

## H₆: the fused codegen gap is algorithm selection, not instruction quality

- **Null:** PCONTIG fusion fails because the IR can't express fused compound kernels
- **Perturbation:** check whether the IR has the primitives for a two-pass fused layernorm; read PyTorch's `LayerNorm.metal` for the target algorithm; search tinygrad's tracker for related work
- **Trajectory:**
  - IR check: 3 of 4 primitives exist (sequential loops, barrier, shared memory). Missing: `simd_sum` warp-level reduction — no op, no renderer path.
  - Algorithm comparison: PyTorch's kernel computes sum and sum_sq in one pass with `simd_sum`, barriers, then re-reads cached data for normalization. tinygrad's PCONTIG=99 fusion concatenates three serial-chain kernels into one bloated kernel — same algorithm, worse occupancy.
  - Tracker: issue #4927 (Qazalin) — "mean/std, softmax one kernel" — considered "mostly solved by PCONTIG." PCONTIG PRs active through Jan 2026. `simd_sum` exists in `extra/thunder/` Metal kernels but is NOT connected to the compiler. Flash attention (#13692, #13697) is the same class of problem, open, assigned to wozeparrot.
- **Shape:** **convergent.** Two specific gaps identified, both tractable:
  1. `simd_sum` as a new op + renderer rule (warp-level reduction)
  2. Algorithm-selection pattern matcher: recognize mean-variance-normalize and emit a fused two-pass kernel with two accumulators, not a concatenation of three serial loops
- **Kill condition:** none — the hypothesis refines rather than kills. PCONTIG is the right mechanism but it needs two additions to produce competitive fused kernels.
- **Codex:** not yet filtered — factual findings from source code and issue tracker
- **Provenance:** subagent (opus), issue #4927, PyTorch `LayerNorm.metal` source

## Diagnostic findings

The investigation has identified root causes for the performance gap at 80-99% confidence:

| Finding | Confidence | Mode |
|---------|-----------|------|
| Small-op gap is realize overhead, not kernel quality | 90% | induction (benchmarked, profiled) |
| Large-op kernel quality is competitive | 85% | induction (gemm_4096 ≈ torch, codex softened to "roughly competitive") |
| Compound-op gap is 3 layers (no ILP + host overhead + vendor primitives) | 95% | induction (all three tested independently) |
| PCONTIG fusion regresses because it concatenates, not restructures | 90% | induction (PCONTIG=99 measured) |
| Two missing pieces: simd_sum op + algorithm-selection pattern matcher | 80% | abduction (identified from code reading, not tested) |
| Realize overhead is known and deliberately accepted | 99% | deduction (issue #7698, maintainer quotes) |
| Vendor dispatch is philosophically rejected | 99% | deduction (geohot quote, issue #429) |
| Matvec loop ordering is the dominant decode bottleneck (F16) | 85% | deduction (Metal kernel dump) + analogy (CPU PR #14630) |
| **Lazy Q6K dequant chains are the dominant model-level bottleneck** | **99%** | **induction (realizing dequant chains: 10.5→130 tok/s, 11→325 GB/s, 12.4x speedup)** |
| F16 model inference matches op-level prediction (2.8x) | 95% | induction (3B F16: 173 GB/s, consistent with matvec diagnosis) |

The diagnosis is strong. The performance gap is not closed. The graph is open.

## PR status

| PR | Title | Status | Gap addressed |
|----|-------|--------|---------------|
| [#16094](https://github.com/tinygrad/tinygrad/pull/16094) | llm: contiguous weights + rollout prune for quantized GGUF inference | **Open** | Lazy dequant 8-14x regression (H₁₂) |
| [#16085](https://github.com/tinygrad/tinygrad/pull/16085) | onnx: deduplicate simple proto parsers | **Merged** | Line budget for WARP_REDUCE |
| [#16072](https://github.com/tinygrad/tinygrad/pull/16072) | increase matvec MV_ROWS_PER_THREAD from 4 to 16 | **Open** | Matvec loop ordering (decode) |
| [#16070](https://github.com/tinygrad/tinygrad/pull/16070) | add Ops.WARP_REDUCE for GROUPTOP reductions | **Draft** | Compound-op reduction (simd_sum) |

## Open frontier

The gap persists. These are hypotheses, not implementation tasks — each needs a perturbation and a trajectory classification before it can be confirmed or killed.

### H₇: matvec fix closes the decode gap on CUDA

- **Null:** the matvec loop ordering bug is Metal-specific; CUDA codegen produces correct ordering
- **Perturbation:** run `extract.py` on the Windows/CUDA machine for the same 1×4096 × 4096×4096 shape. Dump the PTX. Check whether the inner loop has stride-1 or stride-N access on the weight matrix.
- **Predicted trajectory:** if same stride bug → divergent (confirms cross-backend root cause). If correct ordering → convergent against (Metal-specific, CUDA already handles it).
- **Why this matters:** the matvec experiment confirmed the bug on CPU and Metal. CUDA is the third backend and the one with the most competitive pressure (cuBLAS). If CUDA also has the bug, the fix is higher priority upstream. If CUDA doesn't, the scheduler may already have a backend-specific path we missed.
- **Resources:** Windows machine with NVIDIA GPU.

### H₈: WARP_REDUCE closes the reduction gap on CUDA

- **Null:** `__shfl_down_sync()` is already used or GROUPTOP reductions are already fast on CUDA
- **Perturbation:** benchmark softmax and layernorm on CUDA with and without the WARP_REDUCE draft PR. Compare to PyTorch CUDA.
- **Predicted trajectory:** if CUDA reductions are already fast → convergent against (Metal-specific gap). If CUDA reductions are slow and WARP_REDUCE helps → divergent (confirms the hypothesis across backends).
- **Why this matters:** the compound-op gap (H₁ₐ through H₃ᵧ) was only measured on Metal. Metal's MPSGraph primitives are a confound — CUDA's cuDNN is a different vendor primitive surface. The gap decomposition may look completely different.
- **Resources:** Windows machine with NVIDIA GPU.

### H₉: the performance gap has a different shape on CUDA — CONFIRMED (same shape)

- **Null:** the Metal gap decomposition (realize overhead + vendor primitives + codegen) transfers directly to CUDA
- **Perturbation:** ran the contiguous+prune fix on RTX 5000 Ada (Windows, NV backend). LLaMA 1B Q6_K.
- **Trajectory:**

  | Backend | Before | After | Speedup |
  |---------|--------|-------|---------|
  | Metal (M5 Max) | 10.5 tok/s | 147 tok/s | 14.0x |
  | NV (RTX 5000 Ada) | 10.4 tok/s | 85.8 tok/s | 8.2x |

  Bit-exact output on both backends. The lazy dequant regression is cross-backend — same root cause (fused dequant chains replayed every token), same mechanism fixes it (contiguous + prune). The magnitude differs (14x vs 8.2x) because Metal and CUDA handle the fused scalar-byte-load kernel differently.
- **Shape:** **convergent.** The gap decomposition transfers. The lazy dequant finding is not Metal-specific.
- **Mode:** induction. Confidence: 95%.

### H₁₀: algorithm-selection pattern matcher closes the compound-op gap

- **Null:** PCONTIG is sufficient once the fused kernel quality improves
- **Perturbation:** implement a pattern matcher that recognizes softmax/layernorm and emits fused two-pass kernels with two accumulators (not serial-chain concatenation). Test on both Metal and CUDA.
- **Why this is still a hypothesis:** H₆ identified the missing pieces (simd_sum + algorithm selection) but neither has been tested. The claim "these two additions would produce competitive fused kernels" is abductive at 80% confidence. It needs measurement.
- **Resources:** both machines.

### H₁₁: full model inference validates op-level findings — TESTED

- **Null:** op-level findings predict model-level performance
- **Perturbation:** benchmark LLaMA inference on M5 Max (Metal 4, 48GB) using tinygrad master vs llama.cpp, two configs: LLaMA 3.2 1B Q6_K and LLaMA 3.2 3B F16.
- **Trajectory:**

  | Config | llama.cpp | tinygrad | Ratio | tinygrad BW |
  |--------|-----------|----------|-------|-------------|
  | 1B Q6_K decode | 341 tok/s | 10.5 tok/s | **32.5x** | 11 GB/s |
  | 3B F16 decode | 74.5 tok/s | 26.5 tok/s | **2.8x** | 173 GB/s |

  The op-level investigation predicted a 2-3x gap from matvec loop ordering. The F16 result (2.8x, 173 GB/s) is consistent with that prediction. The Q6_K result (32x, 11 GB/s) is not — it's an order of magnitude worse.

  DEBUG=2 breakdown per token (1B Q6_K, JIT-warmed):
  - 4 MetalGraph batches: 8.7ms + 22.6ms + 45.4ms + 17.5ms = 94ms GPU
  - Wall time: 96ms — JIT eliminates host overhead
  - Q6_K dequant+matvec kernels (`r_4_8_2_8_4_8_2_2_2_32`): **3 GB/s**, 1321μs each
  - Non-quantized kernels (`r_2048_16_2_2_2_2_32`): **929 GB/s**, 129μs each

  The bottleneck is quantized dequantization fused into matmul. tinygrad's Q6_K dequant (gguf.py:69-73) creates bitwise ops (bitcast, shift, or, subtract, multiply by scales) that fuse into the matmul as elementwise producers. The resulting kernel has 10 nested loop dimensions and achieves 0.6% of peak bandwidth. llama.cpp uses hand-written Metal kernels per quantization type with SIMD bit-unpacking.

- **Shape:** **oscillatory.** The op-level prediction is correct for F16 (convergent) but wildly wrong for quantized inference (divergent). Two modes: the gap depends on whether the model is quantized.
- **Kill condition:** the hypothesis is too coarse. Split into: H₁₁ₐ (F16: op-level findings transfer) and H₁₂ (quantized: dequant codegen is the dominant bottleneck).
- **Provenance:** direct measurement, M5 Max, tinygrad master @ 9a6f7f757, llama.cpp ad0922465
- **Mode:** induction. Confidence: 95%.

### H₁₁ₐ: F16 model inference — op-level findings transfer — CONFIRMED

- **Null:** op-level findings predict F16 model-level performance
- **Result:** 2.8x gap at 173 GB/s (35% of M5 Max peak). Consistent with the matvec investigation's prediction of 2-3x from loop ordering + missing vectorization.
- **Shape:** **convergent.** The op-level diagnosis holds for F16.
- **Remaining gap attribution (F16):** matvec loop ordering (H₁ₘ from sibling graph), missing SIMD reduction, scalar loads. The existing PRs (#16072 matvec, #16070 WARP_REDUCE) target this gap.

### H₁₂: quantized dequant codegen is the dominant model-level bottleneck — CONFIRMED

- **Null:** quantized inference achieves similar bandwidth fraction as F16
- **Perturbation 1:** benchmark Q6_K model — dequant+matvec kernels achieve 3 GB/s (2% of peak). Kernel dump: 328-line monster with 271 scalar byte loads, 4 nested loops, no shared memory, no tiling.
- **Perturbation 2:** force-realize Q6K dequant chains to F16 before JIT warmup (`p.replace(p.contiguous().realize())`).
- **Trajectory:**

  | Config | tok/s | Bandwidth | Weight reads/tok |
  |--------|-------|-----------|-----------------|
  | Lazy Q6K (default) | 10.5 | 11 GB/s | 1.0 GB (Q6K bytes) |
  | **Realized F16** | **130** | **325 GB/s** | 3.0 GB (F16) |
  | llama.cpp Q6K | 341 | ~340 GB/s | 1.0 GB (Q6K bytes) |

  **12.4x speedup** from realizing dequant chains. Bandwidth goes from 2% to 65% of M5 Max peak.

- **Shape:** **divergent.** Every measurement monotonically confirms: the lazy Q6K dequant chain is the bottleneck. The model doesn't pre-compute dequantization — it re-executes the 130-UOp dequant graph on every forward pass, producing 328-line Metal kernels with scalar byte loads.
- **Root cause:** the GGUF loader (`gguf.py:69-73`) returns lazy tensors. The model stores these as parameters without realizing them. `nn.state.get_parameters()` reports `dtype=half` (the output type of the lazy chain), masking the fact that the underlying computation is still reading raw Q6K bytes.
- **Remaining gap after fix:** 130 tok/s vs llama.cpp's 341 tok/s = 2.6x. This is a **memory footprint gap**: tinygrad reads 3.0 GB of F16 data per token; llama.cpp reads 1.0 GB of Q6K data with hand-written SIMD dequant kernels. At similar bandwidth (~325 vs ~340 GB/s), the 3x data ratio explains the 2.6x speed ratio.
- **Two fix paths:**
  1. **Realize on load** (easy, +2x memory): add `.contiguous().realize()` in the GGUF loader or model loading path. Closes the 12.4x gap immediately. Trades memory for speed.
  2. **Native Q6K matmul kernels** (hard, optimal): hand-written or pattern-matched kernels that read Q6K blocks with vectorized loads and dequant inline with SIMD. Matches llama.cpp's approach. Closes the full 32x gap.
- **Provenance:** direct measurement, M5 Max, tinygrad master @ 9a6f7f757. Kernel dump in `/tmp/q6k_debug.txt`.
- **Provenance check:** geohot deliberately flipped the default from `REALIZE=1` to `REALIZE=0` in PR #15144 (March 5, 2026). Commit message: "make realize not the default." No benchmarks, no discussion, empty PR body. Bundled with an unrelated render depth bug fix. The code comment on line 384 says: "we shouldn't need this, but for now it's faster." The team treats the lazy-dequant-every-pass as a scheduler/JIT deficiency that should be fixed at a lower level, not papered over with realize. Flipping the default back would be rejected on principle.
- **Codex review:** risk is medium, not low. The memory contract changes (Q6K 1GB → F16 2.5GB). Models chosen to fit at quantized size would OOM. Missing: benchmarks across model sizes, quant formats, peak memory during load, backend variance.
- **Mode:** induction. Confidence: 99%.
- **Edge →** the real fix is at the scheduler level: why does the JIT re-execute 130-UOp dequant chains on every forward pass instead of caching the materialized result?

### H₁₃: the scheduler can detect and cache repeated dequant subgraphs — TESTED

- **Null:** the scheduler already handles repeated subgraphs efficiently
- **Perturbation 1:** trace the JIT capture path (`jit.py:260-304`).
  - `prune_linear` (`jit.py:15-23`) exists and does exactly this: separates kernels into `kept` (touch input buffers) and `onetime` (don't). The `onetime` kernels run once during capture.
  - `TinyJit(self.forward)` at `model.py:308` does NOT pass `prune=True`. The flag defaults to `False`.
- **Perturbation 2:** patch `prune=True` on both `prefill_jit` and `rollout_jit`, benchmark.
  - **Result: no improvement.** Still 10.5 tok/s, 11 GB/s.
- **Why prune fails:** the dequant ops are **fused into the matmul kernel** at the scheduler level. The scheduler sees `(raw_bytes → bitops → scale → cast) → matmul(activations)` and fuses the entire chain into one kernel. This fused kernel touches both weight buffers (GGUF raw bytes) and input buffers (activations). Since it touches input buffers, `prune_linear` classifies it as "kept," not "onetime." Prune operates at kernel granularity — it can't split a fused kernel.
- **Shape:** **convergent against the hypothesis.** `prune_linear` is the right mechanism but operates at the wrong granularity. The fix needs to happen before fusion, not after.
- **The real fix location:** the scheduler's fusion pass. Currently, the scheduler fuses elementwise producers into consumer reduces unconditionally. For quantized weights, this means "dequant bytes → matmul" becomes one kernel. The scheduler would need to recognize that the dequant subgraph is invariant (its inputs are constant buffers) and schedule it as a separate kernel that writes to an intermediate F16 buffer. Then the matmul reads from that buffer, and prune correctly identifies the dequant kernel as "onetime."
- **This is a scheduler invariance analysis** — detecting "this subgraph's inputs are all non-input buffers, so its output is constant across JIT replays." The scheduler doesn't currently have this concept.
- **Provenance:** code trace of `jit.py:15-23,240,286-289`, `model.py:308-309`. Benchmark with `prune=True` on M5 Max.
- **Provenance check:**
  - PR #15082 (Mar 3, geohot): added `contiguous()` + `realize()` behind `REALIZE=1` flag. Default was 1.
  - PR #15144 (Mar 5, geohot): flipped default to `REALIZE=0`. No benchmarks, no discussion.
  - PR #15423 (Mar 23, nimlgen): added `prune_linear` to JIT, refactored to work on UOp linear. `prune=True` used in openpilot, ONNX benchmarks, tests — never in LLM model.
  - **The gap:** contiguous (fusion barrier) and prune (onetime detection) were built by different people 18 days apart for different purposes. Nobody combined them for GGUF weight loading. The contiguous was always paired with realize; prune didn't exist when geohot wrote the "we shouldn't need this" comment.
  - **The hack works:** `.contiguous()` on weights (no realize) + `prune=True` on JITs → 141 tok/s at 354 GB/s. Dequant runs once during JIT warmup, pruned from replay. No new scheduler concepts needed.
- **Mode:** deduction (code trace) + induction (benchmark). Confidence: 95%.

### H₁₄: native quantized matmul kernels close the remaining 2.6x gap

- **Null:** F16 realized weights are close enough to optimal
- **Evidence against the null:** after realizing to F16, tinygrad achieves 130 tok/s at 325 GB/s. llama.cpp achieves 341 tok/s at ~340 GB/s. The gap is 2.6x. Both achieve similar bandwidth (~325 vs ~340 GB/s), but tinygrad reads 3.0 GB of F16 per token while llama.cpp reads 1.0 GB of Q6K. The gap is memory footprint, not bandwidth efficiency.
- **Perturbation:** implement a pattern-matched or custom kernel that reads Q6K blocks with vectorized loads (uint4), dequants with SIMD, and accumulates in registers — matching llama.cpp's approach. Benchmark against both the lazy and realized paths.
- **Predicted trajectory:** if native Q6K matmul achieves >300 GB/s on the compressed format, throughput should be ~330 tok/s (1.0 GB × 330 GB/s). That would close the full 32x gap without the 2x memory penalty.
- **Why this matters:** this is the difference between "tinygrad can match llama.cpp if you have enough RAM" and "tinygrad matches llama.cpp." The memory tradeoff from REALIZE is a ceiling — native quant kernels remove it.
- **Resources:** both machines. Implementation effort: high (new kernel type or pattern matcher in codegen).

### H₁₅: prune misclassifies cache/state kernels as onetime — CONFIRMED

- **Null:** prune correctly identifies all kernels that must run on every token
- **Perturbation:** multi-turn test — 30 tokens from BOS, 20 from prefix reuse, 20 from divergent prompt [1,2,3]. Compare lazy baseline vs contiguous+prune hack.
- **Trajectory:** Turn 1 match, Turn 2 match, **Turn 3 FAIL**. First divergence at token 3. The hack produces `[1,2,3,13,2475,...]` while lazy/realize both produce `[1,2,3,16,13,...]`.
- **Isolation:** contiguous-only (no prune) passes all turns. Prune-only (no contiguous) passes all turns. The combination fails. Further isolation: **prune on rollout only passes all turns. Prune on prefill only fails turn 3.** The prefill JIT's prune misclassifies a cache-related kernel as onetime.
- **Shape:** **divergent.** Codex's prediction was correct — prune misclassification is a real risk on the prefill path.
- **Fix:** prune only on `rollout_jit`, not on `prefill_jit`. Rollout is T=1 steady-state decode — all pruned kernels are genuinely onetime weight dequants. Prefill handles variable-length prompts with cache state dependencies.
- **Performance with fix:** 138 tok/s at 343 GB/s (vs 141 with full prune). Negligible difference — decode dominates.
- **Provenance:** codex flagged this risk. Multi-turn test confirmed it. Isolation traced it to prefill prune specifically.
- **Mode:** induction. Confidence: 99%.

### H₁₆: contiguous+prune breaks non-LLaMA architectures (MoE, SSM, MLA) — CONFIRMED SAFE

- **Null:** the hack works for all model architectures in tinygrad's GGUF loader
- **Perturbation 1:** OLMoE 1B-7B Q4_K_M (MoE). Initial test showed non-deterministic baseline — three lazy runs produced different outputs. Fixed by seeding: `Tensor.manual_seed(42)` makes OLMoE deterministic. **Seeded lazy == seeded hack. PASS.**
- **Perturbation 2:** Qwen3.5 0.8B Q8_0 (SSM/recurrent). Has both `GatedDeltaNetBlock` (mutable `conv_state` + `recurrent_state`) and `TransformerBlock`. The recurrent state updates (`_attention` lines 271-277) use `.uop.store()` for in-place mutation — this is the mutable implicit state codex flagged. **Seeded lazy == seeded hack. Multi-turn no crash. PASS.**
- **Why SSM is safe:** the recurrent state updates depend on input `x` (which derives from tokens), so prune correctly classifies them as "kept." The state initialization (`_init_state`) runs before JIT capture (first forward call, cnt=0), not inside the captured graph.
- **Shape:** **convergent.** Both architectures pass seeded equivalence. The initial non-determinism was from `Tensor.rand_like(logits)` in the Gumbel-max sampling, not from the hack.
- **Mode:** induction. Confidence: 95%.

### H₁₇: the hack changes peak memory during warmup — CONFIRMED (benign)

- **Null:** peak memory equals REALIZE=1
- **Perturbation:** measure RSS at load, after warmup, and steady state for lazy, hack, and REALIZE=1.
- **Trajectory:**

  | Stage | Lazy | Hack | REALIZE=1 |
  |-------|------|------|-----------|
  | After load | 1086 MB | 2531 MB | 2640 MB |
  | After warmup | 1514 MB | 2598 MB | 2640 MB |
  | Steady state | 1514 MB | 2598 MB | 2640 MB |

- **Shape:** **convergent.** Hack peak ≈ REALIZE=1 peak (2598 vs 2640 MB). Both are ~1.7x the lazy peak. The hack does NOT create a worse peak than REALIZE=1 — codex's concern about coexisting raw+lazy+F16+KV was overstated.
- **Mode:** induction. Confidence: 95%.

### H₁₈: the hack works across quant formats — CONFIRMED

- **Null:** the speedup generalizes to all GGUF quantization types
- **Perturbation:** benchmark LLaMA 3.2 1B Q4_K_M with and without the hack (contiguous + rollout prune).
- **Trajectory:**

  | Format | Lazy | Hack | Speedup | Output match |
  |--------|------|------|---------|-------------|
  | Q6_K | 10.5 tok/s | 138 tok/s | 13.1x | Yes |
  | Q4_K_M | 36.0 tok/s | 136 tok/s | 3.8x | Yes |

  Q4_K_M has simpler dequant chains (fewer bitwise ops), so the lazy path is already 3.4x faster than Q6_K lazy. But both converge to ~136-138 tok/s with the hack — the ceiling is F16 matmul bandwidth, not dequant complexity.
- **Shape:** **divergent.** The hack works across quant formats. Speedup magnitude varies (3.8x–13.1x) based on dequant complexity, but the ceiling is the same.
- **Mode:** induction. Confidence: 95%.

## Reasoning modes per node

Each node used a composition of the [three modes](https://june.kim/modes-of-reason):

| Node | Abduction | Deduction | Induction |
|------|-----------|-----------|-----------|
| H₀ | — | — | benchmark (run, measure) |
| H₁ | "encoding is the cause" | codex traced consequences | NOOPT/heur/BEAM benchmark |
| H₁ₐ | "fusion explains multi-kernel gap" | "if fusion helps, PCONTIG=99 should be faster" | ran PCONTIG=99, measured |
| H₁ᵦ | "codegen quality is the cause" | read linearizer code, derived limitations | #1477 barrier result, #4931 PTX ordering |
| H₂ | "IR can't express choreography" | enumerated IR ops vs requirements | partial — needs frontier experiments |
| H₇ | "matvec bug is cross-backend" | — | needs CUDA kernel dump |
| H₈ | "WARP_REDUCE helps CUDA reductions" | — | needs CUDA benchmark |
| H₉ | "Metal gap decomposition transfers to CUDA" | — | **confirmed: 10.4→85.8 tok/s on RTX 5000 Ada, 8.2x** |
| H₁₀ | "algorithm selection closes compound-op gap" | H₆ identified the pieces | needs implementation + measurement |
| H₁₁ | "op-level predicts model-level" | matvec experiment showed it doesn't always | **measured: oscillatory (F16 yes, quant no)** |
| H₁₁ₐ | "F16 model-level matches op-level" | — | **confirmed: 2.8x gap, 173 GB/s** |
| H₁₂ | "quant dequant codegen is dominant" | — | **confirmed: 32x gap, 3 GB/s per kernel, REALIZE=1 gives 12.4x** |
| H₁₃ | "scheduler can cache dequant subgraphs" | prune_linear exists but operates post-fusion | **tested: prune=True, no effect — dequant fused into matmul** |
| H₁₃' | "contiguous+prune = auto-realize" | contiguous breaks fusion, prune makes onetime | **tested: 141 tok/s, 354 GB/s** |
| H₁₄ | "native Q6K matmul closes remaining 2.6x" | — | needs implementation |
| H₁₅ | "prune misclassifies cache/state kernels" | codex flagged | **confirmed: prefill prune breaks turn 3. Fix: rollout-only prune** |
| H₁₆ | "hack breaks MoE/SSM/MLA architectures" | codex flagged | **confirmed safe: seeded MoE + SSM both pass** |
| H₁₇ | "hack changes peak memory" | codex flagged | **confirmed benign: hack ≈ REALIZE=1 peak** |
| H₁₈ | "hack works across quant formats" | codex flagged | **confirmed: Q4_K_M 3.8x, Q6_K 13.1x** |

Confidence tracks which mode produced the claim: deduction (read the code) → 99%. Induction (ran the experiment) → 95%. Abduction (proposed from observation) → 60–85%.
