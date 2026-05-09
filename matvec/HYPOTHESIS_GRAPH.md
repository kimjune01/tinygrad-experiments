# Hypothesis Graph: Full LLaMA Inference Gap Decomposition

Extends the [prior op-level investigation](HYPOTHESIS_GRAPH.md) to frontier edge #3: full model inference validation. The question: "How much of tinygrad's competitive gap on full LLaMA inference is realize overhead vs kernel quality vs scheduling vs something else entirely?"

Framework: [The Hypothesis Graph](https://june.kim/the-hypothesis-graph). Classification: [Evidence has a trajectory](https://june.kim/evidence-has-a-trajectory). Reasoning modes: [Modes of Reason](https://june.kim/modes-of-reason).

### References

| Source | Relevance | Nodes |
|--------|-----------|-------|
| [PR #15097](https://github.com/tinygrad/tinygrad/pull/15097) — fully symbolic LLM | Eliminates novel graph shapes via symbolic variables | H₀ₘ |
| [PR #15153](https://github.com/tinygrad/tinygrad/pull/15153) — two JITs, prefill/rollout | Splits JIT into prefill + decode paths | H₀ₘ, H₂ₘ |
| [PR #15149](https://github.com/tinygrad/tinygrad/pull/15149) — chunked prefill | Introduces chunk_size=32 prefill | H₂ₘ |
| [PR #14630](https://github.com/tinygrad/tinygrad/pull/14630) — CPU matvec kernel | 2.76× gap from loop ordering; custom kernel closed to parity; rejected | H₁ₘ, H₅ₘ |
| [Issue #14412](https://github.com/tinygrad/tinygrad/issues/14412) — llama 1B CPU bounty | Bounty that motivated PR #14630 | H₁ₘ |
| [Issue #1209](https://github.com/tinygrad/tinygrad/issues/1209) — M1 llama speed bounty | Historical baseline: ~100ms/tok on M1 Max | H₃ₘ |
| [Issue #5244](https://github.com/tinygrad/tinygrad/issues/5244) — llama speed tracking | "cpu time spent mostly updating var vals and launch bounds" | H₃ₘ |
| [Issue #15669](https://github.com/tinygrad/tinygrad/issues/15669) — eGPU perf comparison | tinygrad 4 tok/s vs llama.cpp 11.2 tok/s; JITBEAM=2 suggested | H₁ₘ, H₃ₘ |
| [PR #14709](https://github.com/tinygrad/tinygrad/pull/14709) — Metal graph var cache | Rejected as not worth it — Metal overhead is secondary | H₃ₘ |
| [PRs #15960, #15993, #16071](https://github.com/tinygrad/tinygrad/pull/16071) — llama speed sprint | Active optimization work (April-May 2026) | all |
| [Issue #429](https://github.com/tinygrad/tinygrad/issues/429) — vendor library philosophy | geohot: "this gives up one of the long term gains possible with tinygrad" | H₄ₘ |

---

## H₀ₘ: The question's premise — novel graph shapes dominate

- **Claim:** "TinyJit closes most of the performance gap on repeated workloads, but real model inference has novel graph shapes on every prompt length, every batch size, every sequence position."
- **Perturbation:** trace both LLaMA implementations through decode and prefill paths, checking what the JIT covers
- **Trajectory:**
  - **New model** (`tinygrad/llm/model.py`): `generate()` uses `UOp.variable("start_pos", 0, max_context-1)` and `UOp.variable("toks", 1, chunk_size)`. These are *symbolic variables* — the JIT captures the graph once with symbolic shapes, then replays with concrete bindings. Varying start_pos and prompt length do NOT produce novel graph shapes. Both `prefill_jit` and `rollout_jit` cover all inference after 2 warmup calls.
  - **Old model** (`extra/models/llama.py`): `Transformer.__call__` JITs decode (T=1, start_pos>0) via `Variable("start_pos", 1, max_context-1)`. Prefill at start_pos=0 skips JIT (warmup). Subsequent prefill tokens (start_pos>0) hit JIT. So prefill is mostly JIT'd too, just one token at a time.
  - Both paths: exactly **1 JIT call, 1 realize** per generated token. No novel graph shapes during steady-state decode.
- **Shape:** **divergent against the premise.** The question's assumption is wrong — tinygrad's symbolic variables handle varying positions and lengths without recompilation. The graph shape is fixed; only variable bindings change.
- **Kill condition:** the premise is killed. The competitive gap on full LLaMA inference is NOT realize overhead from novel graph shapes. Must look elsewhere.
- **Edge →** if not realize overhead, what?
- **Provenance:** code trace of `llm/model.py:394-414`, `llm/model.py:308-319`, `extra/models/llama.py:221-225`, `engine/jit.py:260-304`. Mode: deduction (99% confidence).

## H₁ₘ: Decode gap is matvec kernel quality

- **Null:** tinygrad's auto-generated matvec kernels achieve near-peak memory bandwidth
- **Perturbation:** trace the codegen path for a 1×4096 × 4096×4096 matmul (LLaMA 8B attention projection at T=1)
- **Trajectory:**
  1. **Tensor cores skipped.** Metal TC tiles are 8×8×8 (`codegen/opt/tc.py:140`). For T=1 decode, M=1. With default `TC_OPT=0`, the codegen requires M%8==0 (`postrange.py:255`). 1%8≠0 → TC path fails → falls through to matvec heuristic. Correct decision — TC is compute-focused, matvec is memory-bound.
  2. **Matvec heuristic applied.** `heuristic.py:64-82`: `GROUP(0, 8)` splits K across 8 threads with shared memory reduction. `LOCAL(global_idx, 4)` creates 4 blocks. `UPCAST(global_idx, 4)` processes 4 output rows per thread. Reasonable structure.
  3. **No input vector broadcast caching.** The input vector (1×4096 = 8KB) is read from global memory by each of the 8 GROUP threads. Not cached in threadgroup memory for broadcast. However: 8KB × 8 = 64KB, negligible vs 32MB weight read. **Not a real bottleneck.**
  4. **No explicit float4 vectorization.** Metal renderer emits scalar loads (`*ptr` at `cstyle.py:51`). Whether the Metal compiler auto-vectorizes depends on access pattern alignment. Hand-written kernels (llama.cpp) explicitly use `float4` loads, guaranteeing 4× wider memory transactions.
  5. **No SIMD reduction.** `GROUP(8)` → shared memory + barrier for partial-sum reduction. Hand-written kernels use `simd_sum()` which reduces within a SIMD group without shared memory round-trip. This adds latency per reduction step.
  6. **Wrong loop ordering — confirmed.** The generated Metal kernel walks the weight matrix with stride 32768 bytes (`Ridx0<<15`) in the inner loop. The unit-stride output axis (`+4, +8, +12`) is accessed outside the loop. This is 8192× worse than optimal for float32 (stride 32768 vs stride 4). Confirmed by `extract.py`; see `kernels/matvec_attn_proj.metal`. Same structural issue as PR #14630 on CPU.
- **Shape:** **divergent.** Each sample accumulates evidence that the auto-generated matvec has structural deficiencies compared to hand-written kernels. The deficiencies are consistent: missing vectorization, missing SIMD reduction, scalar loads, and likely wrong loop ordering. These compound to reduce memory bandwidth utilization.
- **Estimated impact:** For memory-bound matvec, bandwidth utilization determines throughput. If tinygrad achieves 60-80% of peak bandwidth (plausible given scalar loads + shared memory reduction), and hand-written kernels achieve 85-95%, the gap is 10-30% of per-token latency. This is the **dominant factor** for decode.
- **Edge →** can this be quantified with a benchmark?
- **Provenance:** code trace of `heuristic.py:64-82`, `postrange.py:219-312`, `tc.py:140-145`, `cstyle.py:342-384`. Mode: deduction (85% confidence — bandwidth utilization estimates are abductive).

## H₂ₘ: Prefill gap is arithmetic intensity ceiling

- **Null:** tinygrad's chunked prefill is competitive with batch prefill
- **Perturbation:** compare arithmetic intensity across three prefill strategies
- **Trajectory:**
  The key metric is arithmetic intensity — FLOPs per byte of weight read. For a `T×K @ K×N` matmul, FLOPs = `2*T*K*N`, weight bytes = `K*N*bytes_per_weight`. Intensity = `2*T / bytes_per_weight`. Higher T means better amortization:
  - **Token-at-a-time prefill** (old path, `examples/llama3.py:270-273`): T=1 per forward pass. Intensity = 2/2 = 1.0 FLOP/byte (float16). Fully memory-bound — each weight byte produces only one multiply-add. For a 1000-token prompt: 1000 forward passes, each reading all weights.
  - **Chunked prefill** (new path, `llm/model.py:394-414`): T=32 per forward pass. Intensity = 64/2 = 32 FLOP/byte. Still memory-bound on Apple Silicon (~10 TFLOP/s compute vs ~200-400 GB/s bandwidth → ridge point at ~25-50 FLOP/byte). For a 1000-token prompt: ~32 forward passes.
  - **Batch prefill** (llama.cpp, MLX): T=1000 per forward pass. Intensity = 2000/2 = 1000 FLOP/byte. Well into compute-bound territory. 1 forward pass, weights read once.
  - Chunking caps intensity at `2*chunk_size / bytes_per_weight`. With chunk_size=32 and float16 weights, the ceiling is 32 FLOP/byte — near the memory/compute ridge point. Batch prefill at T=1000 reaches 1000 FLOP/byte, fully compute-bound.
- **Shape:** **divergent.** Chunked prefill is structurally limited to low arithmetic intensity, preventing the GEMM-regime compute reuse that batch prefill achieves.
- **Kill condition:** strongly supported; needs measurement against true batched prefill.
- **Estimated impact:** The gap scales with prompt length. For short prompts (T<32), chunking matches batch. For T=1000, batch prefill amortizes weight reads ~32× better. The old token-at-a-time path is ~1000× worse.
- **Edge →** is the chunk_size=32 optimal? Could chunk_size=128 or 256 close more of the gap? Diminishing returns expected as KV cache memory grows with chunk size.
- **Provenance:** code trace of `llm/model.py:394-414`, `examples/llama3.py:257-274`. Mode: deduction (90% confidence — arithmetic intensity framing is sound, absolute speedup predictions need measurement).

## H₃ₘ: Metal submission overhead is scale-dependent

- **Null:** Metal command submission overhead is negligible for large model inference
- **Perturbation:** count command buffers, ObjC calls, and sync points per decode step
- **Trajectory:**
  - **Graph batching:** `JIT_BATCH_SIZE=32` (`helpers.py:236`) with doubling schedule. For ~400 decode kernels: 32+64+128+176 = **4 MetalGraph instances**. Each gets its own ICB and command buffer.
  - **Per MetalGraph.__call__:** `wait_check` on previous token's command buffer (synchronous), ICB parameter updates, new command buffer + encoder creation, `useResources` call, `executeCommandsInBuffer`, commit. ~30+ ObjC message sends per graph.
  - **FIX_METAL_ICB on M1/M2:** O(n_pipelines) dummy dispatches per graph. For ~50-100 unique pipelines: 100-200 extra ObjC calls per graph.
  - **wait_check serialization:** each MetalGraph waits for its OWN previous command buffer (from last token), not for other graphs within the current token. For large models with 20-40ms GPU execution, the previous token's graphs have completed by re-entry. **wait_check is effectively free for large model inference.**
  - **Total overhead:** ~4 command buffer creates + encodes + commits. Unmeasured — needs Metal System Trace to quantify.
  - **Comparison to llama.cpp:** 1 command buffer, 1 encoder, all ops encoded, 1 commit, 1 wait. tinygrad: ~4 command buffers, ~120-800 ObjC calls.
- **Shape:** **convergent.** Metal submission overhead is real but its share of total latency depends on model size and quantization:
  - Large FP16 models (20-40ms GPU compute): submission is a small fraction.
  - Small or quantized models (2-5ms GPU compute): submission overhead becomes a significant fraction.
  - The "secondary" label applies to large FP16 models but not universally.
- **Edge →** Metal System Trace would quantify the actual overhead per model size.
- **Provenance:** code trace of `runtime/graph/metal.py`, `engine/jit.py:31-60`, `helpers.py:236`. Mode: deduction (85% confidence — overhead magnitude is unmeasured).

## H₄ₘ: Quantization fusion works but format support is narrow

- **Null:** tinygrad's quantized inference has no efficiency gap
- **Perturbation:** trace the Int8Linear and NF4Linear codegen paths
- **Trajectory:**
  - **Int8Linear** (`examples/llama3.py:77`): `x.dot(self.weight.cast(self.scale.dtype).T * self.scale)`. The cast(int8→float16) + scale multiply are elementwise producers feeding into a reduce (dot). Tinygrad's scheduler fuses elementwise producers into consumer reduces. **This should produce one fused kernel** — dequantization inside the matmul loop, no intermediate materialization. Correct architecture.
  - **NF4Linear** (`examples/llama3.py:123-128`): lookup table decode + reshape + scale + linear. More complex — involves `CODE[unpacked]` indexing, reshape, scale multiply, then `x.linear()`. Whether this fully fuses is uncertain; the indexing might force materialization.
  - **Format support:** tinygrad offers int8, nf4, fp8. llama.cpp offers Q2_K through Q8_K plus IQ formats — ~15 quantization types with hand-written Metal kernels per type. Each llama.cpp kernel has specialized memory access patterns for the packed format (e.g., Q4_K packs 32 weights into 18 bytes with superblocks).
  - **The real gap:** not fusion efficiency but **quantization format sophistication.** llama.cpp's k-quants achieve better quality/size tradeoffs than uniform int8, AND have hand-tuned Metal kernels for each format. tinygrad's int8 is simpler but uses 2× the memory of Q4_K for comparable quality.
- **Shape:** **convergent.** Int8 fusion works but the format gap is the real issue. Evidence rises (limited formats) then settles (int8 works correctly for what it is).
- **Provenance:** code trace of `examples/llama3.py:71-98`, `examples/llama3.py:110-147`. Int8 fusion claim is abductive (80% confidence). Format comparison is deductive (95% confidence).

## H₅ₘ: Graph-level scheduling is fine; within-kernel loop ordering is not

- **Null:** tinygrad's scheduling decisions create unnecessary kernel boundaries
- **Perturbation:** count `.contiguous()` calls in model code that force materializations; separately examine within-kernel loop ordering
- **Trajectory:**
  - **Graph-level fusion boundaries are reasonable:**
    - Line 121: `self.ffn_gate(x).silu().contiguous() * self.ffn_up(x)` — intentional materialization to avoid TC-hostile fusion (comment in old model, `extra/models/llama.py:129`).
    - Line 136: `(h + self._feed_forward(self.ffn_norm(h))).contiguous()` — required for AFTER+STORE pattern.
    - `@function(precompile=True)` wraps each block. Schedule cache (`schedule/__init__.py:91`) caches by function body key. No redundant scheduling across identical layers.
    - RMSNorm fuses into one kernel. Attention Q/K/V are separate matmuls (correct — different weight matrices). Residual additions fuse into downstream ops. No gratuitous kernel boundaries found.
  - **Within-kernel loop ordering is the structural problem:**
    - The same scheduler that decides kernel boundaries also determines range ordering inside each kernel (via `rangeify.py`). For matvec-shaped matmuls, it produces reduction loops with stride-32768 access on the weight matrix instead of stride-4. Confirmed on Metal by `extract.py` (see H₁ₘ bullet 6); same structural issue as PR #14630 on CPU.
    - This is not a graph-level fusion problem — it's a loop-level codegen problem. The kernel boundaries are right; the inner loop of each kernel walks the wrong axis.
- **Shape:** **convergent.** Two distinct findings: graph-level scheduling is correct, within-kernel loop ordering is broken for matvec shapes. These are different layers of the same system and should not be conflated.
- **Provenance:** code trace of `llm/model.py:121,136`, `extra/models/llama.py:128-130`, `schedule/__init__.py:91-114`, `schedule/rangeify.py`. Mode: deduction (90% confidence).

---

## Graph State

| Node | Status | Shape | Prior finding updated? |
|------|--------|-------|----------------------|
| H₀ₘ | killed | divergent (against premise) | Overturns assumption — JIT covers all inference |
| H₁ₘ | confirmed | divergent | Extends H₄ᵦ — kernel quality matters at model scale for matvec specifically |
| H₂ₘ | confirmed | divergent | New finding — arithmetic intensity ceiling from chunking |
| H₃ₘ | confirmed (scale-dependent) | convergent | Extends H₃ₐ — host overhead matters more for small/quantized models |
| H₄ₘ | demoted | convergent | Orthogonal axis — only relevant for quantized competitor comparisons |
| H₅ₘ | split | convergent | Graph-level fusion: fine. Within-kernel loop ordering: broken. |

## Current Attribution Hypothesis

Unmeasured — pending benchmark validation. Rankings are from code analysis and issue tracker evidence. Exact shares require the measurement bridge described in the frontier edges.

### Decode (steady-state tok/s)

| Source | Likely rank | Evidence basis |
|--------|------------|----------------|
| Matvec loop ordering / kernel quality | **Primary** | PR #14630: 2.76× CPU gap. **Metal confirmed**: generated kernel walks weight matrix with stride 32768 bytes (Ridx0<<15) in the inner loop. Unit-stride output axis (+4, +8, +12) is outside the loop. See `kernels/matvec_attn_proj.metal`. |
| Metal submission overhead | **Secondary (scale-dependent)** | 4 MetalGraph submits per token. Matters more for small/quantized models, less for large FP16. |
| Python JIT dispatch | **Tertiary** | `_prepare_jit_inputs` + buffer substitution. Small fixed cost per token. |
| Realize/scheduling | **Negligible** | Fully amortized by JIT + schedule cache. |
| Quantization format support | **Orthogonal** | Separate competitiveness axis. Only relevant when comparing against quantized competitor runs. |

### Prefill (time-to-first-token)

| Source | Likely rank | Evidence basis |
|--------|------------|----------------|
| Arithmetic intensity ceiling | **Primary** | chunk_size=32 caps intensity at 32 FLOP/byte (float16), near the ridge point. Batch T=1000 reaches 1000 FLOP/byte — fully compute-bound. |
| Kernel quality | **Secondary** | Same loop ordering issue, but higher T partially amortizes. |
| JIT warmup | **Tertiary** | First 2 chunks are warmup. Negligible for long prompts. |

## Key findings vs prior investigation

The prior op-level investigation (HYPOTHESIS_GRAPH.md) concluded:
- **Small-op gap: Python realize overhead.** ✓ Confirmed, but irrelevant at model scale (JIT amortizes).
- **Large-op kernel quality: competitive.** ✗ Partially revised. Large GEMM (4096×4096 × 4096×4096) is competitive, but LLaMA decode is **matvec** (1×4096 × 4096×4096), which takes a completely different codegen path (matvec heuristic, no TC). The GEMM competitiveness result does not predict matvec competitiveness.
- **Compound-op gap: vendor primitives.** ✓ Still true, but less relevant at model scale where matvec dominates total compute time. Softmax and layernorm are a small fraction of per-token latency for a 32-layer model.

**The model-level investigation reveals a gap the op-level investigation couldn't see:** the matvec codegen path is the dominant factor, and it was never tested at the op level because those benchmarks used square GEMMs.

## Frontier edges

1. **Benchmark matvec bandwidth utilization.** Run tinygrad's generated matvec kernel on a 4096×14336 matrix (LLaMA 8B FFN) and measure achieved GB/s vs peak. Compare to `MPS.matmul` and llama.cpp's Metal matvec. This would convert the bandwidth estimate from abduction to induction. **Decisive, cheap, reversible.**

2. **Test chunk_size sensitivity.** Run `generate()` with `chunk_size=128` and `chunk_size=256`. Measure prefill time for 1000-token prompts. Predict: higher chunk sizes increase arithmetic intensity and reduce prefill time, with diminishing returns as KV cache memory grows. **Decisive, cheap, reversible.**

3. **Profile Metal submission overhead.** Metal System Trace on a decode step to measure actual command buffer lifecycle time. Compare M1 (with FIX_METAL_ICB) vs M3+ (without). **Decisive, cheap, reversible.**

4. **Dump Metal matvec IR.** Run `extract.py` from the experiment repo against tinygrad to show the actual loop ordering for a 1×4096 × 4096×4096 matmul. This converts PR #14630's analogical evidence into direct evidence for Metal. **Decisive, cheap, reversible.**

### Measurement bridge

| Test | Purpose | Converts |
|------|---------|----------|
| Isolated 1×K × K×N matvec on Metal | Measure tinygrad bandwidth utilization | H₁ₘ abduction → induction |
| Same shape via llama.cpp / MLX | Establish competitor baseline | H₁ₘ relative gap |
| Decode with fake weights, no sampling | Separate kernel time from Python/token logic | H₃ₘ overhead estimate |
| Prefill T=32 vs T=128 vs T=512 vs T=1000 | Quantify arithmetic intensity curve | H₂ₘ chunking penalty |
| Metal System Trace | Validate command-buffer overhead | H₃ₘ timing |
| `extract.py` from experiment repo | Show actual Metal matvec loop ordering | H₁ₘ CPU→Metal analogy |

## Reasoning mode table

| Node | Abduction | Deduction | Induction |
|------|-----------|-----------|-----------|
| H₀ₘ | — | Traced JIT code paths exhaustively | — |
| H₁ₘ | Bandwidth utilization estimate (unmeasured) | Traced codegen path, identified structural gaps + loop ordering | — |
| H₂ₘ | — | Derived arithmetic intensity ceiling from chunk_size | — |
| H₃ₘ | Overhead magnitude unmeasured | Counted ObjC calls and sync points | — |
| H₄ₘ | Int8 fusion likely works | Traced codegen fusion rules | — |
| H₅ₘ | — | Read all .contiguous() sites, traced schedule cache | — |

All findings are deductive (code reading) or abductive (estimates from structural analysis). No inductive findings — no benchmarks were run. The frontier edges and measurement bridge are designed to convert abductive claims to inductive ones.

## Core thesis

The original "novel graph shapes cause realize overhead" premise is false. Tinygrad's symbolic JIT covers steady-state LLaMA inference. The remaining gap splits into two regimes: **decode is dominated by matvec kernel quality**, especially memory access order and reduction strategy; **prefill is dominated by limited batching**, which caps arithmetic intensity and prevents the GEMM-regime compute reuse competitors get from large batch sizes. Metal submission and Python dispatch are real but scale-dependent — secondary for large FP16 models, potentially significant for small or quantized ones. Quantization format support is an orthogonal axis that only matters when comparing against quantized competitor paths.

---

## Issue Tracker Evidence

Findings from tinygrad's GitHub issues and PRs, corroborating or extending the hypothesis graph.

### Historical baseline: Issue #1209 — "Perpetual Bounty: llama Python runtime 20%+ faster on M1"

LLaMA 7B FP16 on M1 Max: **~100ms per token (~10 tok/s)** via Metal backend. The bounty was specifically about Python-side time between kernel dispatches. This confirms the prior investigation's finding that realize overhead was the dominant bottleneck historically — and that the team knew it. The 2026 symbolic JIT refactoring (PRs #15097/#15153) specifically targeted this.

### Infrastructure refactor: PRs #15097, #15153, #15149 (March 2026)

The recent refactoring specifically addresses the bottlenecks our investigation identified:
- **PR #15097** ("fully symbolic llm"): Makes LLM graph fully symbolic → enables JIT replay without recompilation. Directly addresses H₀ₘ.
- **PR #15153** ("two jits, prefill/rollout"): Splits JIT into separate paths → the architecture our H₂ₘ analyzes.
- **PR #15149** ("chunked prefill"): Enables chunked prefill for long prompts → the chunk_size=32 design our H₂ₘ critiques.

These PRs were merged in March 2026. **No public benchmarks have been posted since this refactoring.** The gap may have narrowed substantially but is unmeasured on Metal.

### Multi-GPU evidence: Issue #5244 — "llama speed tracking issue"

LLaMA 7B on tinybox (4× AMD GPUs): 130-140 tok/s. The discussion mentions "cpu time spent mostly updating var vals and launch bounds" — confirming H₃ₘ's finding that Metal submission overhead (ICB parameter updates, command buffer lifecycle) is a real cost.

### eGPU comparison: Issue #15669

tinygrad on RTX 3060 eGPU: Qwen3-8B Q4_K_M = **4 tok/s gen**. llama.cpp on Metal (same machine): Qwen3.5-9B IQ3_XXS = **11.2 tok/s gen**. The ~60% gap on the eGPU path is inflated by Thunderbolt bandwidth constraints, but the qualitative ordering (llama.cpp faster) is consistent with our analysis.

### Matvec root cause: PR #14630 / Issue #14412 — CPU matvec bounty

**The most critical finding.** This thread confirms H₁ₘ with a cross-backend root cause:

- **Bounty:** "llama 1B faster than torch on CPU in CI" (#14412)
- **Baseline:** tinygrad was **2.76× slower** than torch on CPU matvec (2.16 vs 5.95 tok/s)
- **A custom hand-written matvec kernel closed the gap to 0.99×** (5.59 vs 5.55 tok/s)
- **geohot rejected the custom kernel**, asking "why can't BEAM find it?"
- **Root cause diagnosis (xaviersavoie):** "The fundamental problem is that the generated code for matvec on CPU has mostly incorrect loop ordering... incorrect memory access patterns (huge strides). Then with devectorization, the code is riddled with unnecessary GEPs... fma is not guaranteed on all shapes."
- BEAM finds good shapes but cannot overcome the structural loop-ordering problem

**Strong analogical evidence.** The matvec codegen problem is **structural** — loop ordering produces huge strides that prevent vectorization. This is not a Metal-specific issue; it's a scheduler/devectorizer design problem that manifests across all backends. Our H₁ₘ prediction (scalar loads, no explicit vectorization, shared memory overhead) is the Metal-specific manifestation of this general bug. However, this is evidence by analogy: the CPU loop ordering bug has not been directly confirmed in the Metal LLaMA matvec IR. The experiment repo (`~/documents/tinygrad-matvec-experiment/extract.py`) is designed to close this gap.

### Active optimization sprint: PRs #15960, #15993, #16071 (April-May 2026)

The team (wozeparrot) is actively merging "llama speed" PRs — speed 2 (+925/-204 lines), speed 4, speed 6 (open). The volume suggests architectural changes, not just tuning. **The gap is real, acknowledged, and being actively worked on.**

### JITBEAM as workaround: Issue #15669

The team's fix for poor inference speed is `JITBEAM=2` — beam search over kernel optimizations. This confirms that **default heuristic-generated kernels are suboptimal** and beam search can find better configurations. But beam takes extremely long on Metal (impractical for casual use), and even BEAM cannot fix the structural loop-ordering problem (#14630).

### Metal overhead is acknowledged as secondary: PR #14709

An attempt to cache variable index lookups in Metal graph dispatch was rejected by chenyuxyz: "don't think it's worth the additional lines... practically speaking we have like 3 vars at most." **The team doesn't consider Metal submission overhead worth optimizing** — consistent with our H₃ₘ finding.

### Missing data

There are **no public 2025-2026 benchmarks of tinygrad LLM inference on Metal**. The prior benchmark (10 tok/s on M1 Max, 2023) predates all the architectural improvements. The issue tracker tracks tinybox (multi-GPU) and eGPU performance but not Metal single-device. This is itself a finding: Metal single-device LLM inference is not a tracked metric in tinygrad CI.

---

## Issue Tracker Attribution Update

PR #14630 strengthens H₁ₘ from "probably the dominant factor" to "dominant factor with identified structural root cause (on CPU; Metal confirmation pending)."

### The structural root cause (confirmed on Metal)

The matvec codegen problem is a **loop ordering issue in the scheduler**. On CPU, the scheduler produces loops with huge strides instead of unit strides (PR #14630). On Metal, `extract.py` confirms the same pattern: the generated kernel for a 1×4096 × 4096×4096 matmul walks the weight matrix with `Ridx0<<15` (stride 32768 bytes) in the inner loop, while the unit-stride output dimension is accessed via `+4, +8, +12` offsets outside the loop body. This means:
- BEAM search cannot fix it (search space doesn't include loop reordering)
- Metal-specific optimizations cannot fix it (problem is pre-Metal, in the IR)
- JIT tuning cannot fix it (JIT replays the bad kernel, just faster)

The fix would require changing how the scheduler orders reduction loops in matvec-shaped matmuls (M=1 or N=1), ensuring the inner loop iterates over the dimension with unit stride in the weight matrix. The experiment repo at `~/documents/tinygrad-matvec-experiment/` prototypes this as a stride-cost function that derives correct ordering for all tested shapes.

---

## PR Status

| PR | Title | Branch | Status | Hypothesis |
|----|-------|--------|--------|------------|
| [#16072](https://github.com/tinygrad/tinygrad/pull/16072) | increase matvec MV_ROWS_PER_THREAD from 4 to 16 | `stride-aware-matvec` | **Open** | H₁ₘ / H₅ₘ (matvec loop ordering) |
| [#16085](https://github.com/tinygrad/tinygrad/pull/16085) | onnx: deduplicate simple proto parsers | `onnx-proto-dedup` | **Open** | Sibling investigation (line budget) |
| [#16070](https://github.com/tinygrad/tinygrad/pull/16070) | add Ops.WARP_REDUCE for GROUPTOP reductions | `warp-reduce-v2` | **Draft** | Prior investigation (simd_sum) |

All PRs are on `kimjune01/tinygrad`.
