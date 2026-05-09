# Benchmark Cross-Reference

Maps each hypothesis to existing tinygrad benchmarks that validate or kill it.

## H1: Instruction Scheduling (RCPSP)

### Validates

| Benchmark | Path | What it measures | Why it matters for H1 |
|---|---|---|---|
| **bench_perceive.py** | `test/speed/bench_perceive.py` | 11 workloads at NOOPT/heuristic/BEAM, vs PyTorch MPS | The single-kernel ops (gemm_1024, add_4096, relu_4096, permute) isolate codegen quality from fusion. CPL scheduling should close the 1.2-2x gap on these. The `heuristic_lift` metric (NOOPT/heuristic) directly measures how much the optimizer improves raw lowering — CPL should increase this. |
| **speed_v_torch** | `test/speed/external_test_speed_v_torch.py` | Comprehensive op-level comparison vs PyTorch (287 lines, many ops) | Broader coverage than bench_perceive. Tests add, gemm, conv, permute at multiple sizes. |
| **speed_v_theoretical** | `test/external/speed_v_theoretical.py` | Kernel speed vs roofline model | Directly measures how close generated code is to theoretical peak. CPL scheduling should move ops closer to the roofline. |
| **test_linearizer.py** | `test/backend/test_linearizer.py` | Correctness of linearized code | Regression gate — any linearizer change must pass this. |
| **test_linearizer_dumb.py** | `test/backend/test_linearizer_dumb.py` | Edge cases in linearization | Additional regression gate. |
| **benchmark_schedule.py** | `test/external/external_benchmark_schedule.py` | Scheduling + linearization time for ResNet50 | CPL adds a backward DFS pass — must not blow up compile time. This benchmark measures linearization time explicitly. |

### Key experiments to run

1. `python3 test/speed/bench_perceive.py` — baseline (current linearizer)
2. Same with CPL prototype patched in — measure `heuristic_lift` and `vs_torch` deltas
3. `DEBUG=2` on gemm_1024 and matvec to inspect instruction ordering before/after
4. `BEAM=2 python3 test/speed/bench_perceive.py` — check if CPL changes the BEAM baseline

### Kill metric

If CPL scheduling does not improve `heuristic_lift` on gemm_1024 and permute (the single-kernel, latency-sensitive ops), the hypothesis is weakened. If it regresses matvec (already heuristic_lift=0.8x), the kill condition fired (greedy constraining future steps via register pressure).

---

## H2: Bank Conflict Avoidance (GF(2) Assignment)

### Validates

| Benchmark | Path | What it measures | Why it matters for H2 |
|---|---|---|---|
| **bench_perceive.py** | `test/speed/bench_perceive.py` | gemm_1024, gemm_256 | GEMM uses shared memory tiles. Bank conflicts in shared memory directly affect GEMM throughput. XOR-swizzle should close part of the 1.8x gap on gemm_1024. |
| **test_opt_gemm.py** | `test/backend/test_opt_gemm.py` | GEMM correctness with various optimization options (UPCAST, UNROLL) | Regression gate — swizzled indices must produce correct results under all tiling configurations. |
| **test_asm_gemm.py** | `test/backend/test_asm_gemm.py` | Assembly-level GEMM tests | Low-level correctness for PTX/assembly GEMM kernels. |
| **bert_matmuls** | `test/external/external_benchmark_bert_matmuls.py` | BERT attention matmuls (q@k, qk@v, linear projections) | Non-square matmuls that use shared memory differently. Tests whether swizzle generalizes beyond square GEMM. |
| **specific_conv** | `test/speed/external_test_specific_conv.py` | Openpilot convolutions (1x1, 3x3) | Convolutions also use shared memory — swizzle should help here too. |
| **speed_v_theoretical** | `test/external/speed_v_theoretical.py` | Roofline comparison | Bank conflicts reduce effective shared memory bandwidth below theoretical. Swizzle should move closer to roofline. |

### Key experiments to run

1. Profile gemm_1024 with Metal/CUDA profiler to count bank conflicts (before)
2. Apply XOR-swizzle rewrite, re-profile (after)
3. Run bench_perceive gemm_1024 and gemm_256 — measure throughput delta
4. Run test_opt_gemm with all tiling options — correctness regression gate

### Kill metric

If profiling shows gemm_1024 already has few bank conflicts (because tinygrad's tile sizes happen to avoid them), the hypothesis doesn't apply to this workload. Check gemm_256 and bert_matmuls — smaller tiles are more likely to conflict.

---

## H4: Fused Dequantization (Knapsack)

### Validates

| Benchmark | Path | What it measures | Why it matters for H4 |
|---|---|---|---|
| **speed_llama** | `test/external/external_test_speed_llama.py` | LLaMA codegen speed | Measures code generation for the model that would benefit most from fused dequant. |
| **llama_schedule** | `test/external/external_benchmark_llama_schedule.py` | Scheduling time for 8B/405B LLaMA | Fused dequant changes the schedule — must not blow up scheduling time. |
| **bert_matmuls** | `test/external/external_benchmark_bert_matmuls.py` | Matmul shapes found in transformer layers | Same matmul shapes appear in quantized inference. |
| **benchmark_onnx.py** | `examples/benchmark_onnx.py` | ONNX model inference (20 timed runs) | End-to-end benchmark for quantized ONNX models. |
| **test_quantize_onnx.py** | `test/backend/test_quantize_onnx.py` | Quantized ONNX correctness | Regression gate for quantization paths. |
| **bench_perceive.py** | `test/speed/bench_perceive.py` | matvec workload | Decode-phase LLM inference is dominated by matrix-vector multiplies. The matvec workload (currently 2.25x vs torch) is the proxy. |

### Key experiments to run

1. Profile LLaMA GGUF inference: count global memory transactions for weight loading (before)
2. Implement fused dequant UOp rewrite, re-profile (after)
3. bench_perceive matvec — this is the decode-phase proxy
4. End-to-end tokens/sec on a quantized LLaMA model

### Kill metric

If fused dequant increases register pressure past an occupancy boundary (measured via `DEBUG=2` kernel stats or profiler), the over-fusion kill condition fired. Must check APRP before/after. If matvec doesn't improve, the bottleneck is elsewhere (memory coalescing, launch overhead).

### Dependency

Requires H1 (APRP-aware scheduling) to be safe. Without register pressure awareness, fused dequant may regress like PCONTIG=99 did for softmax.

---

## H5: Reduction Fusion (Algebraic Decomposition)

### Validates

| Benchmark | Path | What it measures | Why it matters for H5 |
|---|---|---|---|
| **bench_perceive.py** | `test/speed/bench_perceive.py` | softmax (3 kernels, 5.85x vs torch), layernorm (3 kernels, 5.32x vs torch) | **The primary targets.** These are the workloads where tinygrad is furthest behind. The 3-kernel → 1-kernel transition with O(1) state is exactly what algebraic decomposition enables. |
| **test_softmax_fusion.py** | `test/backend/test_softmax_fusion.py` | Single-kernel softmax correctness + fusion verification | **Critical regression gate.** Already has `single_kernel_softmax()` implementation and `run_one_schedule_item()` that verifies single-kernel emission. Any algebraic fusion must pass this. |
| **bert_softmax** | `test/external/external_benchmark_bert_softmax.py` | BERT-scale softmax (BS×16×512×512) | Real-world softmax shape from transformer attention. Tests whether fusion scales to production sizes. |
| **speed_v_torch** | `test/speed/external_test_speed_v_torch.py` | Multiple softmax/reduction shapes | Broader shape coverage than bench_perceive. |
| **mul_sum, sum_4096** | `test/speed/bench_perceive.py` | Reduction workloads (3 kernels each) | Additional reduction patterns. mul_sum has heuristic_lift=0.5x (heuristic HURTS). Algebraic decomposition should fix this. |
| **model_benchmark** | `test/external/external_model_benchmark.py` | ResNet50, efficientnet, shufflenet vs PyTorch/ONNX Runtime | End-to-end model benchmarks. Softmax/layernorm are on the critical path of transformer models. |

### Key experiments to run

1. `python3 test/speed/bench_perceive.py` — baseline softmax/layernorm (3 kernels, ~5x vs torch)
2. Apply Flashlight-style online softmax rewrite → measure kernel count (should be 1) and runtime
3. Apply RedFuser-style layernorm decomposition → same measurement
4. `python3 test/backend/test_softmax_fusion.py` — correctness gate (MUST pass)
5. `python3 test/external/external_benchmark_bert_softmax.py` — production-scale validation
6. Compare against PCONTIG=99 (naive fusion, known 1.8x SLOWER) — algebraic fusion should be faster, not slower

### Kill metric

If algebraic fusion produces a single kernel that is SLOWER than 3 kernels (like PCONTIG=99), the kill condition fired — the O(1) state claim didn't hold, or register pressure still dominated. Check register count via `DEBUG=4` generated code inspection.

### The decisive test (from PERCEIVE_ANALYSIS claim 4)

> If a scheduler or canonicalization change lowers softmax/layernorm from 3 compute dispatches to 1–2, and runtime improves materially, then the algebraic decomposition hypothesis is supported. If kernel count drops without runtime improvement, the bottleneck is not just launch/materialization overhead.

PCONTIG=99 already proved kernel count alone doesn't help. Algebraic decomposition must show that O(1)-state fusion is qualitatively different from naive fusion.

---

## Cross-cutting benchmarks

These benchmarks validate multiple hypotheses simultaneously:

| Benchmark | H1 | H2 | H4 | H5 |
|---|---|---|---|---|
| bench_perceive.py | gemm, matvec, permute | gemm | matvec | softmax, layernorm, mul_sum |
| speed_v_torch | all single-kernel ops | gemm, conv | — | softmax, layernorm |
| speed_v_theoretical | roofline gap | smem bandwidth | — | — |
| bert_matmuls | — | non-square GEMM | matmul shapes | — |
| bert_softmax | — | — | — | production softmax |
| test_softmax_fusion | — | — | — | correctness gate |
| benchmark_schedule | compile time | — | schedule time | — |

## Benchmark gaps

Missing benchmarks that would be needed:

1. **Shared memory bank conflict profiler** — no existing benchmark counts bank conflicts. Need Metal/CUDA profiler integration or a synthetic benchmark that isolates bank conflict overhead.
2. **Register pressure measurement** — no benchmark tracks register count per kernel. Need `DEBUG=4` code inspection or profiler data to validate H1's APRP claims.
3. **Quantized LLM end-to-end** — no existing benchmark measures tokens/sec for GGUF inference. The LLaMA benchmarks test codegen speed, not inference throughput.
4. **Linearizer ordering quality metric** — no benchmark directly measures instruction ordering quality. Proxy: runtime of single-kernel ops before/after CPL change.
