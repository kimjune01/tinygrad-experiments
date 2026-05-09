# Baseline: bench_perceive with both PRs folded in

Branch: `or-baseline` (master + #16072 matvec + #16094 contiguous-prune-llm)
Date: 2026-05-07
Platform: arm64, macOS 26.4.1, Apple Silicon (Metal)
PyTorch: MPS backend, float32

## Results

```
workload         torch     noopt      heur  #k   lift  v_torch
gemm_1024       1.16 ms  20.40 ms    973 us   1  21.0x   0.84x
gemm_256         480 us   2.21 ms    819 us   1   2.7x   1.71x
add_4096        1.22 ms  11.84 ms    998 us   1  11.9x   0.82x
mul_sum          982 us   1.25 ms   1.46 ms   3   0.9x   1.49x
relu_4096       1.07 ms  11.91 ms   1.33 ms   1   8.9x   1.25x
exp_2048         586 us   3.78 ms    941 us   1   4.0x   1.61x
sum_4096         585 us   1.00 ms   1.31 ms   3   0.8x   2.24x
permute          507 us   1.58 ms    783 us   1   2.0x   1.54x
softmax          471 us   1.55 ms   1.14 ms   3   1.4x   2.43x
layernorm        460 us   4.42 ms   1.83 ms   3   2.4x   3.98x
matvec           774 us   1.43 ms   1.89 ms   1   0.8x   2.45x
```

Summary: median v_torch = 1.61x, range 0.82x - 3.98x

## Comparison with PERCEIVE_ANALYSIS (pre-PR baseline)

```
workload       v_torch_old  v_torch_new  delta   notes
gemm_1024         1.81x       0.84x      +++     BEATS PyTorch (was 1.81x behind)
gemm_256          5.02x       1.71x      ++      major improvement
add_4096          1.76x       0.82x      +++     BEATS PyTorch
mul_sum           1.51x       1.49x      ~       no change (3 kernels, fusion gap)
relu_4096         1.20x       1.25x      ~       slightly worse (noise?)
exp_2048          2.13x       1.61x      +       improved
sum_4096          1.97x       2.24x      -       slightly worse
permute           2.96x       1.54x      ++      major improvement
softmax           5.85x       2.43x      ++      improved but still 3-kernel gap
layernorm         5.32x       3.98x      +       improved but still 3-kernel gap
matvec            2.25x       2.45x      ~       PR #16072 effect not visible here (needs BEAM?)
```

## What improved since PERCEIVE_ANALYSIS

- gemm_1024 and add_4096 now BEAT PyTorch (0.84x and 0.82x)
- gemm_256, permute: 2-3x improvement
- softmax, layernorm: improved ~2x but still 2.4-4.0x behind (fusion gap)

## What the OR hypotheses target

| Hypothesis | Target workloads | Current gap | Expected after |
|---|---|---|---|
| H1 (CPL scheduling) | gemm_256, exp_2048, permute, relu_4096 | 1.25-1.71x | <1.2x (close to parity or beat) |
| H2 (XOR-swizzle) | gemm_1024, gemm_256 | 0.84-1.71x | further improvement on GEMM |
| H4 (fused dequant) | matvec (decode proxy) | 2.45x | <1.5x |
| H5 (algebraic fusion) | softmax, layernorm, mul_sum, sum_4096 | 1.49-3.98x | <1.5x (single-kernel) |

## Platform caveat

gemm_1024 and add_4096 beat PyTorch MPS, which is PyTorch's least mature GPU backend. On CUDA with cuBLAS the bar is higher. tinygrad hits ~2700 GFLOPS on M5 Max 40-core; PyTorch MPS hits ~1970 GFLOPS on the same chip — tinygrad's compiled kernel outperforms MPS's library dispatch for this shape. Verify on CUDA Windows machine before claiming generality.

The fusion gap (softmax, layernorm) is the backend-agnostic result: PyTorch uses fused primitives on all backends and wins decisively. H5 (algebraic fusion) is the generalizable finding.

## Key observations

1. **gemm_1024 and add_4096 beat PyTorch MPS.** Metal-specific — tinygrad's compiled kernel outperforms MPS library dispatch for these shapes. Does not generalize to CUDA without verification.

2. **The fusion gap is now the dominant bottleneck.** softmax (2.43x) and layernorm (3.98x) are the worst. These are H5 targets.

3. **matvec is still bad (2.45x).** PR #16072 increased MV_ROWS_PER_THREAD but this benchmark may not exercise the specific path. Or the fix needs BEAM to find the right schedule.

4. **mul_sum and sum_4096 have heuristic_lift < 1.0** — the heuristic HURTS. These are the cases where H1's CPL scheduling should help most, since the current priority ordering is actively counterproductive.
