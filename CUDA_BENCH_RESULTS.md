# CUDA Benchmark: post-TC heuristic fix (PR #16104)

**Date:** 2026-05-08
**GPU:** NVIDIA GeForce RTX 4080 (16GB)
**Driver:** 596.36, CUDA 13.2
**OS:** Windows 11 Enterprise
**Python:** 3.14.4
**tinygrad:** 0.12.0 (editable install)

## Results

### Baseline (`skip-root-op-check`)

```
16x4096 @ 4096x4096:  r_64_4_16_4_4_1024_4   tm 1085.44us  1607 GFLOPS
256x256 @ 256x256:    r_8_4_8_16_4_4_64_4     tm   72.70us  1500 GFLOPS
8x2048 @ 2048x2048:   r_32_2_16_4_4_512_4     tm  329.73us   661 GFLOPS
```

### Fix (`post-tc-upcast-fix`)

```
16x4096 @ 4096x4096:  r_64_4_16_4_4_1024_4   tm 1084.19us  1609 GFLOPS
256x256 @ 256x256:    r_8_4_8_16_4_4_64_4     tm   77.82us  1401 GFLOPS
8x2048 @ 2048x2048:   r_32_2_16_4_4_512_4     tm  329.73us   661 GFLOPS
```

### Comparison

| Matmul Shape | Baseline (GFLOPS) | Fix (GFLOPS) | Kernel Shape | Delta |
|---|---|---|---|---|
| 16x4096 @ 4096x4096 | 1607 | 1609 | `r_64_4_16_4_4_1024_4` | ~neutral |
| 256x256 @ 256x256 | 1500 | 1401 | `r_8_4_8_16_4_4_64_4` | -6.6% (noise) |
| 8x2048 @ 2048x2048 | 661 | 661 | `r_32_2_16_4_4_512_4` | neutral |

## Linearizer Tests (`post-tc-upcast-fix`)

```
18 passed, 7 skipped, 2 deselected, 1 xfailed in 2.13s
```

2 tests deselected (`test_arg_acc_dtype`, `test_sum_acc_dtype`) — both fail with `KeyError: dtypes.bfloat16` in the PTX renderer. Confirmed pre-existing: same failure on `skip-root-op-check` baseline. Not caused by the PR.

## Conclusion

Kernel shapes are identical between baseline and fix on all three matmul sizes. The heuristic change does not alter CUDA's kernel selection. The 256x256 dip is measurement noise (same kernel, same shape). All non-bfloat16 linearizer tests pass. PR is safe on CUDA.
