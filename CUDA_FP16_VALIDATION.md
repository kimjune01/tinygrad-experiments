# CUDA real-renderer validation of post-TC heuristic fix (PR #16107)

**Date:** 2026-05-08
**Hardware:** Windows 11 Enterprise, RTX 4080 (sm_89), driver 596.36
**Toolchain:** CUDA Toolkit 13.2.1, NVRTC 13.0, Python 3.14, tinygrad master @ `4d1a9dca4` vs `origin/post-tc-upcast-fix-v2` @ `c1349b9ad`
**Renderer:** `CUDARenderer` (verified — see [Provenance](#provenance))

## TL;DR

PR #16107 was previously labeled "CUDA neutral (identical kernel shapes on RTX 4080)." That label was based on the [earlier bench](CUDA_BENCH_RESULTS.md), which **silently ran on PTXRenderer** because NVRTC was missing — see [tinygrad#16112](https://github.com/tinygrad/tinygrad/issues/16112). With NVRTC now present and `CUDARenderer` actually selected, the picture is:

- **fp32:** kernel shapes are identical between master and PR. Trivially so — the diff modifies the post-TC opt block, and TC requires fp16/bf16, so the changed code path is never executed for fp32 inputs. "Neutral" is correct but uninformative.
- **fp16 (the actual test):** the PR delivers a real win on the largest shape — **3.35× speedup on 16×4096 × 4096×4096**. Smaller shapes are neutral (one because TC didn't trigger, one because the kernel changed but timing was within noise).

Recommended copy for the PR description:

> **Metal (M4 Max, fp32):** -47% / -51% / -59% on three matmul shapes (TC fires).
> **CUDA (RTX 4080, fp16):** **3.35× speedup on 16×4096 × 4096×4096**; neutral on 256² and 8×2048 (TC didn't fire on the latter, kernel changed but performance was within noise on the former).
> **CUDA (RTX 4080, fp32):** untouched — kernel shapes identical to master, as expected (TC code path not reached).
> **gfx12:** validated via CI (PR ships with conservative fallback to original heuristic on gfx12).

## fp32 results (TC code path NOT exercised)

| Shape | Master kernel | μs | PR kernel | μs | Identical? |
|---|---|---|---|---|---|
| 16×4096 × 4096×4096 | `r_64_4_16_4_4_1024_4` | 632 | `r_64_4_16_4_4_1024_4` | 638 | YES |
| 256×256 × 256×256 | `r_8_4_8_16_4_4_64_4` | 43.9 | `r_8_4_8_16_4_4_64_4` | 24.6 | YES (timing noise) |
| 8×2048 × 2048×2048 | `r_32_2_16_4_4_512_4` | 148 | `r_32_2_16_4_4_512_4` | 198 | YES (timing noise) |

Identical kernel names across all three shapes — confirms the post-TC opt block was unreached. Timing differences within noise (we did one realize per shape, no warmup beyond the trivial one). This matches the original PR claim.

## fp16 results (TC code path IS exercised)

| Shape | Master kernel | Master μs | Master GFLOPS | PR kernel | PR μs | PR GFLOPS | Speedup |
|---|---|---|---|---|---|---|---|
| 16×4096 × 4096×4096 | `r_32_32_4_2_2_4_256_2` | 223 | 2405 | `r_256_32_2_2_2_64_2_4` | **66.6** | **8066** | **3.35×** |
| 256×256 × 256×256 | `r_4_2_32_4_2_2_4_4_16_2` | 42.9 | 781 | `r_8_16_32_2_2_2_2_4_2_4` | 45.1 | 745 | 0.95× (~neutral, noise) |
| 8×2048 × 2048×2048 | `r_32_2_16_4_4_512_4` | 122.6 | 547 | `r_32_2_16_4_4_512_4` | 117.5 | 571 | 1.04× (TC didn't fire — same kernel) |

**The 16×4096 × 4096×4096 case is the load-bearing one.** Master's post-TC opts produce 2.4 TFLOPS; the PR's UPCAST(M)+UPCAST(N)+UNROLL(K) variant produces 8.1 TFLOPS — same hardware, same shape, same dtype. The kernel-name change confirms a real code-path difference, not a noise effect.

The 8×2048 case is interesting: kernel name is identical between master and PR, *and* identical to the fp32 version on the same shape. Suggests TC isn't firing for this shape on PTX even with fp16 — likely the heuristic's TC gate rejects it (small M dim, certain alignment). Worth a separate edge if you care about closing this — but not in scope for PR #16107.

## Provenance

Renderer was confirmed `CUDARenderer` (not PTXRenderer fallback) before each run:
```
$env:DEV="CUDA"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"   # toolkit dir for headers
$env:NVRTC_PATH = "$CUDA_PATH\bin\x64\nvrtc64_130_0.dll"                       # workaround for #16112
$env:NVJITLINK_PATH = "$CUDA_PATH\bin\x64\nvJitLink_130_0.dll"                 # workaround for #16112
$env:PATH = "C:\Users\junekim\.cuda-shim;$CUDA_PATH\bin\x64;" + $env:PATH      # cuda.dll → nvcuda.dll shim
```
The shim directory contains a copy of `C:\Windows\System32\nvcuda.dll` renamed `cuda.dll` (`runtime/autogen/cuda.py` calls `c.DLL('cuda', 'cuda')` and tinygrad's findlib looks for the literal name `cuda.dll`). This is the same Windows-loader-name issue as #16112; both should be fixed there.

```python
print('renderer:', type(Device[Device.DEFAULT].renderer).__name__)  # CUDARenderer
```

Bench script: 6 lines, one realize per shape per dtype, `IGNORE_BEAM_CACHE=1`, `DEBUG=2` to capture kernel names and per-call wall-clock times. No warmup beyond the trivial implicit one. Sample size 1 per cell — fine for kernel-name comparison and order-of-magnitude timing claims; the 3.35× headline is well outside noise. Run the bench multiple times if you want tighter intervals on the smaller cells.

## Open follow-ups

1. **Why doesn't TC fire on 8×2048 × 2048×2048 fp16?** The kernel name matches the fp32 version exactly, and PR #16107's edits never get applied. The heuristic's TC gate is rejecting this shape. Worth a short investigation if the goal is to close all gaps.
2. **Update the PR description** with the fp16 number — current "CUDA neutral" framing hides a real 3.4× win on the largest shape.
3. **Other fp16 matmul shapes** — only three were measured. The rest of the H4 theory-transfer matrix could now be re-run on real CUDA fp16, not Metal fp32.
