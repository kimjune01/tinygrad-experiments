# Benchmark: post-TC heuristic fix on CUDA

## What you're testing

PR #16104 changes the post-TC heuristic in `tinygrad/codegen/opt/heuristic.py`. The old heuristic UPCASTs axis 0 (M) twice after applying tensor cores. The fix UPCASTs axis 1 (N) + UNROLLs axis 0 (K). On Metal this gives -43% to -51% on matmul kernels. Broke AMD gfx1201.

Need to verify it doesn't regress on CUDA.

## Setup

```bash
cd Documents\tinygrad
git remote add fork git@github.com:kimjune01/tinygrad.git
git fetch fork
```

## Benchmark

```bash
REM Baseline
git checkout fork/skip-root-op-check
set IGNORE_BEAM_CACHE=1
set DEBUG=2
python -c "from tinygrad import Tensor; Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(256, 256).matmul(Tensor.randn(256, 256)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048)).realize()"

REM Fix
git checkout fork/post-tc-upcast-fix
set IGNORE_BEAM_CACHE=1
set DEBUG=2
python -c "from tinygrad import Tensor; Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(256, 256).matmul(Tensor.randn(256, 256)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048)).realize()"
```

## What to look for

The `r_` line is the reduction kernel (the actual matmul). Compare the kernel shape and `tm` (time in microseconds) and `GFLOPS` between baseline and fix.

- If fix is faster or neutral on CUDA → PR is safe
- If fix is slower on CUDA → post in the PR, we need backend-conditional strategy

## Also run tests

```bash
git checkout fork/post-tc-upcast-fix
python -m pytest test/backend/test_linearizer.py -x -q
```

## Results

Record results in CUDA_BENCH_RESULTS.md in this repo.
