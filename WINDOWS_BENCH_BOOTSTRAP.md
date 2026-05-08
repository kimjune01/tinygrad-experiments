# Benchmark: post-TC heuristic fix on CUDA

## What you're testing

PR #16104 changes the post-TC heuristic in `tinygrad/codegen/opt/heuristic.py`. The old heuristic UPCASTs axis 0 (M) twice after applying tensor cores. The fix UPCASTs axis 1 (N) + UNROLLs axis 0 (K). On Metal this gives -43% to -51% on matmul kernels.

Need to verify it doesn't regress on CUDA.

## Setup

```bash
git clone git@github.com:kimjune01/tinygrad.git
cd tinygrad
git fetch origin
```

## Benchmark

```bash
# Baseline (main branch with skip-root-op-check)
git checkout skip-root-op-check
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096)).realize()" 2>&1 | findstr "r_"
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(256, 256).matmul(Tensor.randn(256, 256)).realize()" 2>&1 | findstr "r_"
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048)).realize()" 2>&1 | findstr "r_"

# Fix
git checkout post-tc-upcast-fix
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096)).realize()" 2>&1 | findstr "r_"
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(256, 256).matmul(Tensor.randn(256, 256)).realize()" 2>&1 | findstr "r_"
IGNORE_BEAM_CACHE=1 DEBUG=2 python -c "from tinygrad import Tensor; Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048)).realize()" 2>&1 | findstr "r_"
```

## What to look for

The `r_` line is the reduction kernel (the actual matmul). Compare the kernel shape and `tm` (time in microseconds) and `GFLOPS` between baseline and fix.

- If fix is faster or neutral on CUDA → PR is safe
- If fix is slower on CUDA → post in the PR, we need shape-aware post-TC logic

## Also run tests

```bash
git checkout post-tc-upcast-fix
python -m pytest test/backend/test_linearizer.py -x -q
```
