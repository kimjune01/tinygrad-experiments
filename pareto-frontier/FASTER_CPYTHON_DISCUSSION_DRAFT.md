# Benchmark idea: tinygrad as a "compiler inner loop" workload

Hi all,

I've been investigating CPython performance on tinygrad's compiler, and I think it might be a useful addition to pyperformance as a benchmark — it exercises a code shape that the current suite doesn't cover.

tinygrad is a small (~10K line) deep learning framework that compiles neural network graphs to GPU kernels at runtime. The compiler is pure Python — no C extensions. The hot function is `unified_rewrite` ([source](https://github.com/tinygrad/tinygrad/blob/main/tinygrad/uop/ops.py#L1476)), a ~100-line `while` loop that does `dict.get`, `deque.pop`, `set.__contains__`, `tuple()` construction, and callback dispatch on an in-memory graph. It accounts for about 68% of compilation time.

This "tight loop over dict/deque/set" pattern shows up in a lot of Python tools — type checkers, linters, code generators — but nothing in pyperformance currently stresses it. The existing benchmarks are mostly string processing (2to3, html5lib), I/O (dulwich, json), or numerical (nbody).

One thing that makes tinygrad interesting as a benchmark target: a Cython transpile of just `unified_rewrite` (same algorithm, no type annotations) gives -7.3% end-to-end. The tier 2 JIT on 3.16 creates substantial traces for it (up to 393 uops, 15+ executors) but shows 0% improvement. So there's a measurable gap between "same code compiled to C" and "JIT-compiled" that doesn't surface on the current benchmarks.

Easy to reproduce, no GPU needed:
```bash
pip install tinygrad
python -c "
import time
from tinygrad import Tensor
for _ in range(5):
    Tensor.randn(1,3,32,32).conv2d(Tensor.randn(16,3,3,3)).relu().conv2d(Tensor.randn(32,16,3,3)).relu().reshape(1,-1).matmul(Tensor.randn(32*28*28,128)).relu().matmul(Tensor.randn(128,10)).realize()
N=50; t0=time.perf_counter()
for _ in range(N):
    Tensor.randn(1,3,32,32).conv2d(Tensor.randn(16,3,3,3)).relu().conv2d(Tensor.randn(32,16,3,3)).relu().reshape(1,-1).matmul(Tensor.randn(32*28*28,128)).relu().matmul(Tensor.randn(128,10)).realize()
print(f'{(time.perf_counter()-t0)/N*1000:.2f}ms/iter')
"
```

Full investigation (28 hypotheses, including the Cython/JIT comparison): [kimjune01/tinygrad-pareto-frontier](https://github.com/kimjune01/tinygrad-pareto-frontier)

Tested on macOS 15.5, Apple M4 Max, CPython 3.16.0a0 (d36e5b8), `--enable-experimental-jit`.
