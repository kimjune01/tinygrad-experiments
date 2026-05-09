"""Benchmark matvec bandwidth utilization with different MV parameters.

Measures achieved GB/s for matvec on Metal with:
  1. Default heuristic (MV_THREADS_PER_ROW=8, MV_ROWS_PER_THREAD=4)
  2. No GROUP (MV_THREADS_PER_ROW=1, MV_ROWS_PER_THREAD=4)
  3. No GROUP + wider upcast (MV_THREADS_PER_ROW=1, MV_ROWS_PER_THREAD=16)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~/documents/tinygrad"))

from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Context, Timing

SHAPES = [
    ("attn_proj",  1, 4096, 4096),
    ("ffn_up",     1, 4096, 14336),
    ("ffn_down",   1, 14336, 4096),
]

CONFIGS = [
    ("default",       {"MV_THREADS_PER_ROW": "8", "MV_ROWS_PER_THREAD": "4", "MV_BLOCKSIZE": "4"}),
    ("no_group",      {"MV_THREADS_PER_ROW": "1", "MV_ROWS_PER_THREAD": "4", "MV_BLOCKSIZE": "4"}),
    ("no_group_w16",  {"MV_THREADS_PER_ROW": "1", "MV_ROWS_PER_THREAD": "16", "MV_BLOCKSIZE": "4"}),
    ("no_group_w32",  {"MV_THREADS_PER_ROW": "1", "MV_ROWS_PER_THREAD": "32", "MV_BLOCKSIZE": "4"}),
]

WARMUP = 3
TRIALS = 10

def bench_matvec(M, K, N, config_env):
    for k, v in config_env.items():
        os.environ[k] = v

    A = Tensor.randn(M, K).realize()
    B = Tensor.randn(K, N).realize()

    # warmup (also triggers JIT capture if any)
    for _ in range(WARMUP):
        C = (A @ B).realize()
    Device[Device.DEFAULT].synchronize()

    # benchmark
    times = []
    for _ in range(TRIALS):
        Device[Device.DEFAULT].synchronize()
        GlobalCounters.reset()
        t0 = time.perf_counter()
        C = (A @ B).realize()
        Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # cleanup env
    for k in config_env:
        del os.environ[k]

    median = sorted(times)[len(times) // 2]
    total_bytes = (M * K + K * N + M * N) * 4  # float32
    bandwidth = total_bytes / median / 1e9

    return median, bandwidth


if __name__ == "__main__":
    print(f"Backend: {Device.DEFAULT}")
    print(f"Warmup: {WARMUP}, Trials: {TRIALS}")
    print()

    # header
    config_names = [c[0] for c in CONFIGS]
    print(f"{'Shape':<22}", end="")
    for name in config_names:
        print(f"  {name:>16}", end="")
    print()
    print(f"{'':22}", end="")
    for _ in config_names:
        print(f"  {'ms':>7} {'GB/s':>7}", end="")
    print()
    print("=" * (22 + len(config_names) * 18))

    for shape_name, M, K, N in SHAPES:
        print(f"({M:>4},{K:>5},{N:>5})    ", end="")
        for config_name, config_env in CONFIGS:
            median, bw = bench_matvec(M, K, N, config_env)
            print(f"  {median*1e3:>7.2f} {bw:>7.1f}", end="")
        print()

    print()
    print("Higher GB/s = better bandwidth utilization.")
    print("If no_group >> default, the GROUP on k breaks memory coalescing.")
