"""Benchmark: tinygrad-turbo vs original."""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad import Tensor
from tinygrad.schedule import schedule_cache


def bench(label, fn, n=9):
    times = []
    for _ in range(n):
        schedule_cache.clear()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return sorted(times)[n // 2]


def do_softmax(): Tensor.randn(32, 128).softmax().realize()
def do_conv(): Tensor.randn(1, 64, 56, 56).conv2d(Tensor.randn(64, 64, 3, 3), padding=1).relu().realize()
def do_4conv():
    x = Tensor.randn(1, 64, 56, 56)
    for _ in range(4):
        w = Tensor.randn(64, 64, 3, 3)
        x = x.conv2d(w, padding=1).relu()
    x.realize()
def do_transformer():
    B, T, C = 1, 32, 128
    x = Tensor.randn(B, T, C)
    q, k, v = x @ Tensor.randn(C, C), x @ Tensor.randn(C, C), x @ Tensor.randn(C, C)
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    (attn.softmax() @ v).realize()


workloads = [
    ("softmax", do_softmax),
    ("conv+relu", do_conv),
    ("4x conv", do_4conv),
    ("transformer", do_transformer),
]

# Warmup
for _, fn in workloads:
    fn()

# Baseline
print("=== Original ===")
baseline = {}
for name, fn in workloads:
    t = bench(name, fn)
    baseline[name] = t
    print(f"  {name:>15}: {t*1e3:.1f}ms")

# Install turbo
from turbo import install, uninstall
install()

# Warmup turbo
for _, fn in workloads:
    fn()

print("\n=== tinygrad-turbo ===")
turbo_times = {}
for name, fn in workloads:
    t = bench(name, fn)
    turbo_times[name] = t
    delta = (t / baseline[name] - 1) * 100
    print(f"  {name:>15}: {t*1e3:.1f}ms ({delta:+.1f}%)")

uninstall()

print("\n=== Summary ===")
for name in baseline:
    b, t = baseline[name], turbo_times[name]
    print(f"  {name:>15}: {b*1e3:.1f}ms → {t*1e3:.1f}ms ({(t/b-1)*100:+.1f}%)")
