"""tinygrad-turbo v2: C extension for the entire rewrite hot path.

No flattening, no boundary crossing for the 90% no-match case.
The C code accesses Python objects directly via CPython API.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UOp
import turbo_ext

_ORIGINAL_REWRITE = PatternMatcher.rewrite


def install():
    def _patched_rewrite(self, uop, ctx=None):
        return turbo_ext.turbo_rewrite(self, uop, ctx)
    PatternMatcher.rewrite = _patched_rewrite


def uninstall():
    PatternMatcher.rewrite = _ORIGINAL_REWRITE


if __name__ == "__main__":
    from tinygrad import Tensor
    import time
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

    workloads = [("softmax", do_softmax), ("conv+relu", do_conv),
                 ("4x conv", do_4conv), ("transformer", do_transformer)]

    # Warmup
    for _, fn in workloads: fn()

    print("=== Original ===")
    baseline = {}
    for name, fn in workloads:
        t = bench(name, fn)
        baseline[name] = t
        print(f"  {name:>15}: {t*1e3:.1f}ms")

    install()
    for _, fn in workloads: fn()  # warmup turbo

    print("\n=== tinygrad-turbo v2 (C extension) ===")
    for name, fn in workloads:
        t = bench(name, fn)
        delta = (t / baseline[name] - 1) * 100
        print(f"  {name:>15}: {t*1e3:.1f}ms ({delta:+.1f}%)")

    uninstall()
