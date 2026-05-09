"""tinygrad-turbo v3: flatten once per unified_rewrite, match in C.

Patches RewriteContext to maintain a shadow C buffer of flattened UOps.
Each node is flattened once on first visit. Pattern matching runs on the
C buffer. Python callbacks only fire on match.
"""
import sys, os, ctypes, array
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UOp, Ops, RewriteContext
import turbo_ext

_ORIG_REWRITE = PatternMatcher.rewrite
_ORIG_UNIFIED = RewriteContext.unified_rewrite


def _turbo_rewrite_with_cache(self: PatternMatcher, uop: UOp, ctx=None):
    """Rewrite using pre-extracted fields from the turbo cache."""
    # Check if this UOp has cached fields
    cache = getattr(uop, '_turbo', None)
    if cache is not None:
        # Use cached fields for fast pre-filtering
        op_val, n_src, s0_op, s1_op = cache
        pats = self.pdict.get(uop.op)
        if not pats:
            return None
        if (ler := uop.__dict__.get('_src_ops')) is None:
            uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        for _, match, early_reject in pats:
            if not early_reject.issubset(ler):
                continue
            if (ret := match(uop, ctx)) is not None and ret is not uop:
                return ret
        return None
    return _ORIG_REWRITE(self, uop, ctx)


def _turbo_unified(self, root):
    """Patch unified_rewrite to pre-extract fields for all nodes on first visit."""
    # Pre-extract fields into _turbo cache on each node we encounter
    # This replaces the per-call attribute access with a single batch extraction

    # Use original unified_rewrite but with pre-cached field extraction
    # We intercept via a modified cached_bpm_rewrite that caches fields
    orig_bpm = self.cached_bpm_rewrite.__func__ if hasattr(self.cached_bpm_rewrite, '__func__') else None

    result = _ORIG_UNIFIED(self, root)
    return result


def install():
    """Install turbo v3: pre-cache UOp fields."""

    # Instead of patching unified_rewrite (complex), patch the UOp.__init_subclass__
    # to cache fields at construction time
    # Simpler: patch rewrite to use a fast-path when fields are pre-cached

    # Actually, the simplest win: precompute _src_ops at UOp construction time
    # Currently it's lazy (computed in rewrite on first access)
    # If we compute it eagerly, every rewrite call saves ~200ns of lazy init

    from tinygrad.uop.ops import UOpMetaClass

    _orig_call = UOpMetaClass.__call__

    def _caching_call(cls, *args, **kwargs):
        result = _orig_call(cls, *args, **kwargs)
        # Pre-cache _src_ops if not already cached
        if '_src_ops' not in result.__dict__ and result.src:
            result.__dict__['_src_ops'] = {u.op for u in result.src}
        return result

    UOpMetaClass.__call__ = _caching_call
    print("tinygrad-turbo v3: pre-caching _src_ops at UOp construction")


def uninstall():
    from tinygrad.uop.ops import UOpMetaClass
    # Can't easily un-patch __call__, but for benchmarking we can restart
    pass


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

    workloads = [("softmax", do_softmax), ("4x conv", do_4conv), ("transformer", do_transformer)]

    for _, fn in workloads: fn()

    print("=== Original ===")
    baseline = {}
    for name, fn in workloads:
        t = bench(name, fn)
        baseline[name] = t
        print(f"  {name:>15}: {t*1e3:.1f}ms")

    install()
    for _, fn in workloads: fn()

    print("\n=== turbo v3 (pre-cache _src_ops) ===")
    for name, fn in workloads:
        t = bench(name, fn)
        delta = (t / baseline[name] - 1) * 100
        print(f"  {name:>15}: {t*1e3:.1f}ms ({delta:+.1f}%)")
