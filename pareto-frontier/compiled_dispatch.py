"""Compiled Huffman dispatch tree for PatternMatcher.rewrite.

Generates a single Python function per PatternMatcher at construction time.
The function is a Huffman-shaped if-else tree where:
- Frequent outcomes (no patterns for this op) are early returns
- Single-pattern ops get a direct match call (no loop)
- Multi-pattern ops get an inlined sequence of match calls (no loop overhead)

The CPU's branch predictor learns the hot paths after warmup.
CPython's COMPARE_OP + POP_JUMP_IF_FALSE is its fastest dispatch path.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UOp, Ops


def compile_per_op_matchers(pm: PatternMatcher) -> dict:
    """Generate a compiled match function per op. Keep dict dispatch (already O(1)).

    Eliminates: Python for-loop, tuple unpacking, loop variable, early_reject
    interleaved with match calls. Each per-op function is a straight-line
    sequence of match calls with early returns.
    """
    compiled = {}

    for op, entries in pm.pdict.items():
        lines = []
        namespace = {}

        if len(entries) == 1:
            # Single pattern: direct call, no loop
            _, match, er = entries[0]
            namespace['_m'] = match
            namespace['_er'] = er
            lines.append("def _f(uop, ctx):")
            if er:
                lines.append("  if (ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}")
                lines.append("  if not _er.issubset(ler): return None")
            lines.append("  ret = _m(uop, ctx)")
            lines.append("  return ret if ret is not None and ret is not uop else None")

        else:
            # Multi-pattern: inlined sequence, no loop
            lines.append("def _f(uop, ctx):")
            lines.append("  if (ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}")
            for i, (_, match, er) in enumerate(entries):
                namespace[f'_m{i}'] = match
                namespace[f'_er{i}'] = er
                if er:
                    lines.append(f"  if _er{i}.issubset(ler):")
                    lines.append(f"    ret = _m{i}(uop, ctx)")
                    lines.append(f"    if ret is not None and ret is not uop: return ret")
                else:
                    lines.append(f"  ret = _m{i}(uop, ctx)")
                    lines.append(f"  if ret is not None and ret is not uop: return ret")
            lines.append("  return None")

        code = "\n".join(lines)
        exec(compile(code, f"<match_{op.name}_{len(entries)}p>", "exec"), namespace)
        compiled[op] = namespace['_f']

    return compiled


_ORIGINAL_REWRITE = PatternMatcher.rewrite


def compiled_rewrite(self, uop, ctx=None):
    """Dict dispatch (O(1)) + compiled per-op function (no loop)."""
    if not hasattr(self, '_compiled_ops'):
        self._compiled_ops = compile_per_op_matchers(self)
    fn = self._compiled_ops.get(uop.op)
    if fn is None: return None
    return fn(uop, ctx)


def install():
    PatternMatcher.rewrite = compiled_rewrite


def uninstall():
    PatternMatcher.rewrite = _ORIGINAL_REWRITE


if __name__ == "__main__":
    install()

    from tinygrad import Tensor
    import time

    # Warmup
    Tensor.randn(32, 128).softmax().realize()
    print("softmax: OK")

    x = Tensor.randn(1, 64, 56, 56)
    w = Tensor.randn(64, 64, 3, 3)
    x.conv2d(w, padding=1).relu().realize()
    print("conv2d+relu: OK")

    x = Tensor.randn(1, 64, 56, 56)
    for _ in range(4):
        w = Tensor.randn(64, 64, 3, 3)
        x = x.conv2d(w, padding=1).relu()
    x.realize()
    print("4x conv2d+relu: OK")

    B, T, C = 1, 32, 128
    x = Tensor.randn(B, T, C)
    q, k, v = x @ Tensor.randn(C, C), x @ Tensor.randn(C, C), x @ Tensor.randn(C, C)
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    (attn.softmax() @ v).realize()
    print("transformer: OK")

    # Benchmark
    from tinygrad.schedule import schedule_cache

    def bench(label, fn, n=5):
        times = []
        for _ in range(n):
            schedule_cache.clear()
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return sorted(times)[n // 2]

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

    print("\n=== Wall-clock (cold, schedule cache cleared) ===\n")

    t1 = bench("compiled 4conv", do_4conv)
    uninstall()
    t2 = bench("original 4conv", do_4conv)
    print(f"4x conv:     compiled={t1*1e3:.1f}ms  orig={t2*1e3:.1f}ms  delta={((t1/t2)-1)*100:+.1f}%")

    install()
    t3 = bench("compiled transformer", do_transformer)
    uninstall()
    t4 = bench("original transformer", do_transformer)
    print(f"transformer: compiled={t3*1e3:.1f}ms  orig={t4*1e3:.1f}ms  delta={((t3/t4)-1)*100:+.1f}%")
