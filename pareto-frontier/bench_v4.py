"""Benchmark turbo v4: C-native CUOp pattern matching."""
import sys, os, time, gc, itertools
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

import turbo_v4
from tinygrad.uop.ops import PatternMatcher, UPat, UOp, Ops, RewriteContext
from tinygrad import Tensor
from tinygrad.schedule import schedule_cache


def _extract_src0_ops(upat):
    if upat.src is None or len(upat.src) == 0: return None
    all_ops = set()
    has_wild = False
    for perm in upat.src:
        if isinstance(perm, itertools.repeat):
            inner = upat._in_src
            if isinstance(inner, UPat) and inner.op is not None: all_ops.update(int(o) for o in inner.op)
            else: has_wild = True
        elif hasattr(perm, '__len__') and len(perm) > 0:
            first = perm[0]
            if isinstance(first, UPat) and first.op is not None: all_ops.update(int(o) for o in first.op)
            else: has_wild = True
        else: has_wild = True
    if has_wild: return None
    return tuple(all_ops) if all_ops else None


def build_pattern_table(pm, op):
    """Build a C-compatible pattern table for one op's entries."""
    entries = pm.pdict.get(op, [])
    table = []
    for entry in entries:
        upat = entry[0]
        req_len = upat.required_len
        strict = 1 if upat.strict_length else 0
        s0_ops = _extract_src0_ops(upat)
        dtype_ids = tuple(id(dt) for dt in upat.match_dtype) if upat.match_dtype else None
        arg_val = upat.arg if isinstance(upat.arg, int) else 0
        has_arg = 1 if isinstance(upat.arg, int) else 0
        table.append((req_len, strict, s0_ops, dtype_ids, arg_val, has_arg))
    return table


_ORIG_REWRITE = PatternMatcher.rewrite
_pm_tables = {}  # (id(pm), op) -> pattern_table


def turbo_rewrite(self, uop, ctx=None):
    """v4 rewrite: one C call for flatten+match, Python callback on hit."""
    pats = self.pdict.get(uop.op)
    if not pats:
        return None

    # Get or build pattern table for this (pm, op)
    key = (id(self), uop.op)
    table = _pm_tables.get(key)
    if table is None:
        table = build_pattern_table(self, uop.op)
        _pm_tables[key] = table

    # ONE C call: flatten + match (104ns vs 625ns Python)
    idx = turbo_v4.rewrite_one(uop, table)

    if idx == -1:
        return None

    if idx >= len(pats):
        return None

    _, match, early_reject = pats[idx]
    if early_reject:
        if (ler := uop.__dict__.get('_src_ops')) is None:
            uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        if not early_reject.issubset(ler):
            return _fallback(self, uop, ctx, idx + 1)

    ret = match(uop, ctx)
    if ret is not None and ret is not uop:
        return ret

    return _fallback(self, uop, ctx, idx + 1)


def _fallback(self, uop, ctx, start):
    entries = self.pdict.get(uop.op, [])
    if (ler := uop.__dict__.get('_src_ops')) is None:
        uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
    for _, match, early_reject in entries[start:]:
        if not early_reject.issubset(ler): continue
        if (ret := match(uop, ctx)) is not None and ret is not uop: return ret
    return None


def install():
    PatternMatcher.rewrite = turbo_rewrite
    _pm_tables.clear()


def uninstall():
    PatternMatcher.rewrite = _ORIG_REWRITE


# ── Benchmark ──

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

for _, fn in workloads: fn()

print("=== Original ===")
baseline = {}
for name, fn in workloads:
    t = bench(name, fn)
    baseline[name] = t
    print(f"  {name:>15}: {t*1e3:.1f}ms")

install()
for _, fn in workloads: fn()

print("\n=== turbo v4 (CUOp + C matching) ===")
for name, fn in workloads:
    t = bench(name, fn)
    delta = (t / baseline[name] - 1) * 100
    print(f"  {name:>15}: {t*1e3:.1f}ms ({delta:+.1f}%)")

uninstall()
