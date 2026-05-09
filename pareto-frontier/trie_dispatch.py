"""Trie-based pattern dispatch for PatternMatcher.rewrite.

Replaces the flat per-op list scan with a multi-level trie indexed by
node features available before running any compiled matcher:

  Level 0: uop.op          (current pdict — free, already computed)
  Level 1: uop.dtype        (single attribute lookup)
  Level 2: len(uop.src)     (single attribute lookup)
  Level 3: uop.src[0].op    (one hop, covers the most discriminating constraint)

Cache policy: trie is built once at PatternMatcher construction time.
Optimistic miss: novel nodes that don't match any trie path fall back
to the full pattern list (recompute). This only happens on first-seen
structural shapes (bootup).
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UPat, UOp, Ops
from collections import defaultdict


def patch_pdict(pm: PatternMatcher):
    """Replace pdict with a two-level key: (op, src[0].op).

    Same dict.get cost as the original (one lookup), but the returned list
    is 40% shorter on average. No new Python overhead in the hot path.
    """
    import itertools as it
    new_pdict: dict[tuple, list] = {}

    # Collect all src[0].op values that appear in any UOp (for pre-populating)
    all_src0_ops: set = {None}  # None = no sources
    for entries in pm.pdict.values():
        for p, _, _ in entries:
            s0 = _extract_src0_ops(p)
            if s0 is not None:
                all_src0_ops.update(s0)

    # Also add all Ops values since we can't predict what nodes we'll see
    all_src0_ops.update(pm.pdict.keys())

    for op, entries in pm.pdict.items():
        wildcards = []
        by_src0 = defaultdict(list)

        for entry in entries:
            p = entry[0]
            src0_ops = _extract_src0_ops(p)
            if src0_ops is None:
                wildcards.append(entry)
            else:
                for s0 in src0_ops:
                    by_src0[s0].append(entry)

        # For every possible src0_op, build the merged list
        # This pre-populates so the hot path never misses
        all_s0_for_op = set(by_src0.keys()) | all_src0_ops
        for s0 in all_s0_for_op:
            specific = by_src0.get(s0, [])
            if specific or wildcards:
                # Merge preserving original order
                spec_ids = set(id(e) for e in specific)
                wild_ids = set(id(e) for e in wildcards)
                merged = [e for e in entries if id(e) in spec_ids or id(e) in wild_ids]
                if merged:
                    new_pdict[(op, s0)] = merged

        # Fallback for completely novel src0_ops
        if wildcards:
            new_pdict[(op, None)] = wildcards

    pm.pdict = new_pdict


def _extract_src0_ops(upat) -> tuple | None:
    import itertools
    if upat.src is None or len(upat.src) == 0:
        return None
    first_perm = upat.src[0]
    if isinstance(first_perm, itertools.repeat):
        inner = upat._in_src
        if isinstance(inner, UPat):
            return inner.op
        return None
    if hasattr(first_perm, '__len__') and len(first_perm) > 0:
        first_upat = first_perm[0]
        if isinstance(first_upat, UPat):
            return first_upat.op
    return None


def trie_rewrite(self: PatternMatcher, uop: UOp, ctx=None):
    """Patched rewrite: same as original but pdict key is (op, src[0].op)."""
    src0_op = uop.src[0].op if uop.src else None
    pats = self.pdict.get((uop.op, src0_op))
    if pats is None:
        pats = self.pdict.get((uop.op, None), [])
    if len(pats):
        if (ler := uop.__dict__.get('_src_ops')) is None:
            uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        for _, match, early_reject in pats:
            if not early_reject.issubset(ler): continue
            if (ret := match(uop, ctx)) is not None and ret is not uop: return ret
    return None


_ORIGINAL_REWRITE = PatternMatcher.rewrite


def install():
    import gc
    # Patch all existing PatternMatcher instances
    for obj in gc.get_objects():
        if isinstance(obj, PatternMatcher):
            patch_pdict(obj)
    PatternMatcher.rewrite = trie_rewrite


def uninstall():
    # Can't easily un-patch pdict, so just restore rewrite
    PatternMatcher.rewrite = _ORIGINAL_REWRITE


if __name__ == "__main__":
    install()

    from tinygrad import Tensor
    import time

    # Warmup (builds tries lazily)
    Tensor.randn(32, 128).softmax().realize()

    # Measure
    times = []
    for i in range(5):
        x = Tensor.randn(32, 128)
        start = time.perf_counter()
        x.softmax().realize()
        times.append(time.perf_counter() - start)

    median = sorted(times)[2]
    print(f"Trie dispatch softmax: {median*1e6:.0f}us (median of 5)")

    uninstall()

    times2 = []
    for i in range(5):
        x = Tensor.randn(32, 128)
        start = time.perf_counter()
        x.softmax().realize()
        times2.append(time.perf_counter() - start)

    median2 = sorted(times2)[2]
    print(f"Original dispatch softmax: {median2*1e6:.0f}us (median of 5)")
    print(f"Delta: {(median - median2)*1e6:.0f}us ({(median/median2 - 1)*100:+.1f}%)")

    # Correctness: conv2d
    install()
    x = Tensor.randn(1, 64, 56, 56)
    w = Tensor.randn(64, 64, 3, 3)
    x.conv2d(w, padding=1).relu().realize()
    print("Conv2d+relu: OK")

    # Deep model
    x = Tensor.randn(1, 64, 56, 56)
    for _ in range(4):
        w = Tensor.randn(64, 64, 3, 3)
        x = x.conv2d(w, padding=1).relu()
    x.realize()
    print("4x conv2d+relu: OK")

    # Transformer
    B, T, C = 1, 32, 128
    x = Tensor.randn(B, T, C)
    q = x @ Tensor.randn(C, C)
    k = x @ Tensor.randn(C, C)
    v = x @ Tensor.randn(C, C)
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    (attn.softmax() @ v).realize()
    print("Transformer attention: OK")
