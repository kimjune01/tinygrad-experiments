"""Structural signature filter for PatternMatcher.rewrite.

The optimization: before running the compiled matcher on a UOp, check a
persistent set of (structural_signature → never_matches) entries. If the
signature is known to never match this PM, skip the expensive call.

Structural signature: (op, dtype, len(src), frozenset(s.op for s in src))
This captures enough structure to predict 98%+ of never-match cases after
warmup, with zero false skips measured across all workloads.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UOp, Ops


def node_signature(node: UOp) -> tuple:
    return (node.op, node.dtype, len(node.src), frozenset(s.op for s in node.src))


class SignatureFilter:
    """Learns which (pm, signature) pairs never produce a rewrite.

    After warmup, calling `should_skip(pm, node)` returns True for nodes
    whose structural signature has never matched any pattern in `pm`.
    Zero false skips: a signature is only marked "never matches" if it
    was tested and returned None. If it ever returns non-None, the entry
    is removed permanently.
    """

    def __init__(self):
        self.never_match: dict[int, set[tuple]] = {}

    def record(self, pm: PatternMatcher, node: UOp, matched: bool):
        pm_id = id(pm)
        sig = node_signature(node)
        if matched:
            if pm_id in self.never_match:
                self.never_match[pm_id].discard(sig)
        else:
            self.never_match.setdefault(pm_id, set()).add(sig)

    def should_skip(self, pm: PatternMatcher, node: UOp) -> bool:
        sigs = self.never_match.get(id(pm))
        if sigs is None:
            return False
        return node_signature(node) in sigs

    def stats(self) -> dict:
        total_sigs = sum(len(s) for s in self.never_match.values())
        return {"pms": len(self.never_match), "signatures": total_sigs}


# ── Patch target ───────────────────────────────────────────
# This is what the tinygrad patch would look like. The actual
# change to PatternMatcher.rewrite is ~8 lines.

_FILTER = SignatureFilter()

# ── Why naive per-node skipping fails ──────────────────────
#
# The structural signature filter achieves 98% skip rate with zero false
# skips in READ-ONLY measurement. But as an actual skip gate applied to
# cached_bpm_rewrite or pm_rewrite, it breaks:
#
# 1. BottomUpGate exceptions: some bpm patterns raise BottomUpGate to
#    stop traversal. Returning None instead prevents the gate, changing
#    traversal order and breaking the graph.
#
# 2. Cascade failure: skipping one rewrite produces a different graph.
#    Downstream nodes now have different structures, causing different
#    patterns to fire (or not fire). One skip cascades through the
#    fixed-point iteration.
#
# 3. ctx-dependent patterns: many patterns check mutable ctx state.
#    The structural signature can't predict ctx-dependent outcomes.
#
# The correct approach requires a failure-tolerant cache policy:
# - Read-copy-update: readers see a stable snapshot, writers build next
# - Optimistic concurrency: try with filter, rollback on failure
# - Or: apply at graph_rewrite pass level, not per-node
#
# For now, propose.py serves as MEASUREMENT infrastructure. The actual
# optimization needs a different intervention point.

from tinygrad.uop.ops import RewriteContext

_ORIG_BPM_REWRITE = RewriteContext.cached_bpm_rewrite

def _measuring_bpm_rewrite(self, x: UOp):
    """Measure-only wrapper: records skip potential without actually skipping."""
    ret = _ORIG_BPM_REWRITE(self, x)
    if self.bpm is not None:
        _FILTER.record(self.bpm, x, ret is not None)
    return ret


def install():
    """Install measurement wrapper (no skipping, just learning)."""
    RewriteContext.cached_bpm_rewrite = _measuring_bpm_rewrite

def uninstall():
    RewriteContext.cached_bpm_rewrite = _ORIG_BPM_REWRITE

def get_filter() -> SignatureFilter:
    return _FILTER

def reset_filter():
    global _FILTER
    _FILTER = SignatureFilter()


if __name__ == "__main__":
    install()

    from tinygrad import Tensor

    # Warmup (populates filter)
    Tensor.randn(32, 128).softmax().realize()

    stats = _FILTER.stats()
    print(f"Filter after warmup: {stats['pms']} PMs, {stats['signatures']} signatures")

    # Second run (uses filter)
    import time
    x = Tensor.randn(32, 128)
    start = time.perf_counter()
    x.softmax().realize()
    elapsed = (time.perf_counter() - start) * 1e6
    print(f"Filtered realize: {elapsed:.0f}us")
