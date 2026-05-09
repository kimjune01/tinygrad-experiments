"""Ground truth: tinygrad's current PatternMatcher.rewrite behavior.

Extracts the exact match/skip decisions for a set of workloads so we can verify
that the bloom-filter prototype produces identical results (minus the measured
false-skip budget).
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import RewriteContext, PatternMatcher, UOp, Ops

def node_signature(node: UOp) -> tuple:
    return (node.op, node.dtype, len(node.src), frozenset(s.op for s in node.src))

_orig_unified = RewriteContext.unified_rewrite

def collect_ground_truth(build_fn, *, warmup=True):
    """Run build_fn, return per-node match decisions from bpm_cache.

    Returns list of dicts: {pm_id, pm_size, sig, matched, node_op, node_dtype}
    """
    records = []

    def _instrumented(self, root):
        result = _orig_unified(self, root)
        if self.bpm:
            pm_id = id(self.bpm)
            pm_size = len(self.bpm.patterns)
            for node, val in self.bpm_cache.items():
                records.append({
                    "pm_id": pm_id,
                    "pm_size": pm_size,
                    "sig": node_signature(node),
                    "matched": val is not None,
                    "node_op": node.op,
                    "node_dtype": node.dtype,
                })
        return result

    RewriteContext.unified_rewrite = _instrumented
    try:
        if warmup:
            build_fn()
            records.clear()
        build_fn()
    finally:
        RewriteContext.unified_rewrite = _orig_unified

    return records


# ── Workloads ──────────────────────────────────────────────

def workload_softmax():
    from tinygrad import Tensor
    Tensor.randn(32, 128).softmax().realize()

def workload_matmul_softmax():
    from tinygrad import Tensor
    x, w = Tensor.randn(64, 64), Tensor.randn(64, 64)
    (x @ w).softmax().realize()

def workload_conv_relu():
    from tinygrad import Tensor
    x = Tensor.randn(1, 64, 56, 56)
    w = Tensor.randn(64, 64, 3, 3)
    x.conv2d(w, padding=1).relu().realize()

def workload_deep_conv(n_layers=4):
    from tinygrad import Tensor
    x = Tensor.randn(1, 64, 56, 56)
    for _ in range(n_layers):
        w = Tensor.randn(64, 64, 3, 3)
        x = x.conv2d(w, padding=1).relu()
    x.realize()

def workload_transformer():
    from tinygrad import Tensor
    B, T, C = 1, 32, 128
    x = Tensor.randn(B, T, C)
    q, k, v = x @ Tensor.randn(C, C), x @ Tensor.randn(C, C), x @ Tensor.randn(C, C)
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    (attn.softmax() @ v).realize()

ALL_WORKLOADS = {
    "softmax": workload_softmax,
    "matmul_softmax": workload_matmul_softmax,
    "conv_relu": workload_conv_relu,
    "deep_conv_4": lambda: workload_deep_conv(4),
    "transformer": workload_transformer,
}


if __name__ == "__main__":
    for name, fn in ALL_WORKLOADS.items():
        records = collect_ground_truth(fn)
        total = len(records)
        matched = sum(1 for r in records if r["matched"])
        sigs = len({r["sig"] for r in records})
        print(f"{name:>20}: {total:5d} checks, {matched:4d} matched, {sigs:4d} unique sigs")
