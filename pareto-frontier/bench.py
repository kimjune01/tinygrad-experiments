"""Benchmark: skip rate and wall-clock scaling across workload sizes.

Measures three things:
1. Skip rate (% of bpm checks avoided) — should be ≥90% after warmup
2. Signature count — should plateau (bounded vocabulary)
3. Wall-clock realize time — should decrease with filter
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import RewriteContext
from propose import node_signature


def bench_workload(name, fn, n_warmup=1, n_measure=3):
    _orig = RewriteContext.unified_rewrite

    # Phase 1: warmup (populates signature knowledge)
    sig_table = {}  # (pm_id, sig) -> ever_matched

    def _learn(self, root):
        result = _orig(self, root)
        if self.bpm:
            pm_id = id(self.bpm)
            for node, val in self.bpm_cache.items():
                sig = node_signature(node)
                key = (pm_id, sig)
                if val is not None:
                    sig_table[key] = True
                elif key not in sig_table:
                    sig_table[key] = False
        return result

    RewriteContext.unified_rewrite = _learn
    for _ in range(n_warmup):
        fn()

    # Phase 2: measure with simulated filter
    skip_counts = []
    total_counts = []

    def _measure(self, root):
        result = _orig(self, root)
        if self.bpm:
            pm_id = id(self.bpm)
            skips = 0
            total = 0
            for node, val in self.bpm_cache.items():
                sig = node_signature(node)
                key = (pm_id, sig)
                total += 1
                if key in sig_table and not sig_table[key]:
                    skips += 1
            skip_counts.append(skips)
            total_counts.append(total)
        return result

    RewriteContext.unified_rewrite = _measure

    times = []
    for _ in range(n_measure):
        skip_counts.clear()
        total_counts.clear()
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    RewriteContext.unified_rewrite = _orig

    total_checks = sum(total_counts)
    total_skips = sum(skip_counts)
    skip_rate = total_skips / total_checks * 100 if total_checks else 0
    n_sigs = len(sig_table)
    n_never = sum(1 for v in sig_table.values() if not v)
    median_time = sorted(times)[len(times) // 2]

    return {
        "name": name,
        "checks": total_checks,
        "skips": total_skips,
        "skip_rate": skip_rate,
        "signatures": n_sigs,
        "never_match_sigs": n_never,
        "median_us": median_time * 1e6,
    }


if __name__ == "__main__":
    from shapes import WORKLOADS

    print(f"{'workload':>25} | {'checks':>6} | {'skips':>6} | {'skip%':>6} | {'sigs':>4} | {'never':>5} | {'time_us':>8}")
    print("-" * 85)

    for name, fn in WORKLOADS.items():
        r = bench_workload(name, fn)
        print(f"{r['name']:>25} | {r['checks']:6d} | {r['skips']:6d} | {r['skip_rate']:5.1f}% | {r['signatures']:4d} | {r['never_match_sigs']:5d} | {r['median_us']:8.0f}")
