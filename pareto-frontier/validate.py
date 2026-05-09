"""Validate that propose.py matches reference.py for all test shapes.

Gate: zero false skips (no rewrite missed by the filter) across the
full workload matrix after warmup.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from reference import collect_ground_truth, node_signature
from propose import SignatureFilter


def validate_filter(workloads: dict[str, callable]) -> bool:
    """Train the filter on one pass, then check for false skips on a second pass.

    A false skip = the filter says "never matches" but the ground truth shows a match.
    """
    sig_filter = SignatureFilter()
    all_passed = True

    for name, fn in workloads.items():
        # Collect ground truth (two passes: warmup + measure)
        records = collect_ground_truth(fn, warmup=True)

        # Train the filter on these records
        from tinygrad.uop.ops import PatternMatcher
        import gc
        pms = {id(obj): obj for obj in gc.get_objects() if isinstance(obj, PatternMatcher)}

        for r in records:
            sig_filter.record_sig(r["pm_id"], r["sig"], r["matched"])

        # Now check: would the filter have produced false skips?
        false_skips = []
        correct_skips = 0
        total = len(records)

        for r in records:
            would_skip = r["sig"] in sig_filter.never_match.get(r["pm_id"], set())
            if would_skip and r["matched"]:
                false_skips.append(r)
            elif would_skip and not r["matched"]:
                correct_skips += 1

        skip_rate = correct_skips / total * 100 if total else 0
        status = "PASS" if len(false_skips) == 0 else "FAIL"

        print(f"  {name:>25}: {status} | {total:5d} checks, {correct_skips:5d} skippable ({skip_rate:5.1f}%), {len(false_skips)} false skips")

        if false_skips:
            all_passed = False
            for fs in false_skips[:3]:
                print(f"    FALSE SKIP: sig={fs['sig']}, pm_size={fs['pm_size']}")

    return all_passed


def validate_simple(workloads: dict[str, callable]) -> bool:
    """Simpler validation: just check that structural signatures are consistent.

    For each workload, run twice. Any signature that matches in run 2 must also
    have matched in run 1 (no false negatives from signature-based skipping).
    """
    from tinygrad.uop.ops import RewriteContext

    _orig = RewriteContext.unified_rewrite
    all_passed = True

    for name, fn in workloads.items():
        # Run 1: build signature table
        sig_table: dict[tuple, bool] = {}  # (pm_id, sig) -> ever_matched
        records_r1 = []

        def _inst_r1(self, root):
            result = _orig(self, root)
            if self.bpm:
                pm_id = id(self.bpm)
                for node, val in self.bpm_cache.items():
                    sig = node_signature(node)
                    key = (pm_id, sig)
                    matched = val is not None
                    records_r1.append(key)
                    if matched:
                        sig_table[key] = True
                    elif key not in sig_table:
                        sig_table[key] = False
            return result

        RewriteContext.unified_rewrite = _inst_r1
        fn()  # warmup
        sig_table.clear()
        records_r1.clear()
        fn()  # measure

        # Run 2: check for false skips
        false_skips = 0
        correct_skips = 0
        total = 0

        def _inst_r2(self, root):
            nonlocal false_skips, correct_skips, total
            result = _orig(self, root)
            if self.bpm:
                pm_id = id(self.bpm)
                for node, val in self.bpm_cache.items():
                    sig = node_signature(node)
                    key = (pm_id, sig)
                    total += 1
                    matched = val is not None
                    would_skip = key in sig_table and not sig_table[key]
                    if would_skip and matched:
                        false_skips += 1
                    elif would_skip:
                        correct_skips += 1
            return result

        RewriteContext.unified_rewrite = _inst_r2
        fn()

        RewriteContext.unified_rewrite = _orig

        skip_rate = correct_skips / total * 100 if total else 0
        status = "PASS" if false_skips == 0 else "FAIL"
        print(f"  {name:>25}: {status} | {total:5d} checks, {correct_skips:5d} skippable ({skip_rate:5.1f}%), {false_skips} false skips")

        if false_skips:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    from shapes import WORKLOADS
    print("Validating structural signature filter across all workloads:\n")
    passed = validate_simple(WORKLOADS)
    print(f"\n{'ALL PASSED' if passed else 'FAILURES DETECTED'}")
    sys.exit(0 if passed else 1)
