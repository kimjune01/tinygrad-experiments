"""Correctness gate: realize outputs must be identical with and without the filter.

Runs each workload twice — once with vanilla tinygrad, once with the signature
filter installed — and compares output tensors element-wise.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

import numpy as np


def check_compat(workloads: dict[str, callable]) -> bool:
    from tinygrad import Tensor

    all_passed = True

    for name, fn in workloads.items():
        # Run 1: vanilla
        np.random.seed(42)
        from tinygrad.helpers import Context
        out_vanilla = _run_and_capture(fn, seed=42)

        # Run 2: with filter
        import propose
        propose.reset_filter()
        propose.install()

        # Warmup the filter
        _run_and_capture(fn, seed=0)

        # Now run with filter active
        out_filtered = _run_and_capture(fn, seed=42)

        propose.uninstall()

        if out_vanilla is None or out_filtered is None:
            status = "SKIP"
            detail = "no output captured"
        elif np.array_equal(out_vanilla, out_filtered):
            status = "PASS"
            detail = f"shape={out_vanilla.shape}"
        else:
            max_diff = np.max(np.abs(out_vanilla - out_filtered))
            status = "FAIL"
            detail = f"max_diff={max_diff:.2e}"
            all_passed = False

        print(f"  {name:>25}: {status} | {detail}")

    return all_passed


def _run_and_capture(fn, seed=42):
    """Run fn with a fixed seed, return the numpy output."""
    from tinygrad import Tensor

    Tensor.manual_seed(seed)
    _orig = Tensor.realize
    captured = [None]

    def _cap(self, *args, **kwargs):
        result = _orig(self, *args, **kwargs)
        captured[0] = self
        return result

    Tensor.realize = _cap
    try:
        fn()
    finally:
        Tensor.realize = _orig

    if captured[0] is not None:
        return captured[0].numpy()
    return None


if __name__ == "__main__":
    from shapes import WORKLOADS
    print("Compatibility check: vanilla vs filtered outputs:\n")
    passed = check_compat(WORKLOADS)
    print(f"\n{'ALL PASSED' if passed else 'FAILURES DETECTED'}")
    sys.exit(0 if passed else 1)
