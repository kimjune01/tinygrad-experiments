"""Dump tinygrad's actual realize outputs for correctness comparison.

Runs each workload and captures the output tensor values. Used by compat.py
to verify that the filtered version produces identical results.
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

import numpy as np


def extract_outputs(workloads: dict[str, callable]) -> dict[str, str]:
    """Run each workload, return {name: sha256 of output bytes}."""
    from tinygrad import Tensor

    results = {}
    for name, fn in workloads.items():
        # Capture the last realized tensor's data
        out = _capture_output(fn)
        if out is not None:
            h = hashlib.sha256(out.tobytes()).hexdigest()
            results[name] = h
        else:
            results[name] = "NO_OUTPUT"
    return results


def _capture_output(fn):
    """Run fn and return the numpy array of the result."""
    from tinygrad import Tensor

    # Patch realize to capture the output
    _orig = Tensor.realize
    captured = [None]

    def _capturing_realize(self, *args, **kwargs):
        result = _orig(self, *args, **kwargs)
        captured[0] = self
        return result

    Tensor.realize = _capturing_realize
    try:
        fn()
    finally:
        Tensor.realize = _orig

    if captured[0] is not None:
        return captured[0].numpy()
    return None


if __name__ == "__main__":
    from shapes import WORKLOADS

    print("Extracting output hashes from current tinygrad:\n")
    hashes = extract_outputs(WORKLOADS)

    for name, h in hashes.items():
        print(f"  {name:>25}: {h[:16]}...")

    # Save to file for compat.py
    with open("baseline_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nSaved to baseline_hashes.json")
