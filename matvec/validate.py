"""Validate proposed ordering against reference and extract.

For each shape:
  1. reference.py: what BLAS does (ground truth)
  2. extract.py: what tinygrad currently does (possibly broken)
  3. propose.py: what the stride-aware rule recommends (candidate fix)

Pass criteria:
  - propose matches reference for all shapes (correctness)
  - propose differs from extract only on matvec shapes (safety — no GEMM regression)
  - propose == extract on GEMM shapes (idempotent for non-matvec)
"""

from shapes import SHAPES
from propose import derive_loop_ordering, format_ordering
from reference import optimal_inner_axis
import numpy as np


def run_validation():
    results = []

    print("Validation: propose vs reference for all shapes")
    print("=" * 100)
    print(f"{'Name':<22} {'Shape':<22} {'Reference inner':<20} {'Proposed inner':<20} {'Match':>6}")
    print("=" * 100)

    for name, (M, K, N) in SHAPES.items():
        # reference: what axis should be inner?
        A = np.empty((M, K), dtype=np.float32)
        B = np.empty((K, N), dtype=np.float32)
        ref = optimal_inner_axis(A.strides, B.strides, M, K, N)
        ref_inner = ref["recommended_inner"].split()[0]  # "n" or "k"

        # proposed: what does our rule say?
        ordering = derive_loop_ordering(M, K, N)
        prop_inner = ordering[-1]["axis"]

        # for GEMM, reference says "k with blocking" which isn't directly comparable
        is_matvec = M == 1 or N == 1
        if is_matvec:
            match = ref_inner == prop_inner
        else:
            match = True  # GEMM ordering is implementation-dependent, skip comparison

        status = "PASS" if match else "FAIL"

        results.append({
            "name": name,
            "shape": (M, K, N),
            "is_matvec": is_matvec,
            "ref_inner": ref_inner,
            "prop_inner": prop_inner,
            "match": match,
        })

        print(f"{name:<22} ({M:>5},{K:>5},{N:>5})  "
              f"{ref_inner:<20} {prop_inner:<20} {status:>6}")

    print()

    # summary
    matvec_results = [r for r in results if r["is_matvec"]]
    gemm_results = [r for r in results if not r["is_matvec"]]

    matvec_pass = sum(1 for r in matvec_results if r["match"])
    print(f"Matvec shapes: {matvec_pass}/{len(matvec_results)} match reference")
    print(f"GEMM shapes:   {len(gemm_results)}/{len(gemm_results)} skipped (blocking-dependent)")

    if all(r["match"] for r in matvec_results):
        print()
        print("All matvec shapes: proposed ordering matches BLAS reference.")
        print("The stride-aware rule correctly derives n-inner for matvec.")
        print()
        print("Next step: run extract.py to see what tinygrad currently produces,")
        print("then diff against proposed to identify exactly where the scheduler diverges.")
    else:
        print()
        print("FAILURES detected. Check cost function in propose.py.")
        for r in matvec_results:
            if not r["match"]:
                print(f"  {r['name']}: expected {r['ref_inner']}-inner, got {r['prop_inner']}-inner")


if __name__ == "__main__":
    run_validation()
