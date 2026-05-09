"""Proposed stride-aware loop ordering rule.

Pure Python, no tinygrad dependency. Takes a matmul shape + memory layout
and derives the optimal loop nesting order.

The core idea: the inner loop should walk the axis with the smallest stride
in the largest buffer. For matvec, this means the output axis (n) should be
innermost, not the reduction axis (k), because B[K,N] has unit stride on N
but stride-N on K.
"""

import numpy as np
from shapes import SHAPES


def derive_loop_ordering(M, K, N, A_strides=None, B_strides=None, C_strides=None):
    """Derive optimal loop nesting for C[M,N] = A[M,K] @ B[K,N].

    Returns a list of (axis_name, size) from outermost to innermost.

    The rule: for each axis, compute the total stride cost across all buffers.
    The axis with the lowest total stride cost should be innermost.

    Stride cost for an axis = sum of (stride_in_buffer * buffer_size_bytes)
    weighted by how many times that buffer is accessed.

    For matmul:
      - A[M,K] is read N times (once per output column)
      - B[K,N] is read M times (once per output row)
      - C[M,N] is written once

    Simplification for matvec (M=1):
      - A[1,K] is tiny (K elements), read once per output
      - B[K,N] is huge (K*N elements), read once total
      - C[1,N] is tiny (N elements), written once
      - B dominates. Inner loop should minimize B's access stride.
    """
    if A_strides is None:
        A_strides = (K * 4, 4)      # C-order: row-major
    if B_strides is None:
        B_strides = (N * 4, 4)      # C-order: row-major
    if C_strides is None:
        C_strides = (N * 4, 4)      # C-order: row-major

    A_bytes = M * K * 4
    B_bytes = K * N * 4
    C_bytes = M * N * 4

    axes = {
        "m": {"size": M, "A_stride": A_strides[0], "B_stride": 0,            "C_stride": C_strides[0]},
        "k": {"size": K, "A_stride": A_strides[1], "B_stride": B_strides[0], "C_stride": 0},
        "n": {"size": N, "A_stride": 0,            "B_stride": B_strides[1], "C_stride": C_strides[1]},
    }

    axis_costs = {}
    for axis_name, info in axes.items():
        if info["size"] <= 1:
            axis_costs[axis_name] = float('inf')  # degenerate axis, push outermost
            continue

        # cost = weighted stride across all buffers
        # weight = buffer size (larger buffers matter more)
        cost = 0
        if info["A_stride"] > 0:
            cost += info["A_stride"] * (A_bytes / (M * K * 4))  # normalize
        if info["B_stride"] > 0:
            cost += info["B_stride"] * (B_bytes / (K * N * 4))  # normalize
        if info["C_stride"] > 0:
            cost += info["C_stride"] * (C_bytes / (M * N * 4))  # normalize

        axis_costs[axis_name] = cost

    # sort: highest cost outermost, lowest cost innermost
    ordering = sorted(axis_costs.items(), key=lambda x: -x[1])

    return [
        {"axis": name, "size": axes[name]["size"], "cost": cost,
         "strides": {b: axes[name][f"{b}_stride"] for b in ("A", "B", "C")}}
        for name, cost in ordering
    ]


def format_ordering(ordering):
    """Format loop ordering as a readable string."""
    parts = []
    for level in ordering:
        stride_info = ", ".join(f"{b}:{s}" for b, s in level["strides"].items() if s > 0)
        parts.append(f"{level['axis']}({level['size']}) cost={level['cost']:.0f} [{stride_info}]")
    return " > ".join(parts)


if __name__ == "__main__":
    print("Proposed stride-aware loop ordering")
    print("Outer > ... > Inner (lowest stride cost = innermost)")
    print("=" * 100)

    for name, (M, K, N) in SHAPES.items():
        ordering = derive_loop_ordering(M, K, N)
        innermost = ordering[-1]["axis"]

        tag = ""
        if M == 1 or N == 1:
            # for matvec, check if the proposal differs from naive k-inner
            if innermost == "k":
                tag = "  !! SAME AS NAIVE (k-inner) — check cost function"
            elif innermost == "n":
                tag = "  OK: n-inner (unit stride in B)"
            elif innermost == "m":
                tag = "  OK: m-inner (unit stride in A)"

        print(f"{name:<22} ({M:>5},{K:>5},{N:>5})")
        print(f"  {format_ordering(ordering)}{tag}")
        print()

    # demonstrate the stride difference
    print("=" * 100)
    print("Why this matters for matvec_attn_proj (1, 4096, 4096):")
    print()
    B = np.empty((4096, 4096), dtype=np.float32)
    print(f"  B.strides = {B.strides}")
    print(f"  B[k, n]:   k-stride = {B.strides[0]} bytes,  n-stride = {B.strides[1]} bytes")
    print(f"  Ratio: k-stride / n-stride = {B.strides[0] / B.strides[1]:.0f}x")
    print()
    print(f"  k-inner loop: each iteration jumps {B.strides[0]} bytes in B (cache-hostile)")
    print(f"  n-inner loop: each iteration jumps {B.strides[1]} bytes in B (cache-friendly)")
    print(f"  -> n-inner reads B at {B.strides[0] / B.strides[1]:.0f}x higher effective bandwidth")
