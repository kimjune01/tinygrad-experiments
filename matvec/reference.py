"""Ground truth: what numpy/BLAS does for each shape.

For each shape, reports:
  - stride layout of both operands
  - which axis BLAS uses as the inner loop (inferred from performance)
  - achieved bandwidth (to confirm BLAS is actually fast)
"""

import numpy as np
import time
from shapes import SHAPES

def measure_bandwidth(M, K, N, layout="C", warmup=5, trials=20):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    if layout == "F_B":
        B = np.asfortranarray(B)
    elif layout == "F_A":
        A = np.asfortranarray(A)

    for _ in range(warmup):
        np.dot(A, B)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = sorted(times)[len(times) // 2]
    total_bytes = (A.nbytes + B.nbytes + M * N * 4)  # read A, read B, write C
    bandwidth_gbs = total_bytes / median_time / 1e9

    return {
        "shape": (M, K, N),
        "layout": layout,
        "A_strides": A.strides,
        "B_strides": B.strides,
        "A_stride_ratio": A.strides[0] / A.strides[1] if A.strides[1] > 0 else float('inf'),
        "B_stride_ratio": B.strides[0] / B.strides[1] if B.strides[1] > 0 else float('inf'),
        "median_time_us": median_time * 1e6,
        "bandwidth_gbs": bandwidth_gbs,
        "is_matvec": M == 1 or N == 1,
    }


def optimal_inner_axis(A_strides, B_strides, M, K, N):
    """Derive which reduction axis ordering BLAS would prefer.

    For matmul C[m,n] = sum_k A[m,k] * B[k,n]:
      - Inner loop over k: reads A[m, k+1] (stride A[1]) and B[k+1, n] (stride B[0])
      - Inner loop over n: reads B[k, n+1] (stride B[1]) — only valid if we restructure
      - Inner loop over m: reads A[m+1, k] (stride A[0]) — only valid if we restructure

    For standard matmul, k is the reduction axis. The question is whether the k-loop
    should be innermost, or whether the output axes (m, n) should be innermost.

    BLAS convention for C-order:
      - A is row-major: A.strides = (K*4, 4) — stride-1 on K axis
      - B is row-major: B.strides = (N*4, 4) — stride-1 on N axis
      - For matvec (M=1): inner loop over K reads A with stride 4 (good),
        B with stride N*4 (bad if N is large). But A is tiny, B dominates.
        The B access pattern for the k-loop is B[k, n] with stride B[0] = N*4.
        If we instead loop n-inner, k-outer: B[k, n+1] has stride 4 (good).

    So for matvec with C-order B: n should be innermost, not k.
    For GEMM: BLAS picks based on blocking, not simple axis ordering.
    """
    a_k_stride = A_strides[1]  # stride to advance k in A
    b_k_stride = B_strides[0]  # stride to advance k in B
    b_n_stride = B_strides[1]  # stride to advance n in B

    if M == 1 and N <= 1:
        # dot product or scalar: k is the only non-degenerate axis
        return {
            "recommended_inner": "k (only axis)",
            "reason": "dot product: k is the only non-degenerate axis",
            "k_stride_in_B": b_k_stride,
            "n_stride_in_B": b_n_stride,
        }
    elif M == 1:
        # matvec: output is 1×N. Reduction over K.
        # k-inner reads B column-wise: stride = B[0] = N*itemsize (bad for large N)
        # For each output n, we sum A[0,k]*B[k,n] over k.
        # If we process multiple n simultaneously (vectorize over n),
        # we want consecutive n in the inner loop: stride B[1] = itemsize (good).
        return {
            "recommended_inner": "n (output axis)",
            "reason": f"B k-stride={b_k_stride} >> B n-stride={b_n_stride}; "
                      f"vectorize over n for unit-stride B access",
            "k_stride_in_B": b_k_stride,
            "n_stride_in_B": b_n_stride,
        }
    else:
        return {
            "recommended_inner": "k (reduction axis) with blocking",
            "reason": "GEMM: BLAS uses tiled blocking; simple axis ordering insufficient",
            "k_stride_in_B": b_k_stride,
            "n_stride_in_B": b_n_stride,
        }


if __name__ == "__main__":
    print("=" * 90)
    print(f"{'Name':<22} {'Shape':<22} {'Med μs':>8} {'GB/s':>7} {'A strides':<16} {'B strides':<16} {'Inner'}")
    print("=" * 90)

    for name, (M, K, N) in SHAPES.items():
        result = measure_bandwidth(M, K, N)
        inner = optimal_inner_axis(result["A_strides"], result["B_strides"], M, K, N)

        print(f"{name:<22} ({M:>5},{K:>5},{N:>5})  "
              f"{result['median_time_us']:>8.1f} {result['bandwidth_gbs']:>7.1f} "
              f"{str(result['A_strides']):<16} {str(result['B_strides']):<16} "
              f"{inner['recommended_inner'][:20]}")

    print()
    print("Key insight: for matvec (M=1), B's k-stride is N*4 bytes.")
    print("Walking k-inner means strided access over the large weight matrix.")
    print("Walking n-inner means unit-stride access — what BLAS actually does.")
