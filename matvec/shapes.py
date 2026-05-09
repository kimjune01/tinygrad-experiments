"""Test matrix of matmul shapes for loop ordering experiments.

Each shape is (M, K, N) where the matmul computes C[M,N] = A[M,K] @ B[K,N].
Matvec: M=1 or N=1. GEMM: M,N >> 1.
"""

SHAPES = {
    # matvec: the broken cases
    "matvec_attn_proj":   (1, 4096, 4096),     # LLaMA 8B attention Q/K/V/O projection, decode T=1
    "matvec_ffn_up":      (1, 4096, 14336),     # LLaMA 8B FFN gate/up projection
    "matvec_ffn_down":    (1, 14336, 4096),     # LLaMA 8B FFN down projection
    "matvec_output":      (1, 4096, 128256),    # LLaMA 8B output head (vocab projection)
    "matvec_small":       (1, 2048, 8192),      # LLaMA 1B FFN
    "vecmat":             (1, 4096, 4096),      # same shape, but we'll test B transposed

    # GEMM: should not regress
    "gemm_square":        (4096, 4096, 4096),   # square GEMM (TC territory)
    "gemm_tall":          (128, 4096, 4096),    # prefill chunk (128 tokens)
    "gemm_short":         (32, 4096, 14336),    # prefill chunk (32 tokens) FFN
    "gemm_small_square":  (256, 256, 256),      # small GEMM

    # edge cases
    "thin_m":             (4, 4096, 4096),      # very small batch, not quite matvec
    "thin_n":             (4096, 4096, 4),       # narrow output
    "single_element":     (1, 1, 1),            # degenerate
    "vector_dot":         (1, 4096, 1),          # dot product
}

# memory layout variants to test
LAYOUTS = {
    "c_order":   "C",       # row-major (default): A.strides = (K*4, 4), B.strides = (N*4, 4)
    "f_order_b": "F_B",     # B is column-major (transposed storage)
    "f_order_a": "F_A",     # A is column-major
}
