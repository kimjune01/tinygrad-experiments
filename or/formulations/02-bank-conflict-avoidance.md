# H2: Bank Conflict Avoidance as Assignment over GF(2)

## tinygrad problem

`DEFINE_LOCAL` creates a flat shared memory array. Indexing is whatever falls out of axis transformations — no bank conflict avoidance. 32 shared memory banks; if multiple threads in a warp access the same bank, accesses serialize. IREE measured 27.9% throughput regression when swizzle was removed.

## OR formulation

**Assignment problem with algebraic structure.**

Given:
- 32 threads in a warp, each accessing shared memory at address `addr_t` for thread `t`
- 32 memory banks, where `bank(addr) = addr mod 32` (equivalently, bits [1:5] of the byte address for 4-byte elements)
- Find a bijection `σ: addr → addr'` such that `bank(σ(addr_t)) ≠ bank(σ(addr_s))` for all `t ≠ s` in the same warp

This is a perfect matching in a bipartite graph (threads × banks). In general, finding conflict-free mappings is the assignment problem.

**But the structure is richer.** Addresses are binary vectors. The XOR-swizzle operates on address bits as a linear transformation over GF(2). This means:
- The space of all possible swizzle patterns is a vector space over GF(2)
- The optimal swizzle is computable by linear algebra (basis selection in the complement of the thread-stride subspace)
- The solution is not a heuristic — it's the unique optimal element in the feasible set (up to basis choice)

## Classical OR / algebra results

| Result | What it gives |
|---|---|
| Assignment problem (Kuhn 1955, Hungarian algorithm) | Optimal matching in O(n³) — overkill here since GF(2) structure gives O(1) |
| GF(2) linear algebra (Triton, Zhou et al. 2025) | Swizzle = binary matrix multiply on address bits. Optimal in O(b³) where b = address bits |
| CuTe Swizzle<B,M,S> (CUTLASS) | Parameterized closed-form: 3 integers determine the XOR mask |

## Proof manual validation

- Claim type: construction × algebraic
- Kill condition: "access pattern has no exploitable group structure" → does NOT fire for GEMM tiles (regular stride pattern)
- Kill condition fires for: scatter/gather, data-dependent indexing, irregular reductions
- Fallback for 1D patterns: coprime-stride mapping (Berney & Sitchinava, SPAA 2025)

## Implementation

The entire solution is one formula:
```python
def swizzle(idx, BBits, MBase, SShift):
    mask = (1 << BBits) - 1
    return idx ^ (((idx >> (MBase + SShift)) & mask) << MBase)
```

Parameter selection from tile dimensions:
```python
MBase = log2(vector_length)          # e.g., 4 for float16 with 8-wide vectors
BBits = log2(128 // element_bytes) - MBase  # e.g., 3 for 128B swizzle
SShift = log2(fast_dim_elements) - MBase
```
