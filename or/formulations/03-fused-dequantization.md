# H4: Fused Dequantization as Amortized Resource Allocation

## tinygrad problem

GGUF inference loads int4 weights, dequantizes to fp16, then runs GEMM. Currently these are separate operations — the fp16 weights materialize in global memory. Fusing dequant into the GEMM kernel eliminates one global memory round-trip (the dominant cost in decode-phase LLM inference).

## OR formulation

**Amortized resource allocation under a step-function budget.**

The fusion decision is a resource allocation problem:
- **Budget:** register file (R registers per thread, determined by target occupancy)
- **Existing allocation:** GEMM inner loop uses R_gemm registers (tile accumulators, operands, loop variables)
- **New allocation:** dequant adds R_dequant registers (scale factor, zero point, unpacked bits, intermediate casts)
- **Constraint:** R_gemm + R_dequant ≤ R (otherwise occupancy drops and throughput tanks)
- **Step-function cost:** register pressure has no effect within an occupancy tier but catastrophic effect at tier boundaries (APRP from H1)

This is a **knapsack variant**: given a fixed register budget (knapsack capacity) determined by the target occupancy tier, pack the dequant operations (items with known register cost) into the GEMM kernel (knapsack) without exceeding capacity.

The twist: the register cost of dequant depends on the weight layout. Offline weight reordering (Ladder/BitBLAS) is a **preprocessing step** that minimizes the register cost of dequant by choosing a layout where scale factors align with tile boundaries and unpacking requires minimal simultaneous live state.

## Classical OR results

| Concept | Application here |
|---|---|
| Knapsack problem | Pack dequant into GEMM register budget |
| Amortized analysis | Dequant cost is amortized across GEMM iterations — per-iteration overhead is small if layout is right |
| Bin packing with item splitting | If dequant doesn't fit in one tile iteration, split it across iterations (software pipelining) |
| Step-function objectives | APRP: cost is 0 below threshold, catastrophic above |

## Proof manual validation

- Claim type: upper_bound × discrete → greedy fusion
- Kill condition: "greedy choice constrains future steps" → **fires** (fusing adds register pressure)
- Escalation: amortized_analysis (offline weight layout minimizes peak register pressure)
- Dependency: requires H1 (APRP-aware scheduling) to be safe

## Key insight: the weight layout is the optimization variable

The register cost of dequant is not fixed — it depends on how weights are laid out in memory:
- **Bad layout:** scale factors are at arbitrary offsets → multiple loads and live values simultaneously
- **Good layout (Ladder):** scale factors align with tile boundaries → one scale per tile, loaded once, reused across the tile

This means the "knapsack" has flexible item sizes. The offline weight reordering is solving a **layout optimization** problem: minimize peak register pressure of the dequant subgraph, subject to the constraint that the layout must be decodable (each weight group maps to a unique scale+offset).

## Implementation path

1. Define dequant as a UOp subgraph: `LOAD(int4_buf) → SHIFT → MASK → SCALE → CAST(fp16)`
2. PatternMatcher rule fuses this subgraph into downstream GEMM kernel
3. APRP check (from H1): if fused kernel register pressure exceeds occupancy threshold, reject fusion and keep separate kernels
4. Offline weight reordering: ensure GGUF loading produces layouts where dequant registers are minimal
