# OR ↔ Compiler Optimization: Formulation Map

## The thesis

Every compiler optimization problem has an OR formulation that predates the compiler community's version by 20-50 years. The OR version is often stronger because it was developed in a field that values provable guarantees and worst-case analysis, not just benchmark improvements.

tinygrad is the ideal test case: small enough to hold in your head (~9500 lines), with well-defined bottlenecks traced to specific codegen decisions.

## The map

| tinygrad problem | File | OR formulation | Classical result | Compiler community's version |
|---|---|---|---|---|
| Instruction ordering | [01](01-instruction-scheduling.md) | RCPSP with renewable resources | Hu 1961, Coffman-Graham 1972 | "List scheduling heuristics" (Shobaki et al.) |
| Bank conflicts | [02](02-bank-conflict-avoidance.md) | Assignment over GF(2) | Hungarian algorithm (overkill); GF(2) basis selection | "Swizzled shared memory" (CUTLASS) |
| Fused dequantization | [03](03-fused-dequantization.md) | Knapsack / amortized resource allocation | Knapsack DP, step-function objectives | "Fused quantized GEMM" (Ladder/BitBLAS) |
| Reduction fusion | [04](04-reduction-fusion.md) | Algebraic decomposition under resource constraints | Benders 1962, Dantzig-Wolfe 1960, Lagrangian relaxation | "Online softmax" (FlashAttention), "cascaded reduction fusion" (RedFuser) |
| BEAM search | (deferred) | Abduction engine over discrete optimization space | (novel — this is the contribution) | Beam search, Bayesian optimization |

## Dependency graph

```
H1 (RCPSP scheduling) ──prerequisite──→ H4 (fused dequant)
                       ──prerequisite──→ H5 (algebraic fusion)
H2 (GF(2) assignment) ──independent
```

H1 is the foundation: register-pressure-aware scheduling makes fusion safe. Without it, any fusion (dequant or reduction) risks the over-fusion kill condition.

## The ambiguity heuristic

When two implementations achieve similar performance, prefer fewer lines. A 40-line CPL scheduler that closes 80% of the gap beats a 400-line two-pass scheduler that closes 85% — unless the proof manual's kill conditions say that last 5% matters for the target workload. Lines of code is the tiebreaker, not the objective. But in a codebase that prizes simplicity (~9500 lines total), it's a strong tiebreaker.

## OR theory cross-reference

See [05-or-theory-crossref.md](05-or-theory-crossref.md) for the full lineage: each OR result, its textbook treatment, the compiler community's independent rediscovery, and the gap between them.

## What's novel here

The individual OR results are old. The compiler techniques are recent. What's new:
1. **The mapping itself** — showing that these are the same problems
2. **The proof manual as discriminant** — using kill conditions and escalation paths to predict which techniques apply before prototyping
3. **The dependency structure** — H1 enables H4 and H5; this ordering wasn't visible from the compiler literature alone
4. **The abduction engine** (future work) — replacing blind search (BEAM) with hypothesis-driven exploration is the genuinely novel contribution; everything else is applying known OR to known compiler problems
