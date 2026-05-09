# H1: Instruction Scheduling as RCPSP

## tinygrad problem

The linearizer (codegen/late/linearizer.py, ~53 lines) orders UOps for execution. Current approach: priority-based topological sort with flat priorities `{LOAD:-1, ALU:0, STORE:+1}`. Produces poor instruction interleaving — random memory barriers improved matmul 1.9x on M1 (tinygrad#1477).

## OR formulation

**Resource-Constrained Project Scheduling Problem (RCPSP).**

Given:
- A DAG `G = (V, E)` where `V` = UOps, `E` = data dependencies
- Processing times `p_j` (latency per op: loads ~100-400 cycles, ALU ~1-4 cycles, stores ~1 cycle)
- A renewable resource: **registers**, with capacity `R` (determined by target occupancy)
- Each op `j` uses `r_j = 1` register from definition to last use

Minimize: **makespan** `C_max = max(C_j)` subject to:
- Precedence: if `(i,j) ∈ E` then `C_i + p_i ≤ S_j`
- Resource: at any time `t`, `Σ{r_j : j active at t} ≤ R`

This is RCPSP with one renewable resource. NP-hard in general, but:
- Polynomial for trees (Sethi-Ullman 1970)
- Strong heuristics exist for DAGs (Hu 1961, Coffman-Graham 1972)
- The resource constraint has step-function cost (APRP): `R` jumps at occupancy tier boundaries

## Classical OR results

| Algorithm | Guarantee | Complexity | Lines (est.) |
|---|---|---|---|
| Hu (1961) / CPL priority | Optimal for unit-time tree DAGs; 2-approx for general DAGs with 1 resource | O(V + E) | ~30 |
| Coffman-Graham (1972) | Optimal for unit-time DAGs with 2 processors; good heuristic for bounded resources | O(V² log V) | ~60 |
| Sethi-Ullman (1970) | Optimal register count for expression trees | O(V) | ~25 |
| RCPSP branch-and-bound | Exact for small instances | Exponential | Too complex |

## Proof manual validation

- Claim type: upper_bound × discrete → greedy is first candidate
- Kill condition: "greedy choice constrains future steps" → **fires** (scheduling a load early increases register pressure)
- Escalation: greedy → potential_method → **APRP ceiling** (Shobaki et al. TACO 2022)
- Revised approach: two-pass (RP-minimizing pass → CPL pass with APRP ceiling)

## What the compiler community missed

1. **RCPSP polyhedral relaxations.** OR has LP relaxations for RCPSP (Artigues et al., 2003; Koné et al., 2011) that give tight lower bounds. These could evaluate schedule quality without exhaustive search.
2. **RCPSP with step-function objectives.** The APRP structure (register pressure has step-function cost at occupancy boundaries) is a special case of RCPSP with generalized resource constraints (Hartmann & Briskorn, 2022). OR has exact algorithms for this when the number of steps is small (which it is — typically 4-8 occupancy tiers).
3. **Priority rule heuristics.** OR's systematic comparison of priority rules for RCPSP (Kolisch 1996, Hartmann 2002) identified that **Latest Finish Time (LFT)** priority outperforms CPL on RCPSP benchmarks. LFT = deadline minus remaining processing time. This has not been tested in compiler scheduling.

## Implementation sketch

```
Phase 1: Compute CPL for each node (backward DFS)
Phase 2: Compute LUC for each node (count last-uses among successors)
Phase 3: Compute APRP ceiling from target GPU's occupancy table
Phase 4: Schedule by CPL priority, with LUC tie-breaking,
         constrained by APRP ceiling on live register count
```
