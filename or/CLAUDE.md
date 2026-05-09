# tinygrad-or

Operations research lens on tinygrad's compiler optimization problems. Each tinygrad bottleneck is mapped to its OR formulation, the best known OR algorithm is identified, and applicability is validated against the proof manual's kill conditions.

## Structure

- `hypothesis-graph.md` — the living hypothesis graph (fan-out cycle results, proof manual validation, pruning log)
- `formulations/` — one file per OR formulation mapping a tinygrad problem to its OR equivalent
- `prototypes/` — minimal Python implementations of candidate algorithms against tinygrad's actual IR
- `evidence/` — benchmark results, profiling data, before/after comparisons

## Relationship to tinygrad

This repo does not modify tinygrad. It produces:
1. Formulations that map tinygrad problems to OR problems
2. Prototypes that demonstrate feasibility
3. Evidence that validates or kills hypotheses

Successful prototypes are candidates for upstream PRs to tinygrad/tinygrad.

## Proof manual integration

Every hypothesis is validated against the proof manual (june.kim/the-proof-manual) before implementation:
- Classify claim type and domain
- Grid lookup for candidate techniques
- Check kill conditions
- Check symmetry mismatches
- Escalation path on failure
