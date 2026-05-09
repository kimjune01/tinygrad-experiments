# BEAM in tinygrad: vision vs. reality

## The vision

BEAM is supposed to make custom kernels unnecessary. When a contributor submitted a hand-written matvec kernel to beat PyTorch on CPU (#14630), geohot's response was:

> why can't BEAM find it? i doubt custom kernels are going to qualify for the bounty

The thesis: the compiler finds optimal schedules automatically. Custom kernels are a failure of the search.

`BEAM=2` is the standard performance flag. It's how tinygrad closes gaps with PyTorch on every backend. It's load-bearing infrastructure.

## The reality

BEAM gets maintenance, not investment. The search algorithm hasn't changed structurally.

### What gets merged

Plumbing and reliability:
- `beam uop` / `rm beam uop` (#15660, #15844) — refactoring IR integration
- `beam: add dev_timeout for am` (#15063) — reliability fix
- `fix process_replay Ops.BEAM` (#15752) — test infrastructure

### What doesn't get merged

Attempts to improve the search itself:
- **Elimination-based optimizer** (#14364): 92,416 configurations narrowed to 1 by applying 21 hardware constraints. Response: *"this is AI slop that is not worth considering to merge."*
- **Beam cache key fix** (#11908): cache serves wrong results when env vars change. Three PRs attempted (#15034, #15057, #15515), all closed ("not generic") or went stale. Bug still open as of May 2026.
- **MCTS** (BEAM>=100): tried on flux (#6759), slower than BEAM (825ms vs 594ms). Nobody followed up.

### What gets done instead

When BEAM can't find the right schedule, the fix is always somewhere else:
- Better heuristics (#15599, #15616, #13677) — hand-coded rules for specific kernel shapes
- Custom kernels (#14630) — hand-written assembly for matvec
- Metal-specific tricks (#14595) — GPU wake kernels, spin-wait sync, direct dispatch
- Backend tuning (#14335) — LLVM flags, vec-reduce intrinsics

Each of these works around BEAM's limitations without touching the search.

## The gap

BEAM is a hyperparameter grid search (see [BASELINE.md](BASELINE.md)). The search hasn't gotten smarter — the action pool, the beam pruning, the plateau exit, the proxy measurement — all unchanged. What changes is the surface area: more actions, more heuristic fallbacks, more special cases.

The pattern: everyone agrees BEAM should find the answer. Nobody is making BEAM smarter. The investment goes into everything around it.

## What this means for the abduction engine

The bar is low and the need is acknowledged. geohot wants BEAM to replace custom kernels. BEAM can't because it's grid search. The question isn't whether a smarter search is needed — it's whether abduction is the right kind of smarter.

The existing attempts to improve search (elimination-based optimizer, MCTS) were rejected for being AI slop or underperforming. An abduction engine that demonstrably outperforms BEAM on the existing benchmark suite — fewer trials, no regressions, better final schedules — would be the first improvement to the search itself that anyone has shipped.

## Sources

| # | Type | Title | Key quote / finding |
|---|------|-------|-------------------|
| 14630 | PR | custom matvec kernel for CPU bounty | geohot: "why can't BEAM find it?" |
| 14364 | PR | elimination-based kernel optimizer | geohot: "this is AI slop" |
| 14595 | PR | make test_sum green/yellow on Mac | geohot: "do not use ai" |
| 15599 | PR | CPU matvec heuristic | geohot: "looks vaguely AI" |
| 11908 | Issue | beam cache doesn't invalidate | open since Aug 2025, three fix PRs rejected |
| 15034, 15057, 15515 | PRs | beam cache key fixes | all closed or stale |
| 4301 | Issue | train_gpt2 slow on 7900 XTX | geohot: "we will beat PyTorch and be in striking distance of llm.c" |
| 5049 | Issue | slow lm_head matmul | chenyuxyz: "needs either a better search, or PADTO+LOCAL" |
| 6759 | Issue | flux + MCTS optimization | MCTS slower than BEAM, no follow-up |
| 5809 | Issue | training with MCTS optimized kernels | BEAM>=100 triggers MCTS, cache remembers results |
| 3921 | Issue | BEAM search 70B llama | OOM during search, 34B causes GPU hang |
| 12677 | PR | bring BEAM_PADTO back | geohot: "I think this is still a good idea" |
