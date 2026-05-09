# Scheduling Overhead: the actual bench_perceive bottleneck

Date: 2026-05-07

## Measurement

Softmax (256, 4096) on Metal:
- GPU kernel time: 42us (3 kernels: 15 + 18 + 9)
- Total wall time: 2243us
- Python scheduling overhead: 2202us (98%)

GPU was at 338 MHz, 91% idle. The workload finishes so fast the GPU never ramps up.

## Where the time goes

Issue #13488 "Scheduler performance" breaks down a ResNet50 schedule:

```
model tensor:      63ms
model schedule:   441ms
model rewrite:   1945ms  ← 73% of total
model linearize:  151ms
model verify:      61ms
```

The dominant cost is `graph_rewrite` — pattern matching on the UOp graph.
Top offenders (at ResNet50 scale):

| Pattern | Time | Status |
|---------|------|--------|
| `expand_index` | 519ms | Fixed (PR #13473) |
| `simplify_valid_load` | 475ms | Open |
| `fold_divmod_general` | 350ms | Reduced ~300ms (PR #13483) |
| `simplify_valid` | 216ms | Reduced ~150ms (PR #13485) |
| `simplify_merge_adjacent` | 153ms | Open |

## Active work

This is a known bottleneck under active optimization:
- PR #13355 "Python speed" (geohot) — skip tracing overhead in hot path
- PR #15987 "cleaner and faster run_linear" — realize pipeline cleanup
- PRs #15960, #15993, #16071 "llama speed" series — ongoing as of 2026-05-07

## Implication for OR hypotheses

The scheduling overhead is proportional to pattern count × graph size, not kernel
count. Fusing softmax from 3 kernels to 1 (H5) saves ~25us of GPU time but doesn't
touch the 2200us of graph_rewrite. The OR hypotheses target kernel quality — which
is already good. The bottleneck is the compiler itself.

At LLM scale (large graphs, many tokens), kernel execution dominates and the OR
hypotheses become relevant. At bench_perceive scale (tiny tensors), the compiler
overhead dominates.

## References

- Issue: https://github.com/tinygrad/tinygrad/issues/13488
- PR #13355: https://github.com/tinygrad/tinygrad/pull/13355
- PR #15987: https://github.com/tinygrad/tinygrad/pull/15987
- PR #14856: schedule_step batching (no Metal improvement — "Metal driver overhead dominates")
