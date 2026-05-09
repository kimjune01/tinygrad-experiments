# Transformation Design: Contiguous weights + rollout prune for GGUF inference

## The change

Two changes to `tinygrad/llm/model.py`:

1. **Move `.contiguous()` outside `if realize:`** (line 386) — always applies to GGUF weights, breaks fusion between dequant and matmul.
2. **Add `prune=True` to `rollout_jit` only** (line 309) — dequant kernels run once during JIT capture, excluded from replay.

## Where it lives

```python
# Line 308-309 (current):
self.prefill_jit = TinyJit(self.forward)
self.rollout_jit = TinyJit(self.forward)

# Line 308-309 (proposed):
self.prefill_jit = TinyJit(self.forward)
self.rollout_jit = TinyJit(self.forward, prune=True)
```

```python
# Lines 383-387 (current):
nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
# NOTE: without this contiguous, it unpacks the weights from the model every time...
if realize:
    for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
    Tensor.realize(*params)

# Lines 383-387 (proposed):
nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
# contiguous breaks fusion so dequant becomes a separate kernel; prune makes it onetime in rollout
for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
if realize:
    Tensor.realize(*params)
```

## How it works

1. **Contiguous** adds a fusion barrier. Without it, the scheduler fuses Q6K dequant (bitwise ops on raw bytes) INTO the matmul kernel, producing 328-line kernels with 271 scalar byte loads at 3 GB/s. With it, dequant becomes a separate kernel writing to an F16 buffer; the matmul reads clean F16.

2. **Prune on rollout** (`prune_linear`, jit.py:15-23) detects that dequant kernels don't touch input buffers (tokens, start_pos, temperature) and classifies them as "onetime." They execute once during JIT capture and are excluded from replay.

3. **NOT on prefill** — H₁₅ confirmed that prefill prune misclassifies cache-related kernels, breaking multi-turn inference when prompts diverge.

## Results

| Config | Lazy (default) | This fix | REALIZE=1 | llama.cpp |
|--------|---------------|----------|-----------|-----------|
| 1B Q6_K tok/s | 10.5 | **138** | 134 | 341 |
| 1B Q6_K bandwidth | 11 GB/s | **343 GB/s** | 336 GB/s | ~340 GB/s |
| 1B Q4_K_M tok/s | 36.0 | **136** | — | — |
| Device memory | ~1.3 GB | ~2.6 GB | ~2.7 GB | ~1.0 GB |

## Verification

- Output equivalence: lazy == hack == REALIZE across 4 multi-turn sequences (90 tokens), including prefix reuse, divergent prompt, and return to original prefix.
- Q4_K_M output equivalence: 20 tokens, match.
- Peak memory: hack ≈ REALIZE=1 (2598 vs 2640 MB RSS).
- Test suite: 91 passed, 7 skipped (GGUF 41 + JIT 50).

## What NOT to change

- `prefill_jit` — do NOT add `prune=True`. Causes multi-turn cache corruption (H₁₅).
- The GGUF loader (`gguf.py`) — lazy dequant chains are correct tensor API behavior.
- The `REALIZE` env var and its `if realize:` block — keep for explicit eager realization.

## Why prune only on rollout

Rollout handles steady-state decode: T=1, same graph structure every call. All pruned kernels are genuinely onetime weight dequants. Prefill handles variable-length prompt processing with KV cache dependencies that prune can't distinguish from weight computation.

## Why not just REALIZE=1 default

geohot deliberately set `REALIZE=0` in PR #15144 (Mar 5, 2026) — the compiler should handle lazy dequant efficiently. This fix respects that: weights stay lazy until JIT warmup, where prune automatically separates onetime dequant from repeated matmul. No eager realization, no change to the REALIZE contract.

## Provenance

- contiguous on GGUF weights: PR #15082 (Mar 3, geohot), inside `if realize:` block
- prune_linear on JIT: PR #15423 (Mar 23, nimlgen), never applied to LLM model
- REALIZE=0 default: PR #15144 (Mar 5, geohot), "make realize not the default"
- These three features were built by two people across three PRs in 20 days. Nobody connected them.
- Experiment repo: https://github.com/kimjune01/tinygrad-realize-experiment
- Hypothesis graph: ~/Documents/tinygrad/HYPOTHESIS_GRAPH.md (H₁₂, H₁₃, H₁₃', H₁₅–H₁₈)
