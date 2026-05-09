"""Propose: contiguous on all GGUF weights + prune on rollout JIT only.

Two changes to tinygrad/llm/model.py:

1. Move contiguous outside the `if realize:` block (line 386):
   Before:
       if realize:
           for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
           Tensor.realize(*params)
   After:
       for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
       if realize:
           Tensor.realize(*params)

2. Add prune=True to rollout_jit only (line 309):
   Before: self.rollout_jit = TinyJit(self.forward)
   After:  self.rollout_jit = TinyJit(self.forward, prune=True)

   NOT on prefill_jit — H15 confirmed prefill prune breaks multi-turn cache reuse.

Effect: dequant chains (48-142 UOps per param) become separate kernels via the
contiguous fusion barrier. prune_linear classifies them as onetime (they don't
touch input buffers). Dequant runs once during JIT warmup, excluded from replay.

Result: 10.5 → 138 tok/s on LLaMA 3.2 1B Q6_K, M5 Max (13.1x speedup).
"""

FIX_FILE = "tinygrad/llm/model.py"

CHANGE_1 = {
    "description": "Move contiguous outside if realize: block",
    "line": 386,
    "old": """if realize:
    for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
    Tensor.realize(*params)""",
    "new": """for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
if realize:
    Tensor.realize(*params)""",
}

CHANGE_2 = {
    "description": "Add prune=True to rollout_jit only",
    "line": 309,
    "old": "self.rollout_jit = TinyJit(self.forward)",
    "new": "self.rollout_jit = TinyJit(self.forward, prune=True)",
}
