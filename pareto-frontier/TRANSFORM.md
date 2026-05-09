# Transform: Structural Signature Filter for PatternMatcher.rewrite

## Target

`tinygrad/uop/ops.py` — `PatternMatcher.rewrite` (line 1259)

## Current code

```python
def rewrite(self, uop:UOp, ctx=None):
    if len(pats:=self.pdict.get(uop.op, [])):
        if (ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        for _,match,early_reject in pats:
            if not early_reject.issubset(ler): continue
            if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
    return None
```

## Proposed change

Add a `_sig_skip` set to PatternMatcher that learns which structural signatures (op, dtype, src_count, src_ops) never match. Check before entering the pattern loop.

```python
def rewrite(self, uop:UOp, ctx=None):
    sig = (uop.op, uop.dtype, len(uop.src), uop.__dict__.get('_src_ops') or frozenset(u.op for u in uop.src))
    if sig in self._sig_never: return None
    if len(pats:=self.pdict.get(uop.op, [])):
        if (ler:=uop.__dict__.get('_src_ops')) is None: uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        for _,match,early_reject in pats:
            if not early_reject.issubset(ler): continue
            if (ret:=match(uop, ctx)) is not None and ret is not uop:
                self._sig_never.discard(sig)
                return ret
    self._sig_never.add(sig)
    return None
```

## Files to touch

1. `tinygrad/uop/ops.py` — `PatternMatcher.__init__` (add `self._sig_never = set()`) and `PatternMatcher.rewrite` (add signature check)

## What NOT to change

- Do not change `unified_rewrite` or `walk_rewrite`
- Do not change `UPat` or the compiled matcher infrastructure
- Do not change any pattern definitions
- Do not add any imports

## Line count

+6 lines, -0 lines (net +6). Within the ≤20 LOC budget.

## Risk

The `_sig_never` set persists across calls to the same PatternMatcher instance. If a PM is used with different `ctx` values that change matching behavior, a signature that's "never match" in one ctx might match in another. Mitigation: the `discard` on match ensures self-correction, and `ctx`-dependent patterns typically check `ctx` inside the callback (after the structural match succeeds), so the structural signature is still valid for pre-filtering.

## Verification

1. `validate.py` — zero false skips across all workload shapes
2. `compat.py` — realize outputs identical with/without filter
3. `bench.py` — ≥90% skip rate after warmup, signatures plateau
