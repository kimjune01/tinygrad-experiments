# Draft CPython Patch: Skip inlining at polymorphic call sites

## Problem

When the JIT traces a loop that calls different functions per iteration (polymorphic dispatch), the trace records one callee and guards on it with `_GUARD_IP__PUSH_FRAME`. On the next iteration, a different callee arrives, the guard fails, and the trace exits. The side exit warms up and starts a new trace that records the SECOND callee — which fails on the THIRD callee. The cycle never converges.

## Proposed change

In `Python/optimizer.c`, when recording a `_PUSH_FRAME` uop at a call site that arrived via a side exit from a previous trace (chain_depth > 0), skip callee inlining. Instead, let the trace continue past the call+return without following into the callee's bytecode.

The effect: the loop trace completes its back-edge and gets JIT-compiled. The callee calls happen through the interpreter (not inlined), but the loop body (early_reject check, return-value check, iteration control) runs as native code.

## Where to change

`Python/optimizer.c`, around line 1047-1055 (the `_PUSH_FRAME` handling during trace recording):

```c
else if (uop == _PUSH_FRAME) {
    _PyJitTracerTranslatorState *ts_depth = &tracer->translator_state;
    
    // NEW: if this is a chained trace (came from a side exit at this call site),
    // the call site is likely polymorphic. Don't inline — trace around it.
    if (tracer->initial_state.chain_depth > 0 && ts_depth->frame_depth == 0) {
        // TODO: emit a non-inlining call uop and skip to the return
        goto unsupported;  // temporary: stop tracing here
    }
    
    ts_depth->frame_depth++;
    ...
```

This is a conservative first step — it stops the trace at polymorphic call sites rather than producing traces that immediately deoptimize. A full solution would emit a call uop and continue tracing past the return.

## Testing strategy

1. Build CPython with JIT: `./configure --enable-experimental-jit && make`
2. Run tinygrad softmax benchmark with `PYTHON_JIT=1`
3. Compare against unmodified CPython
4. Use `PYTHON_LLTRACE=2` to verify traces are longer (completing the loop back-edge)

## Risk

Low — the change only affects traces started from side exits (chain_depth > 0). First-time traces at monomorphic call sites are unaffected. The worst case: traces that would have been created from side exits are now not created, falling back to the interpreter. Since the current behavior is "create a trace that immediately deoptimizes," the fallback to interpreter is equivalent.

## Status

Draft — waiting for LLVM 21 installation to build and test.
