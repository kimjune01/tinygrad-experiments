# The CPython Floor — Partially Retired

> **Status (2026-05-08):** This document claimed the floor was CPython's 15ns `LOAD_ATTR_SLOT`. That claim was accurate for per-access cost but wrong as a stopping point. The 3.16 JIT reduces total overhead by 5.8% by eliminating bytecode dispatch for the traced loop body. More importantly, H₁₂ (skip_op) proved that the real remaining frontier is redundant work inside tinygrad's own compiled matchers — not CPython's object model. See `HYPOTHESIS_GRAPH.md` H₁₃–H₁₆ for the open edges.

## What CPython already does

**`LOAD_ATTR_SLOT` specialization (Python 3.10+, [bpo-42927](https://github.com/python/cpython/issues/87093))**

After warmup, `uop.op` compiles to `LOAD_ATTR_SLOT 0 (op)` — a direct memory offset read. Verified on Python 3.14.4:

```
# dis.dis(test_slot_access, adaptive=True) after 1000 calls:
LOAD_FAST_BORROW         0 (uop)
LOAD_ATTR_SLOT           0 (op)
RETURN_VALUE
```

Nested access `uop.src[0].op` also fully specializes:

```
LOAD_FAST_BORROW         0 (uop)
LOAD_ATTR_SLOT           0 (src)
LOAD_SMALL_INT           0
BINARY_OP_SUBSCR_TUPLE_INT 26 ([])
LOAD_ATTR_SLOT           2 (op)
RETURN_VALUE
```

Cost: ~15ns per slot access. This is already the fastest path CPython offers.

**Breakdown of the 15ns:**
- ~5ns: eval loop dispatch (fetch opcode, decode, jump to handler)
- ~3ns: inline cache check (verify type version tag hasn't changed)
- ~2ns: actual memory read (base pointer + precomputed offset)
- ~5ns: push result to value stack + reference count increment

**[PEP 659](https://peps.python.org/pep-0659/) adaptive interpreter (Python 3.11+)**

Bytecode instructions self-specialize based on runtime types. `LOAD_ATTR` becomes `LOAD_ATTR_SLOT` after a few executions. The specialization is automatic — no user intervention needed.

**Copy-and-patch JIT (Python 3.13+)**

| Python | Build | JIT delta | Mechanism |
|---|---|---|---|
| 3.14.4 | Release | ±0% | Projecting tracer; polymorphic calls prevent trace completion |
| 3.16.0a0 | Release | **-5.8%** | Recording tracer; loop body runs native despite polymorphic deopt |
| 3.16.0a0 | Debug | +54% | `Py_DEBUG` assertions inflate tier transition cost — not production-relevant |

The 3.16 recording tracer (gh-139109) partially addresses the polymorphic dispatch problem. The trace compiles the loop body (dict.get, attribute access, early_reject) to native code. At the polymorphic `match(uop, ctx)` call, `_GUARD_IP__PUSH_FRAME` fails and exits via `_COLD_DYNAMIC_EXIT` to the interpreter. Monomorphic ceiling is -19%; the ~13pp gap requires trace continuation after dynamic exits (planned upstream). See `CPYTHON_JIT_HYPOTHESIS_GRAPH.md`.

## Why the "CPython floor" framing was wrong

The 15ns per `LOAD_ATTR_SLOT` is real but was used to justify stopping the investigation. What it actually shows:

1. **Per-access cost is irreducible from Python** — true. You can't beat `LOAD_ATTR_SLOT` from a C extension or from Python code.

2. **Therefore further optimization requires CPython changes** — false. H₁₂ proved this wrong. The redundant op check in compiled matchers cost 22-29ns per attempt and was pure tinygrad waste, removable with a 2-line diff (-3.2% to -4.0%). The floor was inside tinygrad's code generation, not CPython's object model.

3. **The JIT doesn't help** — was true on 3.14, false on 3.16.

The correct framing: CPython's per-instruction cost is the floor for any SINGLE operation. But tinygrad's compiled matchers perform SEQUENCES of operations, many of which are structurally redundant. The optimization surface is eliminating those redundant operations — which compounds through fractal nesting (H₉).

## What we tried and measured

| Approach | Per-call | End-to-end | Why |
|---|---|---|---|
| Python `uop.op` (LOAD_ATTR_SLOT) | 15ns | baseline | Already specialized |
| C extension `PyObject_GetAttr` | 15ns | ±0% | Same object model underneath |
| ctypes `c_uint64.from_address` | 133ns | N/A | ctypes overhead > slot access |
| C-native CUOp struct | 1ns (C-internal) | +2.5% | Flatten cost + call overhead |
| PYTHON_JIT=1 (3.14) | 15ns | ±0% | Projecting tracer can't handle polymorphism |
| PYTHON_JIT=1 (3.16) | — | **-5.8%** | Recording tracer traces loop body |
| skip_op (H₁₂) | -22ns/attempt | **-3.2 to -4.0%** | Remove redundant op check in compiled matchers |

## The actual floor (revised)

The 15ns per `LOAD_ATTR_SLOT` is not reducible from Python or from a C extension. But the bottleneck is not individual accesses — it's the number of accesses per pattern match attempt. Each failed match attempt costs ~932ns, composed of ~30-60 individual operations. The optimization surface is:

1. **Eliminate redundant operations** (H₁₂ done, H₁₃–H₁₆ pending) — reduces the ~932ns per failed attempt
2. **Skip more attempts via richer early_reject** (H₁₆) — avoids the 932ns cost entirely
3. **JIT the loop body** (3.16, automatic) — reduces the per-operation cost from ~15ns to ~5ns for non-call operations

## Open frontier (from HYPOTHESIS_GRAPH.md)

| Hypothesis | Target | Predicted impact |
|---|---|---|
| H₁₃: Redundant dtype check | `upat.py` `_get_clause` | ~25-30ns/attempt, similar to H₁₂ |
| H₁₄: Redundant len(src) check | `upat.py` `_get_clause` | ~36ns/attempt for binary ops |
| H₁₅: Check ordering (Huffman inside matcher) | `upat.py` `_get_clause` | reorder checks by rejection rate |
| H₁₆: Richer early_reject (dtype + src count) | `ops.py` `PatternMatcher.__init__` | skip more 932ns matcher calls |

## References

- [bpo-42927 / GitHub #87093: Inline cache for slots](https://github.com/python/cpython/issues/87093) — the optimization that made `LOAD_ATTR_SLOT` fast
- [PEP 659: Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/) — the framework for bytecode specialization
- [GitHub #93911: More LOAD_ATTR specialisations](https://github.com/python/cpython/issues/93911) — extending specialization to more attribute patterns
- [CPython interpreter internals](https://github.com/python/cpython/blob/main/InternalDocs/interpreter.md) — how the eval loop works
- [python/cpython#149564](https://github.com/python/cpython/issues/149564) — our issue; JIT × polymorphic dispatch
- [python/cpython#139109](https://github.com/python/cpython/issues/139109) — recording tracer tracking issue
- `CPYTHON_JIT_HYPOTHESIS_GRAPH.md` — full JIT investigation with 9 nodes
- `HYPOTHESIS_GRAPH.md` — tinygrad-side investigation with 17 nodes, H₁₃–H₁₆ open
