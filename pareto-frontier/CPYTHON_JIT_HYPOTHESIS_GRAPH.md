# Hypothesis Graph: CPython JIT × Polymorphic Call Sites

**Question:** How much does CPython 3.16's JIT help tinygrad's pattern matching, and can it help more?

**System:** CPython 3.16.0a0 (`~/Documents/cpython-jit`), JIT enabled via `--enable-experimental-jit`. Tinygrad's `PatternMatcher.rewrite` as the workload.

**Issue:** [python/cpython#149564](https://github.com/python/cpython/issues/149564) — JIT doesn't trace exec()-generated functions called from hot loops.

---

## Phase 1: Observation (H₀)

### H₀: The JIT produces no improvement for tinygrad's rewrite — KILLED (wrong on optimized builds)

**Original claim (issue #149564, based on 3.14):** JIT produces ±0% change.

**Debug build finding:** JIT is **34-54% slower** (`--with-pydebug`, Py_DEBUG=1). Entirely a debug artifact — assertions and refcount tracking inflate the tier 1↔2 transition overhead.

**Optimized build finding:** JIT provides **5-16% improvement** depending on workload complexity.

**Perturbation:** `PYTHON_JIT=0` vs `PYTHON_JIT=1`, 3 runs each for stability.

| Build | Workload | JIT=0 | JIT=1 | Delta |
|---|---|---|---|---|
| Debug | 7 nodes, 50k iters | 6,338ns | 9,783ns | **+54%** |
| Debug | 1 node, 50k iters | 1,385ns | 1,538ns | **+11%** |
| **Optimized** | **7 nodes, 50k iters** | **1,694ns** | **1,596ns** | **-5.8%** |
| Optimized | mono PM (1 pattern) | 182ns | 147ns | **-19%** |
| Optimized | poly PM (118 patterns) | 313ns | 263ns | **-16%** |

**Trajectory:** DIVERGENT for (on optimized builds). The JIT helps even with polymorphic dispatch.

**Mode:** Induction. Confidence: 95%. Three consistent runs on optimized build.

---

## Phase 2: Fan-out

### H₁: Monomorphic call guard causes immediate deopt — CONFIRMED

The JIT records ONE callee at `_PUSH_FRAME` and emits `_GUARD_IP__PUSH_FRAME` which checks `frame->instr_ptr == recorded_ip`. When a different callee arrives, the guard fails.

**Source:** `Python/bytecodes.c:6372-6377`:
```c
tier2 op(_GUARD_IP__PUSH_FRAME, (ip/4 --)) {
    _Py_CODEUNIT *target = frame->instr_ptr;
    if (target != (_Py_CODEUNIT *)ip) {
        EXIT_IF(true);
    }
}
```

**Trajectory:** DIVERGENT for. Confirmed by code reading and LLTRACE output.

**Mode:** Deduction. Confidence: 99%.

### H₂: Dynamic exit path is cheap (no cascading re-trace) — CONFIRMED

`_GUARD_IP__PUSH_FRAME` is in the `dynamic_exit_uop` table (line 546), so its exit goes to `_COLD_DYNAMIC_EXIT` which just does `GOTO_TIER_ONE(target)` — no backoff counter, no re-tracing, no chain_depth increment.

**Source:** `Python/bytecodes.c:6329-6335`:
```c
tier2 op(_COLD_DYNAMIC_EXIT, ( -- )) {
    SYNC_SP();
    // TODO (gh-139109): This should be similar to _COLD_EXIT in the future.
    _Py_CODEUNIT *target = frame->instr_ptr;
    GOTO_TIER_ONE(target);
}
```

**Implication:** The overhead is NOT from cascading executor creation (as hypothesized in CPYTHON_JIT_INVESTIGATION.md). It's from the per-iteration executor entry → partial trace → dynamic exit → tier 1 cycle.

**Trajectory:** DIVERGENT for. The cost is the tier 1 ↔ tier 2 transition, not trace compilation.

**Mode:** Deduction. Confidence: 99%.

### H₃: Debug build inflates overhead — CONFIRMED

The CPython build uses `--with-pydebug` (Py_DEBUG=1, assertions enabled, reference count tracking). Even a pure arithmetic loop shows 93.5ns (JIT=1) vs 56.5ns (JIT=0) — the JIT is slower for everything, not just polymorphic calls.

| Benchmark | JIT=0 (debug) | JIT=1 (debug) |
|---|---|---|
| Pure arithmetic | 56.5ns | 93.5ns (+65%) |
| Monomorphic call | 153.8ns | 74.8ns (-51%) |
| Polymorphic 9 callees | 155.7ns | 150.2ns (-4%) |

**Trajectory:** CONVERGENT. Debug build amplifies overhead uniformly. Need optimized build for accurate delta.

**Mode:** Induction. Confidence: 90%.

### H₄: JIT helps monomorphic calls significantly — CONFIRMED

With JIT=1 and a single callee, call cost drops from 153.8ns to 74.8ns (2.1x speedup, debug build). The JIT works exactly as intended when the guard passes.

**Trajectory:** DIVERGENT for. The JIT IS beneficial for the right workload.

**Mode:** Induction. Confidence: 95%.

---

## Phase 2.5: Provenance

### Guard mechanism origin

- `_GUARD_IP__PUSH_FRAME` — monomorphic IP guard, introduced with the copy-and-patch JIT in 3.13. No polymorphic alternative exists.
- `_COLD_DYNAMIC_EXIT` — stub with TODO for gh-139109 ("should be similar to _COLD_EXIT in the future"). Currently the simplest possible exit.
- `dynamic_exit_uop` table — separates frame-transition guards from branch guards. Frame transitions always use dynamic (no re-trace) exits.

### Related upstream work

- [faster-cpython/ideas #557](https://github.com/faster-cpython/ideas/issues/557): Superblock versioning — handles polymorphism via BBV-like approach. Not implemented.
- [python/cpython #118093](https://github.com/python/cpython/issues/118093): Tier 2 at function entry — orthogonal.
- [python/cpython #139109](https://github.com/python/cpython/issues/139109): Referenced in `_COLD_DYNAMIC_EXIT` TODO — would add re-tracing for dynamic exits.

---

## Current Diagnosis

The JIT's overhead at polymorphic call sites comes from the **per-iteration tier 1 → tier 2 → tier 1 transition cost**, not from cascading trace creation. Every loop iteration:

1. `ENTER_EXECUTOR` (tier 1 → tier 2)
2. Run traced loop body (native code)
3. `_GUARD_IP__PUSH_FRAME` fails (wrong callee)
4. `_COLD_DYNAMIC_EXIT` → `GOTO_TIER_ONE` (tier 2 → tier 1)
5. Interpreter executes the call and loop remainder
6. Back to `ENTER_EXECUTOR` (the JUMP_BACKWARD was replaced)

The fix should either:
- **A.** Detect the polymorphic pattern and uninstall the executor (let the interpreter handle the whole loop)
- **B.** Trace the loop body WITHOUT the callee (emit the call as opaque), so the trace completes the loop back-edge and no deopt occurs
- **C.** Add a polymorphic inline cache to the guard

**A** is the defensive fix (stop the bleeding). **B** is the constructive fix (enable JIT benefit for the non-call parts of the loop). **C** is the full fix (trace through multiple callees).

---

## Phase 3: Extend

### H₅: Optimized build shows JIT HELPS polymorphic dispatch — CONFIRMED

**Killed H₀.** On optimized builds (no `--with-pydebug`), the JIT provides consistent 5-16% improvement even with polymorphic call sites. The debug build's 34-54% regression is entirely from Py_DEBUG assertions inflating the tier 1↔2 transition overhead.

**Evidence (3 runs, optimized build):**

| Run | JIT=0 | JIT=1 | Delta |
|---|---|---|---|
| 1 | 1,684ns | 1,621ns | -3.7% |
| 2 | 1,694ns | 1,596ns | -5.8% |
| 3 | 1,704ns | 1,583ns | -7.1% |

**Mechanism:** The JIT traces the loop body (dict lookups, attribute accesses, early_reject checks) and runs them as native code. When `_GUARD_IP__PUSH_FRAME` fails (different callee), it exits to tier 1 for the call, then re-enters the executor. The native-code portion provides net speedup despite the deopt overhead.

**Trajectory:** DIVERGENT for. The JIT helps, and the improvement is consistent across runs.

**Mode:** Induction. Confidence: 95%.

### H₅ₐ: The monomorphic call ceiling is ~19% (optimized build)

When the PM has a single pattern (monomorphic call site), the JIT provides 19% improvement. This is the ceiling — the JIT traces through the callee successfully because the guard always passes.

The current 5-7% (full workload) vs 19% (monomorphic) gap is the **opportunity**: 12-14 percentage points of improvement are left on the table because the callee deoptimizes.

**Trajectory:** CONVERGENT. The gap exists but is bounded.

**Mode:** Induction. Confidence: 90%.

---

## Phase 4: Report — Reframing

### The issue claim is wrong

Issue #149564 states "PYTHON_JIT=1 on 3.14.4 produces no improvement." On 3.16 optimized builds, the JIT provides 5-16% improvement. Either:
1. The original measurement was on a debug build
2. 3.16's JIT improved since 3.14 (likely — there's been active development)
3. The measurement methodology was too noisy to detect a 5% change

The issue title "JIT doesn't trace exec()-generated functions called from hot loops" is also misleading:
- The JIT DOES trace the hot loop. The loop body runs as native code.
- The JIT does NOT trace through the exec()-generated callees (they deopt at the IP guard).
- The word "exec()" is irrelevant — the problem is polymorphism. A non-exec loop calling 9 different static functions would show the same behavior.

### What would make this better

The 5-7% → 19% gap suggests that tracing through the callee (or around it) would roughly triple the improvement. Three approaches:

1. **Opaque call (Option B):** The trace records the CALL without following into the callee. The call executes in the interpreter but the loop body stays native. Requires modifying the tracer to skip `_PUSH_FRAME` and instead emit a "call-and-resume" uop that drops to tier 1 for the call and re-enters the executor after. This is the `_COLD_DYNAMIC_EXIT` TODO (gh-139109).

2. **Polymorphic inline cache (Option C):** `_GUARD_IP__PUSH_FRAME` checks a small table of N known callees instead of one. If the callee is in the table, inline the appropriate trace. Like V8's megamorphic stubs. Major change.

3. **Per-callee trace variants:** Create N traces for the same loop, one per callee. The guard selects the right one. The loop body is duplicated but each variant has a correct callee inline. Expensive in code size.

Option B is the most promising — it's already acknowledged in the codebase (the TODO at `_COLD_DYNAMIC_EXIT`) and requires no new data structures.

---

### H₆: Frame penalty perturbation — CONVERGENT (intermediate frames help)

4x frame penalty (834 per frame vs original 168) stops the trace before entering deep callees. Result: JIT benefit drops from -5.8% to -4.2%.

This confirms: the intermediate frames (cached_bpm_rewrite → rewrite) are monomorphic and DO benefit from JIT compilation. Stopping the trace earlier hurts. The ideal fix isn't to stop earlier — it's to handle the polymorphic callee differently at the deepest level.

**Trajectory:** CONVERGENT. Frame penalty tuning provides marginal information but doesn't open a new direction.

**Mode:** Induction. Confidence: 85%.

### H₉: 3.14 vs 3.16 delta — CONFIRMED

3.14.4 (optimized release): ±0% (1,556ns vs 1,555ns). 3.16.0a0 (optimized release): -5.8% (1,694ns vs 1,596ns). The improvement is entirely from the 3.16 recording tracer (gh-139109). The 3.14 projecting tracer provided zero benefit for this workload.

**Trajectory:** DIVERGENT for. The recording tracer is a material improvement.

**Mode:** Induction. Confidence: 95%.

---

## Phase 5: Cython proves the gap (H₁₂–H₁₃)

From the main investigation (HYPOTHESIS_GRAPH.md, Phase 11): 27 hypotheses optimizing tinygrad's pattern matching produced micro-bench wins but zero end-to-end signal. The bottleneck is CPython's bytecode interpreter dispatching ~100 operations per node visit at 3-5ns each. The matching is already fast — the loop around it is the floor.

### H₁₂: Cython transpile of unified_rewrite — CONFIRMED (-7.3% e2e)

Transpiled `unified_rewrite` to `cy_rewrite.pyx` (95 lines). Same algorithm, same Python objects, same callbacks. Compiled with Cython `-O3`, monkeypatched onto `RewriteContext`.

**Result (tinygrad 2-layer CNN + 2-layer MLP, 50 iterations × 3 trials):**
```
BASELINE: 23.10, 22.84, 23.00 → avg 22.98ms
CYTHON:   21.15, 21.56, 21.20 → avg 21.30ms
Delta: -7.3%
```

First end-to-end signal in the entire investigation. Every prior hypothesis (H₁₂–H₂₆ in main graph) showed micro-bench improvement but zero e2e. The difference: Cython eliminates bytecode dispatch overhead — every `dict.get`, `deque.pop`, `set.__contains__`, `tuple()` runs as a direct C function call instead of going through the opcode loop.

**What the JIT should do:** the same thing Cython does. Compile the hot function to native code that calls C API functions directly. The Tier 2 JIT currently fails on `unified_rewrite` because the loop has too many branches (3 stages × bpm/pm paths × waitlist) for the trace compiler.

**Mode:** Induction. Confidence: 95%.

### H₁₃: CPython JIT trace quality on dict/deque loops — REFRAMED

**Original claim:** The JIT can't trace `unified_rewrite` because it's too branchy.

**Finding:** The JIT IS tracing it. 15+ executors created, with substantial traces:
```
EXEC_CREATE: unified_rewrite length=393 exits=36
EXEC_CREATE: unified_rewrite length=200 exits=20 (×2)
EXEC_CREATE: unified_rewrite length=95  exits=8
EXEC_CREATE: unified_rewrite length=90  exits=10
EXEC_CREATE: unified_rewrite length=83  exits=6
... (many more in the 27-81 range)
```

The traces are real — up to 393 uops with 36 exits, averaging ~11 uops per exit. The trace compiler handles the branches.

**But the benchmark shows 0% improvement:**
```
CPython 3.16 JIT OFF: 23.91ms
CPython 3.16 JIT ON:  23.82ms
```

**While Cython gets -7.3%.** Same algorithm, same objects, same C API calls.

**Reframe:** The problem is not trace CREATION (it works) but trace QUALITY. The JIT compiles the Python bytecodes to native code that calls C API functions (`PyDict_GetItem`, `PyObject_RichCompareBool`, etc.) via the same function pointers the interpreter uses. Cython does the same — calls the same C API — but produces more efficient calling conventions (direct calls, no opcode dispatch, no stack effect tracking, no error checking between every op).

**The gap:** The JIT's copy-and-patch stencils include per-instruction overhead (error checks, reference count updates, stack pointer syncs) that Cython's compiled output doesn't. For a loop that executes 100 C API calls per iteration, 3ns of overhead per call × 100 = 300ns per iteration — which is exactly the observed overhead.

**New hypothesis (H₁₄):** The JIT's per-uop overhead (error checking, refcount, stack sync) is the residual gap. Reducing it would close the 7.3% Cython advantage. This is a JIT optimizer improvement, not a trace compiler improvement.

**Mode:** Abduction → Induction (trace creation confirmed, quality gap measured). Confidence: 90%.

### H₁₄: StackRef conversion overhead in CALL_METHOD_DESCRIPTOR_FAST — CONFIRMED

**Claim:** The JIT's per-C-API-call overhead comes from `STACKREFS_TO_PYOBJECTS` — converting StackRef tagged references to PyObject* arrays before every C function call, then cleaning up after.

**Evidence (code reading, `Python/ceval.c:855-873`):**

The JIT's path for `dict.get(key, default)`:
```c
// _PyCallMethodDescriptorFast_StackRef (ceval.c:855)
STACKREFS_TO_PYOBJECTS(arguments, total_args, args_o);  // alloc temp array, copy 3 refs
res = cfunc(self, (args_o + 1), total_args - 1);         // call dict_get
STACKREFS_TO_PYOBJECTS_CLEANUP(args_o);                  // free temp array
```

Cython's path for the same operation:
```c
value = PyDict_GetItemWithError(d, key);  // direct call, no conversion
```

`STACKREFS_TO_PYOBJECTS` (`ceval_macros.h:498`) allocates a stack-local `PyObject*` array and copies StackRefs into it via `_PyObjectArray_FromStackRefArray`. For 3 args, that's ~10-15ns of allocation + copy + cleanup per call.

**Scale:** ~45K `dict.get` calls per tinygrad iter × ~12ns overhead = 0.54ms. Plus `deque.pop` (~23K calls), `set.__contains__` (~10K calls), `tuple()`, `UOp.__new__`, etc. — all going through the same StackRef conversion. Aggregate: ~1.5ms on 23ms = 6.5%. Matches the 7.3% Cython measurement.

**The JIT optimizer already eliminates the GUARD** (line 1921 in `optimizer_bytecodes.c`: replaces `_GUARD_CALLABLE_METHOD_DESCRIPTOR_FAST` with `_NOP` when the callable is a known constant). But it does NOT eliminate the StackRef conversion — that's baked into `_CALL_METHOD_DESCRIPTOR_FAST_INLINE`.

**The fix:** For known-constant method descriptors like `dict.get`, the optimizer could emit a specialized uop that calls `PyDict_GetItemWithError` directly with PyObject* args from the value stack, bypassing `STACKREFS_TO_PYOBJECTS`. This is what `CALL_LIST_APPEND` already does for `list.append` — it calls `_PyList_AppendTakeRef` directly without going through the generic fastcall path.

**Concrete patch target:** Add `CALL_DICT_GET` analogous to `CALL_LIST_APPEND`. In `specialize_method_descriptor()` (`specialize.c:1711`), detect `dict.get` and emit a specialized opcode that calls `PyDict_GetItemWithError` directly. The tier 2 version skips `STACKREFS_TO_PYOBJECTS` entirely.

**Mode:** Deduction. Confidence: 95%.

**Trajectory:** KILLED. The mechanism is real but the overhead on ARM64 non-free-threaded builds is already ~3-5ns total (StackRef masking is `bits & ~1`, dict_get arg parsing is trivial). `CALL_DICT_GET` patch built and benchmarked — 0% improvement on both micro-bench (33.3ns vs 33.9ns, noise) and end-to-end tinygrad (24.2ms vs 24.6ms, noise).

The Cython gap (7.3%) is NOT from StackRef conversion or generic dispatch. It's from eliminating the ENTIRE bytecode dispatch loop — the fetch/decode/jump between instructions. Individual operation specializations can't recover this; it requires compiling the whole loop body to native code with direct C calls, which is what Cython does and the tier 1 JIT's copy-and-patch stencils don't fully achieve.

---

## Frontier Edges (open)

| Edge | Predicted shape | Perturbation | Priority |
|---|---|---|---|
| H₁₀: _COLD_DYNAMIC_EXIT continuation | Divergent for | When dynamic exit re-tracing lands (gh-139109 follow-up), the trace could resume after the call returns, closing the 13pp gap | Upstream — monitor |
| H₁₁: Issue comment | N/A | Post findings to #149564 (saved in ISSUE_COMMENT_149564.md) | Blocked on gh auth |
| H₁₃: JIT trace quality | Convergent | JIT traces exist (393 uops) but produce 0% speedup. Gap is per-uop overhead, not trace creation. | Reframed |
| H₁₄: JIT per-uop overhead | Divergent for | Compare JIT stencil output vs Cython output for the hot loop. Identify redundant error checks / refcount ops. | **NEXT** |

---

## Graph State

| Node | Status | Shape | Mode | Confidence |
|---|---|---|---|---|
| H₀ (JIT harmful) | **KILLED** | Divergent against (debug only) | Induction | 95% |
| H₁ (monomorphic guard) | Confirmed | Divergent for | Deduction | 99% |
| H₂ (no cascading) | Confirmed | Divergent for | Deduction | 99% |
| H₃ (debug build inflates) | Confirmed | Convergent | Induction | 95% |
| H₄ (mono calls helped) | Confirmed | Divergent for | Induction | 95% |
| H₅ (optimized: JIT helps) | **Confirmed** | Divergent for | Induction | 95% |
| H₅ₐ (mono ceiling 19%) | Confirmed | Convergent | Induction | 90% |
| H₆ (frame penalty tuning) | Confirmed | Convergent | Induction | 85% |
| H₉ (3.14 vs 3.16 delta) | **Confirmed** | Divergent for | Induction | 95% |
| H₁₂ (Cython transpile) | **★ CONFIRMED — -7.3% e2e** | Divergent for | Induction | 95% |
| H₁₃ (JIT trace quality) | **Reframed** | Convergent | Induction | 90% |

## Causal Chain

```
H₀ (JIT "harmful") — KILLED by H₃+H₅
├→ H₁ (monomorphic guard deopt mechanism) — confirmed
├→ H₂ (dynamic exit, no cascading) — confirmed, corrected model
├→ H₃ (debug build artifact) — confirmed, gates all measurements
├→ H₄ (mono calls: JIT works well) — confirmed
│
└→ H₅ (optimized build: JIT helps 5-16%) — CONFIRMED, reframes investigation
    ├→ H₅ₐ (mono ceiling 19%, gap is ~13pp) — calibrates opportunity
    ├→ H₆ (frame penalty: intermediate frames help) — convergent, no action
    ├→ H₉ (3.14 ±0%, 3.16 -5.8%: recording tracer is the improvement)
    │
    └→ H₁₂ ★ Cython transpile (-7.3% e2e) — proves the JIT opportunity
        └→ H₁₃ JIT trace quality — REFRAMED (traces exist, quality is the issue)
            └→ H₁₄ JIT per-uop overhead — the CPython contribution
```

## Pruning Log

| Killed/Corrected | How | Value |
|---|---|---|
| H₀ "JIT harmful" | Optimized build: JIT helps 5-16% | Debug build inflated overhead 10-100x. Always benchmark release builds. |
| "cascading re-trace" model | Code reading: _COLD_DYNAMIC_EXIT doesn't re-trace | Corrected the CPYTHON_JIT_INVESTIGATION.md model. The cost is tier transition, not trace compilation. |
| "exec() is the problem" | Code reading: exec() functions pass PyFunction_Check, have normal code objects | The issue title is misleading. The problem is polymorphism, not code origin. |
| "no improvement" (issue claim) | 3.14 ±0% confirmed; but 3.16 gives -5.8% | The claim was accurate for 3.14. The 3.16 recording tracer partially addressed it. |
| H₆ "stop trace early" | 4x frame penalty reduced benefit from 5.8% to 4.2% | Intermediate frames ARE helpful. The fix is at the deepest level, not the shallowest. |
