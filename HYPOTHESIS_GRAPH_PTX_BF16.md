# PTX renderer KeyError on `dtypes.bfloat16` — Hypothesis Graph

Investigation target: [tinygrad#16105](https://github.com/tinygrad/tinygrad/issues/16105)
Repo state: `D:\depot\tinygrad` @ `4d1a9dca4`
Repro hardware (issue author): RTX 4080, CUDA, driver 596.36
Investigator's machine: Windows, OpenCL default (no CUDA installed). Codex unavailable. **H₀ reproduced by induction here** (PTX renderer accepts a forced `Target("CUDA","PTX","sm_89")` even without CUDA libs — the crash is pure-Python, no hardware needed). H₃ (silent renderer fallback) still requires CUDA hardware to induce.

## Frontier

| ID  | Hypothesis                                                              | Status      | Mode      | Conf |
|-----|-------------------------------------------------------------------------|-------------|-----------|------|
| H₀  | PTX `types` dict is missing `dtypes.bfloat16`; `ssa()` KeyErrors on it  | confirmed (induction) | induction | 99%  |
| H₀ᵇ | KeyError site is broader than DEFINE_REG: any **ALU / CONST / CAST / RANGE / DEFINE_VAR / DEFINE_LOCAL / PARAM** uop with bf16 dtype hits line 234 → ssa() → line 183 KeyError | confirmed | deduction | 99% |
| H₁  | Omission is deliberate — `is_dtype_supported` excludes bf16 on PTX      | confirmed   | deduction | 95%  |
| H₂  | Test guard relies on `is_dtype_supported`, which should skip the case   | killed by user observation | deduction | 90% |
| H₃  | `CUDARenderer` init fails on user's box → `PTXRenderer` silent fallback → check/renderer mismatch | confirmed (induction, RTX 4080) | induction | 99% |
| H₄  | Fix A: add `dtypes.bfloat16: "bf16"` to PTX `types` + `mem_types`(b16)  | open        | deduction | 70%  |
| H₅  | Fix B: extend `ptx_matcher` to upcast bf16 ALU → f32 (mirror half rule) | open        | deduction | 75%  |
| H₆  | Fix C: tighten `is_dtype_supported` to query the *selected* renderer    | open        | deduction | 80%  |

## H₀ — PTX `types` dict has no bfloat16 entry

**Reproduced (induction, this Windows box, no CUDA needed):**
```
$ python3 -c "from tinygrad.renderer.ptx import PTXRenderer; from tinygrad.helpers import Target;
              from tinygrad.dtype import dtypes, AddrSpace; from tinygrad.uop.ops import UOp, Ops;
              r = PTXRenderer(Target('CUDA','PTX','sm_89'));
              ptr = dtypes.bfloat16.ptr(size=1, addrspace=AddrSpace.REG);
              r.render([UOp(Ops.DEFINE_REG, ptr, (), 0), UOp(Ops.SINK, dtypes.void, ())])"
KeyError reproduced at PTX: KeyError(dtypes.bfloat16)
```
Verified that adding `dtypes.bfloat16: "bf16"` to `types`/`mem_types`/`cast_types` makes that minimal kernel emit valid PTX (`.reg .bf16 %reg_bf16_<1>;`).

**Evidence (deduction, source read).**
`tinygrad/renderer/ptx.py:157-159`:
```python
types: dict[DType, str] = { dtypes.int8: "s16", dtypes.int16: "s16", dtypes.int32: "s32", dtypes.int64: "s64",
                            dtypes.uint8: "u16", dtypes.uint16: "u16", dtypes.uint32: "u32", dtypes.uint64: "u64",
                            dtypes.float16: "f16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "pred" }
```
Plus `mem_types`/`cast_types` (lines 161-162) inherit from `types` — also no bf16.
`render_val` (lines 11-16) has no bf16 branch.
`ptx_matcher` (lines 40-50) has a `dtypes.half` upcast-to-f32 rule (lines 46-47) but **no equivalent rule for bf16**.

`ssa()` at line 183 does `self.types[unwrap(u).dtype.base]` — this is the exact KeyError site reported. **Confirmed.**

**Trajectory shape:** divergent — exactly one cause, exactly one site.

## H₀ᵇ — KeyError site is broader than DEFINE_REG

**Evidence (deduction).** `ssa()` (line 183) is called from line 234:
```python
prefix, dtype = {Ops.CAST: ("cast", None), Ops.BITCAST: ("cast", None), Ops.END: ("pred", "pred"),
  Ops.RANGE: ("ridx", None), Ops.DEFINE_VAR: ("dat", None), Ops.CONST: ("const", None),
  Ops.DEFINE_LOCAL: ("local", self.types[dtypes.ulong]),
  Ops.PARAM: ("dat", self.types[dtypes.ulong]), **{op: ("alu", None) for op in GroupOp.ALU}}.get(u.op, (None, None))
if prefix: r[u] = ssa(prefix, u, dtype)
```
With `dtype=None`, ssa() falls back to `self.types[unwrap(u).dtype.base]` — the user's exact stack trace. So **any** uop in this dispatch table that carries bf16 will crash, not only DEFINE_REG. This matches "Any LOAD op with bfloat16 dtype crashes" in the issue text — and explains why the crash survives even when DEFINE_REG is rejected upstream by the "needs to be memory" assert.

## H₁ — Omission is deliberate

**Evidence (deduction + git blame).**
`tinygrad/device.py:330-331`:
```python
case "CUDA": return (not CI or BENCHMARKS) and target.renderer != "PTX"
case "NV":   return (not CI or BENCHMARKS) and target.renderer not in ("PTX", "NAK")
```
i.e., `is_dtype_supported(dtypes.bfloat16)` returns **False** when the active target renderer is PTX.

`git blame` on the `types` dict: lines have been bf16-free since `347a3acb37` (May 2024) and `9c77e9f9b7` (Dec 2024) — both by George Hotz. The `is_dtype_supported` PTX-exclusion clause has the same vintage. The two facts are consistent: PTX renderer was never extended to bf16, and the support check was set to mirror that.

**Implication.** PTX-without-bf16 is a design choice, not a forgotten table entry. Any fix that adds bf16 support is feature work, not a bug fix.

## H₂ — Test guard *should* skip bf16 cases on PTX

**Evidence (deduction).**
`test/backend/test_linearizer.py:204-212` (test_sum_acc_dtype) and `:229-236` (test_arg_acc_dtype) wrap the bf16 case in:
```python
if is_dtype_supported(tensor_dtype) and is_dtype_supported(acc_dtype) and is_dtype_supported(expected_dtype):
```
With `target.renderer == "PTX"`, `is_dtype_supported(dtypes.bfloat16)` returns False, the bf16 case is skipped, and `ssa()` is never called with bf16. So **if the user's `target.renderer` is "PTX", the bug shouldn't fire.**

**Killed by observation** — the user reports the crash regardless. So `target.renderer` on their box must NOT be "PTX" even though the renderer actually used IS `PTXRenderer`. This is the kill condition that generates H₃.

## H₃ — Silent renderer fallback creates check/renderer mismatch

**Hypothesis.** `Compiled._select_renderer` (`device.py:292-297`) calls `select_first_inited` (`helpers.py:139-148`), which tries renderers in order — for CUDADevice that's `[CUDARenderer, PTXRenderer, NVCCRenderer]` (`runtime/ops_cuda.py:120`). If `CUDARenderer.__init__` raises (e.g., NVRTC missing/mismatched on Windows, no `libnvrtc.so`/`nvrtc64_*.dll` on PATH), the loop **silently falls through** to `PTXRenderer`.

Meanwhile `is_dtype_supported(bfloat16)` reads `DEV.target("CUDA").renderer`. With no `DEV=CUDA:PTX` set, that string is `""` — the PTX-exclusion clause does not fire, bf16 reads as supported, the test runs the bf16 case, the actual renderer is PTX, KeyError.

**Provenance.** `select_first_inited` exception swallowing dates from at least the renderer-list refactor. The PTX-aware support check was added without coupling it to the *resolved* renderer — the gap is structural, not a recent regression.

**Predicted classification if reproduced.** Divergent — setting `DEV=CUDA:CUDA` (force CUDA renderer) on a box where CUDA renderer can init would make the bug vanish; setting `DEV=CUDA:PTX` would skip the test cases and also make it vanish; only the unset state with broken NVRTC is divergent toward the bug.

**Why I can't induce here.** This Windows machine has no CUDA at all (`Device.DEFAULT == "CL"`), so I can't reach the fallback path. The user could classify this in one command — see "Decisive perturbation for the user" below.

## H₄ / H₅ / H₆ — Fix options

| Fix | What | Reach | Risk |
|-----|------|-------|------|
| **A**: add bf16 to `types`/`mem_types` | `dtypes.bfloat16: "bf16"` in `types`; `b16` in `mem_types`. Add `render_val` branch. | Smallest diff. Makes load/store + naming work. | ALU ops (add/mul/cmp/etc.) on `.bf16` are unsupported pre-sm_80 and only partially supported even on sm_80+. Will compile-fail at PTX assembler instead of crashing in renderer — still bad. |
| **B**: bf16 ALU → upcast to f32 | Mirror lines 46-47: `(UPat(doesnt_support_half_or_bf16, dtype=dtypes.bfloat16, name="x"), lambda x: ...cast(f32)...cast(bf16))`. Requires A's load/store entries too. | Correct — matches how `half` is handled. Lets bf16 actually work end-to-end on PTX. | Larger surface, needs tests. Doubles "what does PTX support" matrix. |
| **C**: tighten `is_dtype_supported` | Query `Device[Device.DEFAULT].renderer.suffix` (e.g., `"PTX"`) instead of `target.renderer`, OR have `_select_renderer` write the resolved name back into the target. | Smallest semantic change — preserves the current "PTX has no bf16" contract; just makes the check honest. Tests stop crashing. | Doesn't add bf16 support. User wanting bf16 on RTX 4080 still can't get it via PTX. Other uses of `target.renderer` may need same treatment. |

**Recommendation.** Ship **C** as the bug fix (it matches the existing contract), then file a follow-up issue for **B** as feature work. A is dominated by B and shouldn't be shipped alone.

**Sketch of fix C** (~3 LOC change in `tinygrad/device.py`, replace the two `target.renderer == "PTX"` clauses):
```python
def _selected_renderer_name(target):
  try: return type(Device[target.device].renderer).__name__.upper().removesuffix("RENDERER")
  except Exception: return target.renderer
# then:
case "CUDA": return (not CI or BENCHMARKS) and _selected_renderer_name(target) != "PTX"
case "NV":   return (not CI or BENCHMARKS) and _selected_renderer_name(target) not in ("PTX", "NAK")
```
Caveat: instantiating `Device[...]` from inside `is_dtype_supported` introduces a circular-init risk if called during device construction. Safer alternative: make `_select_renderer` write the resolved name back into the cached `Target` before returning.

## Decisive perturbation for the user (one command)

```bash
python -c "from tinygrad import Device; r = Device[Device.DEFAULT].renderer; \
  print('selected renderer class:', type(r).__name__); \
  print('target.renderer string :', r.target.renderer); \
  from tinygrad.dtype import dtypes; from tinygrad.device import is_dtype_supported; \
  print('is_dtype_supported(bf16):', is_dtype_supported(dtypes.bfloat16))"
```

- If output is `PTXRenderer / "" / True` → **H₃ confirmed**, fix C is the right framing.
- If `PTXRenderer / "PTX" / False` → H₃ killed; tests must already skip bf16; the crash is from a different code path — re-enter Phase 2.
- If `CUDARenderer / ... / True` → tests should run on CUDA renderer (no PTX). Different bug.

## Frontier edges (what would close this)

1. User runs the perturbation above → classifies H₃.
2. If H₃ confirmed, write fix C (~5 LOC), run `pytest test/backend/test_linearizer.py -k "test_arg_acc_dtype or test_sum_acc_dtype"` on user's box. Bug hunt with codex.
3. If H₃ killed, profile which uop carries bf16 into PTX despite the guard, and follow that edge.

## Pruning log

- H₂ killed by user observation — the test guard is necessary but not sufficient.
- H₄ (fix A alone) shouldn't ship — moves crash from Python to PTX assembler, no real improvement.

## Phase 5–7 (executed on local branch `fix-c-ptx-bf16-support-check`)

**Diff** (4 lines, `tinygrad/device.py`):
```python
def is_dtype_supported(dtype:DType, target:Target|None=None) -> bool:
-  target = target or DEV.target(Device.DEFAULT)
+  if target is None:
+    dev = Device[Device.DEFAULT]
+    base = DEV.target(dev.device.split(':')[0], **({"arch":dev.arch} if dev.arch else {}))
+    target = replace(base, renderer=base.renderer or dev._renderer_name(type(dev.renderer)))
```

**Regression check (Phase 5.5)** on this Windows box (Device.DEFAULT == CL):
- `pytest test/backend/test_linearizer.py -k "test_arg_acc_dtype or test_sum_acc_dtype"` → 2 passed.
- `pytest test/backend/test_renderer_failures.py test/null/test_dtype.py test/backend/test_linearizer.py` → 46 passed, 8 skipped, 1 xfailed.
- `test/backend/test_dtype.py` skipped (preexisting torch dependency missing).
- `test/unit/test_dtype_spec.py` skipped (preexisting hypothesis dependency missing).

**Simulated PTX-fallback scenario** (mock CUDADevice with PTXRenderer) — `is_dtype_supported(dtypes.bfloat16) → False`. Pre-fix would have returned True. ✓

**Phase 7 bug hunt** — Gemini 3.1 Pro Preview (codex unavailable on this box). Findings:
1. `AttributeError` defense for non-`Compiled` devices — **not a real risk in current code** (every device class inherits `Compiled`, which sets `self.arch` in its constructor). Could add `getattr` for defense-in-depth, but unnecessary today.
2. Circular initialization — `is_dtype_supported` is **not called** from any `_select_renderer`, `__init__`, or `Renderer` constructor (verified by grep: 4 call sites, all runtime). No risk.
3. CPU/LVP behavior change — `case "CPU": ... and target.renderer != "LVP"` previously compared the empty string to "LVP" and returned True (bf16 spuriously supported on CPU+LVP). Post-fix it correctly resolves to "LVP" and returns False. **Same nature as the bf16/PTX fix** — the exclusion clause was always intended to fire; the empty-string default silently disabled it. Behavior change is in the direction of the existing contract.

Verified renderer-name resolution across devices:
- CL → `OPENCL`, PYTHON → `PYTHON`, NULL → `NULL`, CPU → `CLANGJIT` (this box; would be `LVP` where LVP is default).

**Convergence.** One Gemini round, zero new findings requiring code changes. Codex unavailable — flagging convergence at lower confidence than the skill default (~85% vs 95%).

## Phase 8 (ship — pending human gate)

PR ready on branch `fix-c-ptx-bf16-support-check` in `D:\depot\tinygrad`. Recommended draft:

> **Title:** fix: is_dtype_supported uses resolved renderer when target is None
>
> **Body:**
> Fixes #16105.
>
> When `DEV` is unset, `target.renderer` resolves to the empty string. `Compiled._select_renderer` then calls `select_first_inited`, which silently swallows constructor failures and falls through the candidate list — e.g., `CUDARenderer` → `PTXRenderer` when NVRTC is missing. The PTX-exclusion clauses (`target.renderer != "PTX"`) misread the situation, the test guard runs the bf16 case, and `PTXRenderer.ssa()` KeyErrors on `dtypes.bfloat16`.
>
> Fix: when `target is None`, resolve via the actually-selected renderer instead of the `DEV` ContextVar.
>
> Side benefit: `case "CPU": ... and target.renderer != "LVP"` — and the symmetric `"PYTHON"`/`"NAK"`/`"LLVM"` exclusions — now actually fire when no `DEV` is set. They were silently disabled by the empty-string default.

Not pushed. Awaiting go/no-go.

## H₃ — induced on RTX 4080 (post-hoc)

This Windows box has the same hardware as the issue author: RTX 4080, driver 596.36, **no CUDA Toolkit installed** (driver-only). Driver-only is the reason: `nvrtc` is shipped with the toolkit, not the driver, so `CUDARenderer.__init__` (which depends on NVRTC) raises, `select_first_inited` swallows the exception, and `PTXRenderer` is selected.

```bash
$ DEV=CUDA CUDA_PATH=C:\Windows\System32\nvcuda.dll python3 -c "
from tinygrad import Device; from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported; from tinygrad.helpers import DEV
dev = Device[Device.DEFAULT]
print(type(dev.renderer).__name__, repr(DEV.target('CUDA').renderer), is_dtype_supported(dtypes.bfloat16))"

PTXRenderer '' False    # with fix C
PTXRenderer '' True     # without (master)
```

End-to-end:
- `master`: `pytest test/backend/test_linearizer.py -k "test_arg_acc_dtype or test_sum_acc_dtype"` → **1 failed**, `KeyError: dtypes.bfloat16` at `ptx.py:183`. Exact issue match.
- `fix-c-ptx-bf16-support-check`: same command → **2 passed**.

H₃ confidence: 70% (abduction) → 99% (induction).
