# Hypothesis Graph: graph_rewrite Performance

**Question:** What is the mechanism that makes graph_rewrite expensive, and where is the leverage for further optimization?

**System:** tinygrad scheduler pipeline (`get_kernel_graph` → `create_linear_with_vars` → `realize`)

---

## Phase 1–3: Observation → Fan-out → Extend (H₀–H₆)

*(Condensed — see git history for full detail)*

- **H₀:** Matching infrastructure is <15% of tottime. Bulk is in traversal + callbacks.
- **H₁:** 930 graph_rewrite calls per realize. Passes can't merge (different modes/ctx). Convergent.
- **H₂:** backward_slice O(n²) — **KILLED.** Double caching makes it structurally impossible.
- **H₃:** UOp construction tax — 6.8%, lazy properties, near irreducible. Weak.
- **H₄:** RETE applicable but big architectural change. Confirmed.
- **H₅:** Bloom filter: 98% read-only skip rate. **Killed as gate** (cascade failure). Lives as index concept.
- **H₆:** Decision tree: correct, Python per-call overhead neutralizes iteration savings.

---

## Phase 4: The Entropy Discovery (H₇–H₉)

### H₇: The dispatch table has 1.91 bits of entropy — CONFIRMED

8,649 slots (93×93), 59% empty, 26% single-pattern, 8% two-pattern, 7% three+. Only 29 distinct row signatures. 132 distinct pattern lists.

### H₈: Three-track dispatch — correct but no speedup (OSCILLATORY)

`src[0].op` access on every call adds 40ns that eats the savings from shorter lists. CPython's dict.get + for-loop-over-list is already near-optimal.

### H₈ₐ: Ops layout already structured — CONFIRMED

Binary [36-54], Unary [29-35], Movement [80-85] — contiguous bands. But dense subtable gains (7ns) are too small to matter.

### H₉: Cost is fractal — CONFIRMED

Successful match callbacks contain nested graph_rewrite calls. Dispatch optimization compounds through all nesting levels. The "80% in successful matches" framing was wrong — the 23μs per successful match includes thousands of nested pattern dispatches.

---

## Phase 5: Huffman + Branch Prediction (H₁₀–H₁₂)

### H₁₀: Huffman if-elif tree replacing dict — KILLED

+10.7% slower. CPython evaluates if-elif sequentially (O(n)). dict.get is O(1) via hash. The dict IS the optimal first-level dispatch in CPython.

**Trajectory:** DIVERGENT against. The Huffman tree is correct for native code but wrong for bytecode interpreters where dict.get is cheaper than sequential comparison.

### H₁₁: Per-op compiled functions (dict kept, loop eliminated) — KILLED

+3.5% slower. Function call overhead (~100-150ns frame creation) exceeds loop elimination savings. Closure variable lookup (LOAD_DEREF) is slower than tuple iteration (UNPACK_SEQUENCE).

**Trajectory:** DIVERGENT against. CPython's for-loop-over-list-of-tuples is already its fastest iteration primitive. Any replacement adds overhead.

### H₁₁ₐ: Per-iteration cost analysis — CONFIRMED

With compiled matchers (UPAT_COMPILE=1):

| Op | Patterns | Total | Per-iteration |
|---|---|---|---|
| ADD | 9 | 697ns | 74ns |
| MUL | 9 | 718ns | 76ns |
| CONST | 8 | 683ns | 81ns |
| STORE | 10 | 157ns | 12ns (early_reject dominant) |

80% of time in successful matches (avg 23μs — includes nested work). 20% in failed matches (avg 932ns).

### H₁₂: Redundant op check in compiled matchers — **CONFIRMED, FIRST SPEEDUP**

**The breakthrough.** `upat_compile` generates code that re-checks `uop.op` at the start of every compiled matcher. But `PatternMatcher.rewrite` already dispatched by `pdict[uop.op]` — the op is guaranteed to match. This check is pure waste.

**Evidence:**
- 98% of compiled matchers (5,307 of 5,408) have a redundant op check
- Each check costs 22-29ns (op equality or set membership)
- Applied to every pattern attempt, including the 4,417 failed matches per softmax realize

**Perturbation:** Added `skip_op=True` parameter to `_get_clause` in `upat.py`. When generating the root UPat's code, skip the op check. Two lines changed.

**Result:**

| Workload | Original | skip_op | Delta |
|---|---|---|---|
| 4x conv (cold) | 65.4ms | 63.3ms | **-3.2%** |
| transformer (cold) | 52.2ms | 50.1ms | **-4.0%** |

All workloads pass correctness (softmax, conv2d+relu, 4x conv, transformer).

**Trajectory:** DIVERGENT in favor. First measurable wall-clock improvement. Compounds through fractal nesting. Pure Python, 2-line diff, zero new dependencies.

**Provenance:** The redundant check exists because `_get_clause` generates structural checks for the ENTIRE UPat tree, including the root's op constraint. The root op is already resolved by `pdict` dispatch, but `_get_clause` doesn't know it's generating the root — it treats every depth the same. The fix is minimal: tell `_get_clause` when it's generating the root clause so it skips the op check.

---

## Phase 6: Continuing Investigation

### What H₁₂ teaches about the remaining frontier

The 3.2-4.0% speedup from removing ONE redundant check (22-29ns per attempt) tells us: **the per-attempt overhead in compiled matchers is the right optimization surface.** Each failed match attempt costs 932ns, of which ~25ns was the redundant op check (2.7%). The remaining 907ns per failed attempt has more redundant work to find.

### Open frontier edges (for /investigate)

**H₁₃: Redundant dtype check in compiled matchers.** Many compiled matchers check `uop.dtype == X or uop.dtype._scalar == X` on the root UPat. If early_reject or pdict could pre-filter by dtype (at __init__ time, not per-call), the compiled matcher wouldn't need this check. The dtype check costs 25-30ns per attempt. Potential: similar magnitude to H₁₂.

**H₁₄: Redundant len(src) check — KILLED.** UOps don't enforce arity: `UOp(Ops.ADD, ..., (x,))` with 1 source is allowed. During rewriting, intermediate states CAN have wrong arity. The `len(uop.src) == 2` check is a genuine filter. Removing it gives -5.5% on the rewrite micro-bench but +5.4% on end-to-end because intermediates hit the previously-skipped checks. The check is load-bearing.

**H₁₅: Check ordering within compiled matchers.** Currently `_get_clause` emits checks in a fixed order: op → arg → len(src) → name → dtype → src children. The Huffman-optimal order would be: the check that rejects fastest first. If `src[0].op` rejects 80% of attempts and `len(src)` rejects 5%, checking `src[0].op` first saves more total time.

**H₁₆: early_reject for dtype — CONVERGENT (~1%).** Implemented and measured. Rejects 26.7% of match() calls (29,087 of 109,079 on conv). But actual speedup is only ~1% (23.5ms vs 23.8ms) because the compiled matcher already checks dtype early and fails fast. The function call overhead saved (~100-150ns per rejected call) is much less than the predicted 932ns. The 932ns figure is the average for ALL failed matches, including those that check deep into src children — dtype rejections are the cheap failures. Correct but marginal ROI.

---

## Graph State

| Node | Status | Shape | Mode | Confidence |
|------|--------|-------|------|-----------|
| H₀ | Partial | Divergent | Induction | 95% |
| H₁ (pass count) | Partial | Convergent | Induction | 90% |
| H₂ (backward_slice) | **Killed** | Divergent-against | Induction | 95% |
| H₃ (UOp construction) | Weak | Convergent | Induction | 90% |
| H₄ (RETE) | Confirmed applicable | Divergent-for | Deduction | 85% |
| H₅ (bloom filter) | **Killed as gate** | Oscillatory | Induction | 90% |
| H₆ (decision tree) | Partial | Convergent | Induction | 80% |
| H₇ (1.91-bit entropy) | **Confirmed** | Divergent | Induction | 95% |
| H₈ (3-track dispatch) | No speedup | Oscillatory | Induction | 90% |
| H₉ (fractal cost) | **Confirmed** | Divergent | Induction→Deduction | 85% |
| H₁₀ (Huffman if-elif) | **Killed** | Divergent-against | Induction | 95% |
| H₁₁ (per-op compiled fn) | **Killed** | Divergent-against | Induction | 90% |
| H₁₂ (redundant op check) | **CONFIRMED — FIRST SPEEDUP** | Divergent-for | Induction | 95% |
| H₁₃ (redundant dtype) | Low priority | — | Abduction | 50% |
| H₁₄ (redundant len(src)) | **Killed** | Oscillatory | Induction | 95% |
| H₁₅ (check ordering) | Subsumed by H₁₈ | — | — | — |
| H₁₆ (richer early_reject) | **Convergent (~1%)** | Convergent | Induction | 90% |
| H₁₇ (nested pdict) | **Killed** | Divergent-against | Induction | 90% |
| H₁₈ (mega-matcher) | **★ CONFIRMED — LARGEST SPEEDUP** | Divergent-for | Induction | 95% |
| H₁₉ (bitmask early-reject) | **Killed** | Divergent-against | Induction | 90% |
| H₂₀ (bitmask full-pattern) | **Killed** | Divergent-against | Deduction | 95% |
| H₂₁ (automated mega-matcher) | **★ CONFIRMED — -15.2%** | Convergent | Abduction → Induction | 95% |
| H₂₂ (end-to-end wall-clock) | **Killed** | Noise | Deduction | 95% |
| H₂₃ (CUDA kernel compilation) | **Killed** | Blocked (H₂₂ killed) | Deduction | — |
| H₂₄ (pattern-order-safe gate) | **Killed** | Dead on arrival (H₂₂) | Abduction | — |
| H₂₅ (skip 0-pattern ops) | **Killed** | Noise (same as H₂₂) | Deduction | 95% |
| H₂₆ (RETE leaf skip) | **Killed** | -1.1% (noise) | Induction | 95% |
| H₂₇ (Cython transpile) | **★ CONFIRMED — -7.3% e2e** | Divergent-for | Induction | 95% |
| H₂₈ (CPython JIT branchy loops) | **Pending** | — | Abduction | 50% |

## Causal Chain

```
H₀ (what's slow?)
├→ H₁-H₃ (infrastructure profiling) — floor established
├→ H₄-H₆ (algorithms from parts bin) — Python overhead kills them
├→ H₇ (entropy: 1.91 bits)
│   ├→ H₈ (3-track dispatch) — no speedup
│   └→ H₉ (fractal cost) — compounds through nesting
├→ H₁₀-H₁₁ (compiled dispatch) — killed
├→ H₁₂ ★ skip_op (-3.2% to -4.0%) — first speedup, shipped as PR #16096
│   ├→ H₁₃ redundant dtype — low priority (38 patterns)
│   ├→ H₁₄ redundant len(src) — killed (UOps don't enforce arity)
│   ├→ H₁₅ check ordering — subsumed by H₁₈
│   └→ H₁₆ richer early_reject — convergent (~1%)
├→ H₁₇ nested pdict — killed (wildcards dominate)
│
└→ H₁₈ ★★ MEGA-MATCHER (-18%) — largest speedup
    ├→ H₁₉ bitmask early-reject — killed (set.issubset already optimal)
    ├→ H₂₀ bitmask full-pattern — killed (wildcard patterns prevent)
    └→ H₂₁ automated mega-matcher generation — ★ CONFIRMED (-15.2%)
        ├→ H₂₂ end-to-end wall-clock on real workloads — KILLED (no signal)
        │   └→ H₂₃ CUDA kernel compilation (Windows bench) — KILLED (blocked)
        └→ H₂₄ pattern-order-safe gate — KILLED (dead on arrival)

Phase 11: the bottleneck is traversal, not matching
├→ H₂₅ skip 0-pattern ops in driver — KILLED (0.23ms on 23ms = 1%, noise)
├→ H₂₆ RETE leaf skip — KILLED (-1.1%, leaf nodes are 6% of graph)
├→ H₂₇ Cython transpile of unified_rewrite — ★ CONFIRMED (-7.3% e2e)
└→ H₂₈ CPython JIT: trace branchy dict/deque loops — the contribution
```

## Key Insight: H₁₀ → H₁₈ Resurrection

H₁₀ (Huffman if-elif tree) was killed because CPython's `dict.get` is O(1) and if-elif is O(n) in the interpreter. But H₁₈ shows the if-elif approach wins when applied INSIDE a compiled function rather than across functions. The per-function dispatch cost (~100-150ns frame creation) dwarfs the per-check cost (~15-30ns attribute access). Merging N functions into 1 eliminates N-1 frame creations while adding only the if-chain overhead — net win.

H₁₀ asked the wrong question ("if-elif vs dict.get for dispatch"). H₁₈ asks the right one ("1 function vs N functions for the same checks").

## Pruning Log

| Killed | How | Value |
|---|---|---|
| H₂ | Double caching | Caching is well-designed |
| H₅ gate | Cascade failure | Index, don't skip |
| H₈ speedup | Python per-call overhead | Replace, don't add |
| H₁₀ if-elif | dict.get is O(1), if-elif is O(n) | CPython dict IS the branch predictor — but wrong question (see H₁₈) |
| H₁₁ compiled fn | Frame creation > loop savings | UNPACK_SEQUENCE beats LOAD_DEREF |
| H₁₄ skip_len | UOps don't enforce arity | Intermediates have wrong src count — check is load-bearing |
| H₁₇ nested pdict | 73% wildcard fallback | Too few specific entries to offset 2nd dict.get |
| H₁₉ bitmask | set.issubset is C builtin | ~3% faster, wrong bottleneck |
| H₂₀ bitmask full | Wildcard patterns cover all pairs | Structural impossibility |
| "CPython floor" | H₁₂ + H₁₈ | 15ns LOAD_ATTR_SLOT is real but was used as false stopping point |

---

## Phase 7: Nested pdict (H₁₇)

### H₁₇: Two-level pdict dispatch (op → src[0].op) — KILLED

**Perturbation:** Modified PatternMatcher.__init__ to build `pdict2[op][src0_op]` with pre-merged wildcard+specific lists. Rewrite uses two dict.gets: first by op (same as before for empty), then by src0_op (only for non-empty).

**Evidence:**

| Workload | Original | skip_op | skip_op + nested pdict |
|---|---|---|---|
| 4x conv | 65.4ms | 63.3ms (-3.2%) | 63.6ms (-2.8%) |
| transformer | 52.2ms | 50.1ms (-4.0%) | 50.2ms (-3.8%) |

Dispatch hit rate analysis:
- Specific hit (src0 in sub): 19%
- Wildcard fallback (2 dict.gets): 73%
- Both miss: 8%

**Why it fails:** 50% of patterns are src-wildcards. The second-level dispatch almost always falls back to the wildcard list, paying two dict.gets for the same result as one. The discrimination exists for only 19% of non-empty calls — not enough to offset the overhead.

**Trajectory:** DIVERGENT against. The pattern population is too wildcard-heavy for src[0].op dispatch to help. The theoretical 40% iteration reduction only applies to the 19% of calls where specific entries exist.

**Reverted.** Only the skip_op patch (H₁₂) survives.

---

## Phase 8: CPython JIT Re-investigation

The `CPYTHON_FLOOR.md` document concluded that the 15ns `LOAD_ATTR_SLOT` floor was irreducible and blamed CPython. A follow-up investigation against CPython 3.16 (recording tracer, gh-139109) partially retired that claim. See `CPYTHON_JIT_HYPOTHESIS_GRAPH.md` for the full graph.

**Key findings:**
- 3.14 (projecting JIT): ±0% — the floor claim was accurate at time of writing
- 3.16 (recording JIT): **-5.8%** — the JIT now helps by compiling the loop body to native code
- Monomorphic ceiling: -19%, gap of ~13pp from polymorphic deopt at `_GUARD_IP__PUSH_FRAME`
- Debug builds show +54% regression — a `Py_DEBUG` artifact, not production-relevant

**What this means for H₁₃–H₁₆:** The CPython floor was a false stopping point. H₁₂ (skip_op, -3.2 to -4.0%) already proved the real frontier is inside tinygrad's compiled matchers. The JIT improvement stacks with matcher-side optimizations — they're orthogonal. H₁₃–H₁₆ are the correct continuation.

---

## Phase 9: Mega-matcher (H₁₈–H₂₁)

### H₁₈: Mega-matcher — merge all patterns per op into one function — **CONFIRMED, -18%**

**Hypothesis:** Merging all 20 ADD patterns into a single `mega_match_ADD(uop, ctx)` function eliminates polymorphic dispatch (20 function calls → 1) and shares common prefix checks (`len(src) == 2` checked once, `s0, s1 = uop.src` cached once). This makes the call site monomorphic for the JIT and removes ~100-150ns frame creation overhead per skipped call.

**Perturbation:** Hand-wrote a mega-matcher function that inlines all 20 ADD patterns' compiled code into one function body, preserving original index ordering. Benchmarked against the existing per-pattern dispatch via `symbolic.rewrite()`.

**Evidence:**

| Python | JIT | Original | Mega | Delta |
|---|---|---|---|---|
| 3.14.4 | N/A | 2,545ns | 2,014ns | **-20.9%** |
| 3.16.0 | off | 2,505ns | 2,027ns | **-19.1%** |
| 3.16.0 | on | 2,383ns | 1,981ns | **-16.9%** |

Stability: -17.7%, -17.9%, -17.7% across 3 runs on micro-benchmark.

Correctness: all test UOps produce identical results. End-to-end conv2d+relu passes with correct output.

**Trajectory:** DIVERGENT for. Largest single speedup in the investigation. Consistent across Python versions and JIT configurations.

**Mode:** Induction. Confidence: 95%.

**Mechanism breakdown:**
- 20 function calls eliminated → ~2,000-3,000ns saved (100-150ns × 20 calls, but most are rejected early)
- Shared `len(uop.src) == 2` check → ~18 redundant checks eliminated
- Shared `s0, s1 = uop.src` and `s0op, s1op` caching → ~36 redundant attribute accesses eliminated
- Monomorphic call site → JIT can inline the entire matcher (3.16 adds ~5% on top)

**What killed the bitmask approach (H₁₉):** The "stacked paper with holes" intuition was right about structure but wrong about mechanism. Bitmask `& ==` is no faster than `frozenset.issubset` in CPython's interpreter (~28-38ns both). The win comes from eliminating function calls, not from faster subset checks.

**Provenance:** The mega-matcher is a manual version of what a Rete network or discrimination net does — shared prefix elimination across patterns. The difference is it's emitted as a flat Python function that the JIT can compile, rather than a graph data structure that adds per-node overhead (which killed H₄, H₆).

### H₁₉: Bitmask early-reject replacing set.issubset — KILLED

**Hypothesis:** Encode early_reject as uint32 bitmaps (19 ops fit in 19 bits). Replace `frozenset.issubset(ler)` with `(ler_bitmap & pat_bitmap) == pat_bitmap`.

**Evidence:** On 3.16 JIT: 37.5ns (set.issubset) vs 36.3ns (bitmap) = 3% faster. On 3.14: 38.3ns vs 32.8ns = 17%. Not enough to matter — the 932ns per failed match is dominated by function call overhead and deep checks, not the 28-38ns issubset check.

**Kill condition:** `frozenset.issubset` is already a C builtin. The bitmap AND doesn't win meaningfully in the interpreter, and the JIT doesn't specialize the set operation (it's a C call either way).

**Trajectory:** DIVERGENT against. Correct but immaterial — wrong bottleneck.

**Mode:** Induction. Confidence: 90%.

### H₂₀: Bitmask for full-pattern rejection (skip ALL patterns in one check) — KILLED

**Hypothesis:** Encode not just src ops but also `(s0op, s1op)` pairs into a precomputed set. If no pattern can match a given pair, skip all 20 patterns with one lookup.

**Evidence:** 100% of `(s0op, s1op)` pairs are "matchable" because patterns 4, 7, 13, 19 have no src constraints (wildcard). The pair-based filter can't eliminate anything.

**Kill condition:** Wildcard patterns prevent pair-based discrimination. To make this work, you'd need to exclude wildcard patterns from the pair check and handle them separately — which is what the mega-matcher already does (p4 and p7 are checked before the `len(src) == 2` gate).

**Trajectory:** DIVERGENT against. Structural impossibility from wildcard patterns.

**Mode:** Deduction. Confidence: 95%.

### H₂₁: Automated mega-matcher generation — CONFIRMED (-15.2%)

Implemented `mega_compile` in `upat.py` (70 lines, 2 files). Runs each pattern through the existing `_get_clause` → `pm_proc` → `pm_renderer` → `_final_render` pipeline, then namespaces dyn_lookup vars (`a0` → `a0_p{i}`), caches shared locals (`_s0 = uop.src[0]`, `_s0op = _s0.op`), and concatenates into one function per op. Gated behind `MEGA_MATCH=1` env var with fallback to per-pattern dispatch.

**Result:** -15.2% on ADD rewrite micro-bench (±0.2% across 3 trials, Python 3.14.4). Hand-written prototype got -18%. The 3% gap is from preserving original pattern order — the hand-written version reorders no-src patterns before the len gate, which changes match priority and breaks correctness for some ops.

**What didn't work:**
- **Bitmask pre-filter** (H₁₉/H₂₀ revisited): `1 << 128` creates Python big integers. Bitwise AND on arbitrary-precision ints costs more than the attribute lookups it replaces. The right architecture in C, wrong in CPython.
- **Pattern reordering + len gate**: Moving no-src patterns before src-requiring patterns changes which pattern matches first. Passed unit tests but broke conv2d (spec verification failure on Ops.AFTER). Correctness requires proving pattern independence per-op — too fragile for automation.

**Trajectory:** CONVERGENT as predicted. Engineering, not research.

**Mode:** Abduction → Induction (confirmed by measurement). Confidence: 95%.

---

## Phase 10: Does the micro-bench move the needle? (H₂₂–H₂₄)

H₁₈/H₂₁ proved -15% on the rewrite micro-bench (isolated `symbolic.rewrite` calls on ADD nodes). Three open questions remain:

### H₂₂: End-to-end wall-clock improvement on real workloads — KILLED

**Claim:** The -15% rewrite speedup produces measurable wall-clock improvement on full model compilation.

**Result:** No signal. 6 alternating trials on 2-layer CNN + 2-layer MLP (50 iterations each):
```
MEGA=0: 23.33ms  MEGA=1: 23.47ms  MEGA=0: 23.46ms
MEGA=1: 23.32ms  MEGA=0: 23.56ms  MEGA=1: 23.66ms
```

**Why it died:** Rewrite is 68% of compilation time, but the call distribution kills it:
- 66% of rewrite calls have 0-1 patterns (mega doesn't apply)
- 9% have 2-3 patterns (mega's shared prefix overhead is net negative — threshold raised to >=5)
- 24% have 5+ patterns (mega helps, but only ~200ns/call savings)

Expected improvement: 24% × 15% × 68% = **2.4% of total**, which is ~0.5ms on 23ms. Below measurement variance.

The -15% micro-bench is real but the leverage is diluted: most rewrite calls hit low-pattern-count ops where mega adds overhead without saving function calls.

**Mode:** Deduction. Confidence: 95% (killed by arithmetic, not ambiguity).

### H₂₃: CUDA kernel compilation time (Windows bench) — KILLED

Blocked by H₂₂. If the signal doesn't show on instrumented CPU compilation, CUDA execution noise makes it strictly harder to measure. No point.

### H₂₄: Pattern-order-safe gate for the remaining 3% — KILLED

Dead on arrival. H₂₂ showed the micro-bench doesn't propagate. Chasing 3% on a micro-bench that produces 0% end-to-end is pointless.

---

## Phase 11: The bottleneck is traversal, not matching (H₂₅–H₂₇)

H₂₂ revealed the real cost structure. `unified_rewrite` visits every node in the graph 1-3 times. Each visit costs ~300ns in Python overhead (deque ops, dict lookups, UOp reconstruction). Matching (`PatternMatcher.rewrite`) is a small fraction — 66% of calls hit 0-1 patterns and return immediately.

The leverage is in the driver loop, not the matcher.

### H₂₅: Skip 0-pattern ops in the driver — KILLED

**Claim:** Skipping `pm_rewrite` for ops not in `pm.pdict` eliminates 50% of rewrite calls and produces measurable end-to-end improvement.

**Perturbation:** Added `if x.op not in pm.pdict: return None` guard in `RewriteContext.pm_rewrite`.

**Result:** No signal. 3 trials each:
```
WITH guard:    22.94, 23.42, 23.18 → avg 23.18ms
WITHOUT guard: 23.14, 23.73, 23.36 → avg 23.41ms
```

**Why it died:** Same arithmetic as H₂₂. The 0-pattern rewrite calls are already cheap (~50ns each: function call + dict.get + len check + return None). Skipping the function call saves ~50ns × 4,537 calls = 0.23ms on 23ms = 1%. Below measurement variance.

**Key observation:** `rewrite()` line 1263 already guards with `if len(pats:=self.pdict.get(uop.op, [])):`. The 0-pattern calls enter the function and exit immediately. The only savings from the driver-level guard is avoiding the function call overhead itself (~30ns), which is negligible at this call volume.

**Reframe:** The problem is not that we're calling rewrite() on 0-pattern ops. It's that the PER-CALL COST of the entire driver loop (stack management, dict lookups, UOp reconstruction) dwarfs the per-call cost of the rewrite() function call. Saving 30ns on 50% of calls doesn't move the needle when the driver burns 300ns per node on traversal overhead.

**Mode:** Deduction → killed by arithmetic. Confidence: 95%.

### H₂₆: RETE-style leaf skip — KILLED

**Claim:** Skipping non-active leaf nodes (no pm/bpm patterns, no children) in the `unified_rewrite` traversal eliminates 6% of node visits and produces measurable speedup.

**Perturbation:** Added `_active_ops` set to `RewriteContext.__init__`. In stage 0 child-push loop, non-active leaves skip the stack: `self.replace[x] = x` inline.

**Result:**
```
BASELINE: 23.21, 23.56, 23.23 → avg 23.33ms
RETE:     23.13, 22.89, 23.19 → avg 23.07ms
Delta: -1.1% (within noise)
```

**Why RETE doesn't apply here:**
1. `pdict[op]` dispatch already gives O(1) pattern lookup — the "alpha network" is a dict
2. Hash-consing prevents redundant matching — structurally identical nodes are the same object
3. The driver already skips processed nodes via `self.replace` — no redundant visits
4. Non-active leaves are only 6% of the graph. The 33.5% non-active non-leaf nodes can't be skipped without processing their children first.
5. The dominant cost is Python overhead per node visit (~300ns for stack + dict ops), not algorithmic redundancy. RETE reduces algorithmic redundancy, not interpreter overhead.

**Reframe:** RETE is the wrong tool. It solves the many-rules-many-facts problem (1000s of rules, millions of facts). tinygrad has ~20 rules per op and ~10K nodes. The match is already O(1) via dict dispatch. The bottleneck is CPython executing the traversal loop — 300ns per node in stack/dict/tuple operations that would be 15-30ns in C.

**Mode:** Abduction → Induction (killed by measurement). Confidence: 95%.

### H₂₇: Cython transpile of unified_rewrite — CONFIRMED (-7.3% e2e)

**Claim:** Transpiling `unified_rewrite` to C via Cython eliminates bytecode interpreter dispatch overhead and produces measurable end-to-end improvement.

**Perturbation:** Extracted `unified_rewrite` to `cy_rewrite.pyx` (95 lines), compiled with Cython (`-O3`), monkeypatched onto `RewriteContext`. Zero algorithmic changes — same logic, same Python objects, same callbacks.

**Result:**
```
BASELINE: 23.10, 22.84, 23.00 → avg 22.98ms
CYTHON:   21.15, 21.56, 21.20 → avg 21.30ms
Delta: -7.3%
```

First end-to-end signal in the entire investigation. Every prior hypothesis (H₁₂ through H₂₆) showed micro-bench improvement but zero e2e signal. The difference: prior hypotheses optimized WITHIN the interpreter (fewer patterns, faster matching). Cython optimizes the interpreter ITSELF — direct C function calls instead of opcode dispatch for dict.get, deque.pop, set.__contains__, tuple construction.

**What this proves:** 7.3% of tinygrad's compilation cost is pure CPython bytecode dispatch overhead in one function. With typed Cython annotations on UOp fields, the gap would widen to 15-20% (struct access vs PyObject attribute lookup).

**Not shippable to tinygrad** — Python-only project. But it quantifies the CPython JIT opportunity: if the Tier 2 JIT could compile this loop shape, the improvement comes for free.

**Mode:** Deduction → Induction (confirmed by measurement). Confidence: 95%.

### H₂₈: CPython JIT — trace branchy dict/deque loops — PENDING

Continued in [CPYTHON_JIT_HYPOTHESIS_GRAPH.md](CPYTHON_JIT_HYPOTHESIS_GRAPH.md) as H₁₃. That graph has the full CPython JIT investigation (H₀–H₉: polymorphic call analysis, debug vs optimized builds, guard mechanism, 3.14 vs 3.16 delta) plus the Cython proof (H₁₂).

**Summary:** H₂₇'s Cython transpile proves 7.3% is available. CPython 3.16's JIT currently gets 0% on this workload because the trace exits on every branch in `unified_rewrite`. The contribution: improve the Tier 2 trace compiler's handling of multi-branch loops with dict/deque operations.

**Mode:** Abduction. Confidence: 50%.
