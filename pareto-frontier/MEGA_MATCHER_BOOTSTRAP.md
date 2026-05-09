# Bootstrap: Automated mega-matcher for tinygrad's PatternMatcher

## What you're doing

Tinygrad's `PatternMatcher.rewrite()` calls N separate compiled match functions per op (20 for ADD, 19 for MUL, etc.). Each is a distinct code object — polymorphic dispatch. A hand-written prototype that merges all 20 ADD patterns into ONE function gives **-18% on rewrite calls** by eliminating function call overhead and caching shared locals (`_s0, _s1 = uop.src`).

You're automating this: modify tinygrad's UPat compilation pipeline to emit one merged function per op instead of N separate functions.

## Key files

- `~/Documents/tinygrad/tinygrad/uop/upat.py` — the UPat compiler. `_get_clause` builds a clause tree, `_get_code` renders it to Python source, `upat_compile` exec's it. This is where the mega-matcher generator goes.
- `~/Documents/tinygrad/tinygrad/uop/ops.py` — `PatternMatcher` class (line ~1241). `__init__` builds `pdict[op] -> list[entry]`. `rewrite()` (line ~1259) is the hot loop. Hook the mega-matcher here.
- `~/Documents/tinygrad-pareto-frontier/bench_mega_match.py` — hand-written prototype benchmark. Shows the -18% result.
- `~/Documents/tinygrad-pareto-frontier/HYPOTHESIS_GRAPH.md` — full investigation (21 hypotheses). H₁₈ is the mega-matcher. H₂₁ (automated generation) is pending.

## Why string manipulation failed

The first automation attempt (naive approach):
1. Call `_get_code(upat, has_ctx)` for each pattern → get Python source string
2. Rename `_fxn` → `_f{i}`, `a0` → `_a0_p{i}` via regex
3. Concatenate all bodies into one function
4. Factor out shared `len(uop.src) == 2` prefix, add `_s0 = uop.src[0]; _s1 = uop.src[1]`

Failed because:
- **Indentation breakage**: multi-line pattern bodies (`if len(uop.src) == 2:` / indented children) can't be reliably re-indented via string ops
- **Regex collisions**: `re.sub(r'\ba0\b', '_a0_p5', line)` catches unintended matches in some patterns, changing behavior
- Without the local caching (`_s0`/`_s1`), naive concatenation gives only -1.5% — the -18% comes from eliminating redundant `uop.src[0]` / `uop.src[1]` accesses

## The right approach

Modify the existing AST-level pipeline instead of post-hoc string manipulation.

### Option A: Extend `_final_render` with locals binding

`_final_render(x, has_ctx, depth=1)` renders a clause tree (UOp graph) to Python if-chain lines. It already handles `UOp.src[i]` references via `Ops.GEP` nodes in the clause tree. 

Change: add a `locals_map: dict[str, str]` parameter. When rendering, if a `GEP(base, i)` expression matches a key in `locals_map`, emit the local variable name instead of `base.src[i]`. 

At the mega-matcher level: after the `len(src) == 2` check, bind `_s0 = uop.src[0]` and `_s1 = uop.src[1]`, then pass `locals_map={"uop.src[0]": "_s0", "uop.src[1]": "_s1"}` to each pattern's render.

### Option B: Merge at the clause tree level

Instead of rendering each pattern separately and concatenating strings, merge the clause TREES before rendering:

1. For each pattern, call `_get_clause(upat, base, skip_op=True)` → get UOp clause tree
2. Wrap each clause tree with its pattern-specific `_fxn{i}` reference
3. Build one big `UOp(Ops.AND, src=(...all patterns...))` tree
4. Run `pm_proc` and `pm_renderer` once on the merged tree
5. `_final_render` produces the merged function

This is cleaner but harder — the existing `pm_proc` optimization pass may not handle 20 patterns' worth of OR clauses (it already bails at ≥4 OR clauses in `do_process_and`).

### Option C: Template-based generation (pragmatic)

Skip the existing pipeline. Generate the mega-matcher directly:

1. For each pattern, call `_get_code` to get the per-pattern source
2. Parse each source with `ast.parse` (Python's AST module) instead of regex
3. Use AST transforms to: rename `_fxn` → `_f{i}`, rename dyn_lookup vars, substitute `uop.src[0]` → `_s0`
4. Use `ast.unparse` to emit clean Python with correct indentation
5. Wrap in the shared prefix (`len` check, local bindings)

This is robust to indentation and doesn't require modifying the existing pipeline.

## Recommended: Option B (upstream merge)

Merge the clause trees BEFORE rendering. The pipeline is `_get_clause` → `pm_proc` → `pm_renderer` → `_final_render`. The merge point is between `_get_clause` and `pm_proc`:

1. For each pattern i, call `_get_clause(upat, base, skip_op=True)` → get `UOp(Ops.AND, src=(...))` clause tree
2. Each clause tree references `_fxn`. Replace with `_f{i}` in the tree (rename the BIND/STORE nodes, not strings)
3. Similarly rename dyn_lookup vars in the tree nodes (`a0` → per-pattern unique names)
4. Combine: wrap all N clause trees in a top-level structure that tries each in order — an OR-of-ANDs, or just sequential ANDs with early return
5. Run `pm_proc` on the merged tree (lift the ≥4 OR clause limit for mega mode)
6. Run `pm_renderer` and `_final_render` as usual — they handle indentation, local binding, deduplication naturally

The key insight: `_get_clause` produces a UOp graph where `Ops.CUSTOM` nodes contain format strings like `"{0}.src[0]"`. When two patterns share the same `base.gep(0)` expression, the graph naturally deduplicates via UOp's hash-consing. Shared prefix checks (`len(uop.src) == 2`, `uop.src[0].op == 55`) appear ONCE in the merged graph if the UOps are structurally identical.

The `_s0`/`_s1` local caching falls out naturally: `pm_renderer` resolves `GEP(base, 0)` to `uop.src[0]`. If we pre-bind `base.gep(0)` to a `DEFINE_VAR("_s0")` node in the merged tree, the renderer emits `_s0` everywhere.

The `≥4 OR clauses` limit in `do_process_and` (line 61) needs to be lifted or bypassed. For mega mode with 20 patterns, the OR clause count is 20. Options: raise the limit, or restructure so each pattern is a sequential AND (not an OR alternative) — which is how the hand-written version works (sequential if-return, not if-elif).

```python
def _get_mega_code(patterns_with_fxns: list[tuple[UPat, Callable]]):
    clauses = []
    for i, (upat, fxn) in enumerate(patterns_with_fxns):
        clause = _get_clause(upat, UOp(Ops.NOOP, arg="uop"), skip_op=True)
        clause = _rename_fxn_in_tree(clause, i)  # _fxn → _f{i}
        clauses.append(clause)
    # sequential: try each clause, return first match
    # NOT an OR — each clause is independent with its own return
    merged = _build_sequential_clauses(clauses)
    # run through existing pipeline
    merged = graph_rewrite(merged, pm_proc)
    ...
```

## Verification

1. `MEGA_MATCH=1 python3 -m pytest test/null/test_upat_compile.py test/null/test_pattern_matcher.py -x`
2. `MEGA_MATCH=1 python3 -c "from tinygrad import Tensor; Tensor.randn(1,3,8,8).conv2d(Tensor.randn(4,3,3,3)).relu().realize()"`
3. Benchmark: compare `MEGA_MATCH=0` vs `MEGA_MATCH=1` on the rewrite micro-bench (see bench_mega_match.py for the test nodes)
4. Target: ≥-10% on the rewrite micro-bench (the hand-written prototype got -18%, automated should get ≥-10% to be worth shipping)

## Outcome (2026-05-08)

**Shipped Option B (simplified): -15.2% on ADD rewrite micro-bench.** 70 lines across `upat.py` and `ops.py`.

Approach: run each pattern through the existing pipeline (`_get_clause` → `pm_proc` → `pm_renderer` → `_final_render`), namespace dyn_lookup vars, cache `_s0`/`_s1`/`_s0op`/`_s1op` as shared locals, concatenate into one function. Gated behind `MEGA_MATCH=1`.

**Killed approaches:**
- Bitmask pre-filter: Python big ints (`1 << 128`) cost more than the attribute lookups they replace. Right in C, wrong in CPython.
- Pattern reordering + len gate: changes match priority, broke correctness on conv2d (Ops.AFTER spec failure). The 3% gap to hand-written (-18%) lives here.

Benchmark (3 trials, Python 3.14.4):
```
Trial 1: orig=2640ns mega=2245ns delta=-15.0%
Trial 2: orig=2644ns mega=2243ns delta=-15.2%
Trial 3: orig=2645ns mega=2239ns delta=-15.3%
```

## Context

- tinygrad is at `~/Documents/tinygrad`, branch `skip-root-op-check` (has the skip_op PR merged)
- CPython 3.16 with JIT is at `~/Documents/cpython-jit` (release build, `--enable-experimental-jit`)
- The investigation spans `~/Documents/tinygrad-pareto-frontier/HYPOTHESIS_GRAPH.md` (21 hypotheses)
- The hand-written benchmark is at `~/Documents/tinygrad-pareto-frontier/bench_mega_match.py`
