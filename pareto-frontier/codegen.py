"""Generate C code for PatternMatcher structural checks.

Reads UPat trees from PatternMatcher instances and emits C functions
that do the integer comparisons (op, len(src), src[i].op, dtype, arg)
without going through the Python interpreter.
"""
import sys, os, itertools
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UPat, Ops


# C struct matching the Python UOpFlat
C_HEADER = """
#include <stdint.h>

typedef struct {
    int op;
    int n_src;
    int src_ops[8];
    int64_t dtype_id;
    int64_t arg_int;
    int arg_is_int;
} UOpFlat;
"""


def _upat_src0_ops(upat: UPat) -> list[tuple[int, ...]] | None:
    """Extract (position, required_op_values) pairs from the UPat's src constraints.

    Returns a list of (src_index, op_value) checks, or None if no src constraint.
    """
    if upat.src is None or len(upat.src) == 0:
        return None

    checks = []
    first_perm = upat.src[0]

    if isinstance(first_perm, itertools.repeat):
        return None

    if not hasattr(first_perm, '__len__'):
        return None

    for i, child in enumerate(first_perm):
        if i >= 8:
            break
        if isinstance(child, UPat) and child.op is not None:
            checks.append((i, tuple(int(o) for o in child.op)))

    return checks if checks else None


def generate_match_fn(op: Ops, entries: list, fn_name: str, dtype_ids: dict) -> str:
    """Generate a C function that checks all patterns for one op.

    Returns the C source for a function:
      int fn_name(UOpFlat* u)
    that returns the index of the first matching pattern, or -1.
    """
    lines = [f"int {fn_name}(UOpFlat* u) {{"]

    for idx, entry in enumerate(entries):
        upat = entry[0]
        conditions = []

        # len(src) check
        if upat.strict_length:
            conditions.append(f"u->n_src == {upat.required_len}")
        elif upat.required_len > 0:
            conditions.append(f"u->n_src >= {upat.required_len}")

        # dtype check (compare by id)
        if upat.match_dtype is not None:
            dtype_checks = []
            for dt in upat.match_dtype:
                dt_id = dtype_ids.get(id(dt))
                if dt_id is not None:
                    dtype_checks.append(f"u->dtype_id == {dt_id}LL")
            if dtype_checks:
                if len(dtype_checks) == 1:
                    conditions.append(dtype_checks[0])
                else:
                    conditions.append("(" + " || ".join(dtype_checks) + ")")

        # arg check (integer only)
        if upat.arg is not None and isinstance(upat.arg, int):
            conditions.append(f"u->arg_is_int && u->arg_int == {upat.arg}LL")

        # src[i].op checks (depth 1 only)
        src_checks = _upat_src0_ops(upat)
        if src_checks:
            for src_idx, op_values in src_checks:
                if len(op_values) == 1:
                    conditions.append(f"u->src_ops[{src_idx}] == {op_values[0]}")
                else:
                    checks = " || ".join(f"u->src_ops[{src_idx}] == {v}" for v in op_values)
                    conditions.append(f"({checks})")

        # Handle forked patterns (multiple src permutations)
        if upat.src is not None and len(upat.src) > 1 and all(isinstance(p, tuple) for p in upat.src):
            perm_conditions = []
            for perm in upat.src:
                perm_checks = list(conditions)  # copy base conditions
                for i, child in enumerate(perm):
                    if i >= 8:
                        break
                    if isinstance(child, UPat) and child.op is not None:
                        if len(child.op) == 1:
                            perm_checks.append(f"u->src_ops[{i}] == {int(child.op[0])}")
                        else:
                            checks = " || ".join(f"u->src_ops[{i}] == {int(o)}" for o in child.op)
                            perm_checks.append(f"({checks})")
                if perm_checks:
                    perm_conditions.append("(" + " && ".join(perm_checks) + ")")

            if perm_conditions:
                lines.append(f"    if ({' || '.join(perm_conditions)}) return {idx};")
                continue

        if conditions:
            lines.append(f"    if ({' && '.join(conditions)}) return {idx};")
        else:
            # No structural checks — this pattern matches anything with this op
            # Must still call the Python callback, so return this index
            lines.append(f"    return {idx};")
            break  # no point checking further patterns

    lines.append("    return -1;")
    lines.append("}")
    return "\n".join(lines)


def generate_c_for_pm(pm: PatternMatcher, pm_id: int, dtype_ids: dict) -> str:
    """Generate all C match functions for one PatternMatcher."""
    functions = []
    fn_names = {}  # op -> fn_name

    for op, entries in pm.pdict.items():
        fn_name = f"match_pm{pm_id}_{op.name}"
        fn_names[op] = fn_name
        functions.append(generate_match_fn(op, entries, fn_name, dtype_ids))

    return "\n\n".join(functions), fn_names


def generate_all(pms: list[PatternMatcher], min_patterns: int = 3) -> tuple[str, dict]:
    """Generate C code for all PatternMatchers above the pattern threshold.

    Returns (c_source, {pm_id: {op: fn_name}}).
    """
    # Build dtype id mapping (id(dtype_object) -> integer constant)
    dtype_ids = {}
    for pm in pms:
        for p, _ in pm.patterns:
            if p.match_dtype:
                for dt in p.match_dtype:
                    if id(dt) not in dtype_ids:
                        dtype_ids[id(dt)] = id(dt)

    all_code = [C_HEADER]
    all_fn_names = {}

    for i, pm in enumerate(pms):
        if len(pm.patterns) < min_patterns:
            continue
        code, fn_names = generate_c_for_pm(pm, i, dtype_ids)
        all_code.append(f"// PM {i}: {len(pm.patterns)} patterns")
        all_code.append(code)
        all_fn_names[id(pm)] = (i, fn_names)

    return "\n\n".join(all_code), all_fn_names


if __name__ == "__main__":
    from tinygrad import Tensor
    Tensor.randn(32, 128).softmax().realize()

    import gc
    pms = [obj for obj in gc.get_objects() if isinstance(obj, PatternMatcher) and len(obj.patterns) >= 10]
    pms.sort(key=lambda p: -len(p.patterns))

    c_code, fn_map = generate_all(pms[:5])

    print(f"Generated {len(c_code)} bytes of C code for {len(fn_map)} PMs")
    print(f"Functions: {sum(len(fns) for _, fns in fn_map.values())}")
    print()

    # Show a sample
    lines = c_code.split('\n')
    for line in lines[:60]:
        print(line)
    if len(lines) > 60:
        print(f"... ({len(lines) - 60} more lines)")
