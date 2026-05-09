"""tinygrad-turbo: compile pattern matchers to C.

Usage:
    import tinygrad_turbo
    tinygrad_turbo.install()   # compiles all PMs to C, patches rewrite
    tinygrad_turbo.uninstall() # restores original rewrite
"""
import sys, os, gc, ctypes
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UOp
from compile_c import UOpFlat, flatten_uop, compile_c, bind_function
from codegen import generate_all

_ORIGINAL_REWRITE = PatternMatcher.rewrite
_installed = False


def install(min_patterns: int = 3, verbose: bool = True):
    """Compile all PatternMatchers to C and monkey-patch rewrite."""
    global _installed
    if _installed:
        return

    pms = [obj for obj in gc.get_objects() if isinstance(obj, PatternMatcher) and len(obj.patterns) >= min_patterns]

    if verbose:
        print(f"tinygrad-turbo: compiling {len(pms)} PatternMatchers...")

    c_source, fn_map = generate_all(pms, min_patterns)
    lib = compile_c(c_source)

    # Bind C functions to each PM
    patched = 0
    for pm in pms:
        pm_key = id(pm)
        if pm_key not in fn_map:
            continue
        pm_idx, op_fns = fn_map[pm_key]
        pm._turbo_fns = {}
        for op, fn_name in op_fns.items():
            pm._turbo_fns[op] = bind_function(lib, fn_name)
        patched += 1

    PatternMatcher.rewrite = _turbo_rewrite
    _installed = True

    if verbose:
        total_fns = sum(len(fns) for _, fns in fn_map.values())
        print(f"tinygrad-turbo: {patched} PMs patched, {total_fns} C functions compiled")


def uninstall():
    global _installed
    PatternMatcher.rewrite = _ORIGINAL_REWRITE
    _installed = False


def _turbo_rewrite(self: PatternMatcher, uop: UOp, ctx=None):
    """Patched rewrite: C for structural checks, Python for callbacks."""
    turbo_fns = getattr(self, '_turbo_fns', None)
    if turbo_fns is None:
        return _ORIGINAL_REWRITE(self, uop, ctx)

    c_fn = turbo_fns.get(uop.op)
    if c_fn is None:
        return None

    flat = flatten_uop(uop)
    idx = c_fn(ctypes.byref(flat))

    if idx == -1:
        return None

    # C found a candidate — run the Python callback
    entries = self.pdict.get(uop.op)
    if entries is None or idx >= len(entries):
        return None

    # Still need early_reject check (C only does structural, not src_ops set)
    _, match, early_reject = entries[idx]
    if early_reject:
        if (ler := uop.__dict__.get('_src_ops')) is None:
            uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
        if not early_reject.issubset(ler):
            # C was optimistic — fall back to full Python scan from this point
            return _fallback_from(self, uop, ctx, idx + 1)

    ret = match(uop, ctx)
    if ret is not None and ret is not uop:
        return ret

    # C matched on structure but callback returned None — try remaining patterns
    return _fallback_from(self, uop, ctx, idx + 1)


def _fallback_from(self: PatternMatcher, uop: UOp, ctx, start_idx: int):
    """Continue the Python pattern loop from a given index."""
    entries = self.pdict.get(uop.op, [])
    if (ler := uop.__dict__.get('_src_ops')) is None:
        uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
    for _, match, early_reject in entries[start_idx:]:
        if not early_reject.issubset(ler):
            continue
        if (ret := match(uop, ctx)) is not None and ret is not uop:
            return ret
    return None


if __name__ == "__main__":
    from tinygrad import Tensor

    # Force PM creation
    Tensor.randn(32, 128).softmax().realize()

    install()

    # Test
    import time
    x = Tensor.randn(32, 128)
    start = time.perf_counter()
    x.softmax().realize()
    print(f"turbo softmax: {(time.perf_counter()-start)*1e3:.1f}ms")
