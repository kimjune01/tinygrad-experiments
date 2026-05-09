"""Benchmark: mega-matcher vs per-pattern dispatch for ADD in symbolic PM.

Hypothesis: merging 20 ADD patterns into ONE function makes the call site
monomorphic, enabling JIT inlining and eliminating per-call overhead.

Usage:
  python3 bench_mega_match.py           # system python
  PYTHON_JIT=1 ./python.exe bench_mega_match.py  # cpython 3.16 with JIT
"""
import sys, os, time, types, inspect
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))
os.environ["UPAT_COMPILE"] = "1"

from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, deconstruct_function
from tinygrad.uop.symbolic import symbolic
from tinygrad.uop.upat import upat_compile, _get_code
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context

# === Step 1: Extract all 20 ADD pattern functions and their dyn_lookup bindings ===

add_entries = symbolic.pdict[Ops.ADD]
_fxns = []
_lookups = []

for entry in add_entries:
    p = entry[0]
    for pp, fxn in symbolic.patterns:
        if pp is p:
            real_fxn = types.FunctionType(*deconstruct_function(fxn))
            has_ctx = 'ctx' in inspect.signature(real_fxn).parameters
            with Context(SPEC=0): code = _get_code(p, has_ctx)
            _fxns.append(real_fxn)
            _lookups.append(code[1] if code else {})
            break

# === Step 2: Build mega-matcher globals ===
# Each pattern's _fxn becomes _fxn0, _fxn1, etc.
# Each pattern's dyn_lookup entries get prefixed: a0 → a0_p2, etc.

mega_globals = {}
for i in range(len(_fxns)):
    mega_globals[f"_fxn{i}"] = _fxns[i]
    for k, v in _lookups[i].items():
        mega_globals[f"{k}_p{i}"] = v

# === Step 3: Generate mega-matcher source code ===
# Inline all 20 patterns, preserving index order.
# Shared prefix: len(uop.src) == 2 checked once, s0/s1 cached.

mega_code = '''\
def mega_match_ADD(uop, ctx):
  # p4: dtype==weakint (no src constraint)
  if (uop.dtype == a0_p4 or uop.dtype._scalar == a0_p4):
    if (_ret:=_fxn4(x=uop)) is not None: return _ret
  if len(uop.src) != 2:
    # p7: dtype==weakint (no src constraint)
    if (uop.dtype == a0_p7 or uop.dtype._scalar == a0_p7):
      if (_ret:=_fxn7(x=uop)) is not None: return _ret
    return None
  s0, s1 = uop.src
  s0op, s1op = s0.op, s1.op
  # p0: WHERE(cond, x, CONST(Invalid)) + y
  if s0op == 55 and len(s0.src) == 3 and s0.src[2].op == 63 and s0.src[2].arg == a0_p0:
    if (_ret:=_fxn0(alu=uop, cond=s0.src[0], x=s0.src[1], i=s0.src[2], y=s1)) is not None: return _ret
  # p1: y + WHERE(cond, x, CONST(Invalid))
  if s1op == 55 and len(s1.src) == 3 and s1.src[2].op == 63 and s1.src[2].arg == a0_p1:
    if (_ret:=_fxn1(alu=uop, y=s0, cond=s1.src[0], x=s1.src[1], i=s1.src[2])) is not None: return _ret
  # p2: CONST(Invalid) + x (commutative)
  if s0op == 63 and s0.arg == a0_p2:
    if (_ret:=_fxn2(i=s0)) is not None: return _ret
  if s1op == 63 and s1.arg == a0_p2:
    if (_ret:=_fxn2(i=s1)) is not None: return _ret
  # p3: x + 0 → x (commutative)
  if s1op == 63 and s1.arg == 0:
    if (_ret:=_fxn3(x=s0)) is not None: return _ret
  if s0op == 63 and s0.arg == 0:
    if (_ret:=_fxn3(x=s1)) is not None: return _ret
  # p5: CONST + CONST folding
  if s0op in a0_p5 and s1op in a0_p5:
    if (_ret:=_fxn5(a=uop)) is not None: return _ret
  # p6: bool + bool → xor
  if (uop.dtype == a0_p6 or uop.dtype._scalar == a0_p6):
    if (s0.dtype == a0_p6 or s0.dtype._scalar == a0_p6) and (s1.dtype == a0_p6 or s1.dtype._scalar == a0_p6):
      if (_ret:=_fxn6(x=s0, y=s1)) is not None: return _ret
      if (_ret:=_fxn6(y=s0, x=s1)) is not None: return _ret
  # p7: commutative flip (dtype==weakint)
  if (uop.dtype == a0_p7 or uop.dtype._scalar == a0_p7):
    if (_ret:=_fxn7(x=uop)) is not None: return _ret
  # p8: x*c0 + x*c1 → x*(c0+c1)
  if s0op == 37 and len(s0.src) == 2 and s1op == 37 and len(s1.src) == 2:
    if s0.src[1].op in a0_p8 and s1.src[1].op in a0_p8 and s0.src[0] is s1.src[0]:
      if (_ret:=_fxn8(x=s0.src[0], c0=s0.src[1], c1=s1.src[1])) is not None: return _ret
    if s0.src[1].op in a0_p8 and s1.src[0].op in a0_p8 and s0.src[0] is s1.src[1]:
      if (_ret:=_fxn8(x=s0.src[0], c0=s0.src[1], c1=s1.src[0])) is not None: return _ret
    if s0.src[0].op in a0_p8 and s1.src[1].op in a0_p8 and s0.src[1] is s1.src[0]:
      if (_ret:=_fxn8(c0=s0.src[0], x=s0.src[1], c1=s1.src[1])) is not None: return _ret
    if s0.src[0].op in a0_p8 and s1.src[0].op in a0_p8 and s0.src[1] is s1.src[1]:
      if (_ret:=_fxn8(c0=s0.src[0], x=s0.src[1], c1=s1.src[0])) is not None: return _ret
    if s0.src[1].op in a0_p8 and s1.src[1].op in a0_p8 and s0.src[0] is s1.src[0]:
      if (_ret:=_fxn8(x=s0.src[0], c1=s0.src[1], c0=s1.src[1])) is not None: return _ret
    if s0.src[1].op in a0_p8 and s1.src[0].op in a0_p8 and s0.src[0] is s1.src[1]:
      if (_ret:=_fxn8(x=s0.src[0], c1=s0.src[1], c0=s1.src[0])) is not None: return _ret
    if s0.src[0].op in a0_p8 and s1.src[1].op in a0_p8 and s0.src[1] is s1.src[0]:
      if (_ret:=_fxn8(c1=s0.src[0], x=s0.src[1], c0=s1.src[1])) is not None: return _ret
    if s0.src[0].op in a0_p8 and s1.src[0].op in a0_p8 and s0.src[1] is s1.src[1]:
      if (_ret:=_fxn8(c1=s0.src[0], x=s0.src[1], c0=s1.src[0])) is not None: return _ret
  # p9: (y + x*c0) + x*c1 — complex, many permutations
  if s0op == 36 and len(s0.src) == 2 and s1op == 37 and len(s1.src) == 2:
    if s0.src[1].op == 37 and len(s0.src[1].src) == 2 and s1.src[1].op in a0_p9:
      if s0.src[1].src[1].op in a0_p9 and s0.src[1].src[0] is s1.src[0]:
        if (_ret:=_fxn9(x=s0.src[1].src[0], c0=s0.src[1].src[1], y=s0.src[0], c1=s1.src[1])) is not None: return _ret
      if s0.src[1].src[0].op in a0_p9 and s0.src[1].src[1] is s1.src[0]:
        if (_ret:=_fxn9(c0=s0.src[1].src[0], x=s0.src[1].src[1], y=s0.src[0], c1=s1.src[1])) is not None: return _ret
    if s0.src[1].op == 37 and len(s0.src[1].src) == 2 and s1.src[0].op in a0_p9:
      if s0.src[1].src[1].op in a0_p9 and s0.src[1].src[0] is s1.src[1]:
        if (_ret:=_fxn9(x=s0.src[1].src[0], c0=s0.src[1].src[1], y=s0.src[0], c1=s1.src[0])) is not None: return _ret
      if s0.src[1].src[0].op in a0_p9 and s0.src[1].src[1] is s1.src[1]:
        if (_ret:=_fxn9(c0=s0.src[1].src[0], x=s0.src[1].src[1], y=s0.src[0], c1=s1.src[0])) is not None: return _ret
    if s0.src[0].op == 37 and len(s0.src[0].src) == 2 and s1.src[1].op in a0_p9:
      if s0.src[0].src[1].op in a0_p9 and s0.src[0].src[0] is s1.src[0]:
        if (_ret:=_fxn9(x=s0.src[0].src[0], c0=s0.src[0].src[1], y=s0.src[1], c1=s1.src[1])) is not None: return _ret
      if s0.src[0].src[0].op in a0_p9 and s0.src[0].src[1] is s1.src[0]:
        if (_ret:=_fxn9(c0=s0.src[0].src[0], x=s0.src[0].src[1], y=s0.src[1], c1=s1.src[1])) is not None: return _ret
    if s0.src[0].op == 37 and len(s0.src[0].src) == 2 and s1.src[0].op in a0_p9:
      if s0.src[0].src[1].op in a0_p9 and s0.src[0].src[0] is s1.src[1]:
        if (_ret:=_fxn9(x=s0.src[0].src[0], c0=s0.src[0].src[1], y=s0.src[1], c1=s1.src[0])) is not None: return _ret
      if s0.src[0].src[0].op in a0_p9 and s0.src[0].src[1] is s1.src[1]:
        if (_ret:=_fxn9(c0=s0.src[0].src[0], x=s0.src[0].src[1], y=s0.src[1], c1=s1.src[0])) is not None: return _ret
  if s0op == 37 and len(s0.src) == 2 and s1op == 36 and len(s1.src) == 2:
    if s0.src[1].op in a0_p9 and s1.src[1].op == 37 and len(s1.src[1].src) == 2:
      if s1.src[1].src[1].op in a0_p9 and s1.src[1].src[0] is s0.src[0]:
        if (_ret:=_fxn9(x=s1.src[1].src[0], c0=s1.src[1].src[1], y=s1.src[0], c1=s0.src[1])) is not None: return _ret
      if s1.src[1].src[0].op in a0_p9 and s1.src[1].src[1] is s0.src[0]:
        if (_ret:=_fxn9(c0=s1.src[1].src[0], x=s1.src[1].src[1], y=s1.src[0], c1=s0.src[1])) is not None: return _ret
    if s0.src[1].op in a0_p9 and s1.src[0].op == 37 and len(s1.src[0].src) == 2:
      if s1.src[0].src[1].op in a0_p9 and s1.src[0].src[0] is s0.src[0]:
        if (_ret:=_fxn9(x=s1.src[0].src[0], c0=s1.src[0].src[1], y=s1.src[1], c1=s0.src[1])) is not None: return _ret
      if s1.src[0].src[0].op in a0_p9 and s1.src[0].src[1] is s0.src[0]:
        if (_ret:=_fxn9(c0=s1.src[0].src[0], x=s1.src[0].src[1], y=s1.src[1], c1=s0.src[1])) is not None: return _ret
    if s0.src[0].op in a0_p9 and s1.src[1].op == 37 and len(s1.src[1].src) == 2:
      if s1.src[1].src[1].op in a0_p9 and s1.src[1].src[0] is s0.src[1]:
        if (_ret:=_fxn9(x=s1.src[1].src[0], c0=s1.src[1].src[1], y=s1.src[0], c1=s0.src[0])) is not None: return _ret
      if s1.src[1].src[0].op in a0_p9 and s1.src[1].src[1] is s0.src[1]:
        if (_ret:=_fxn9(c0=s1.src[1].src[0], x=s1.src[1].src[1], y=s1.src[0], c1=s0.src[0])) is not None: return _ret
    if s0.src[0].op in a0_p9 and s1.src[0].op == 37 and len(s1.src[0].src) == 2:
      if s1.src[0].src[1].op in a0_p9 and s1.src[0].src[0] is s0.src[1]:
        if (_ret:=_fxn9(x=s1.src[0].src[0], c0=s1.src[0].src[1], y=s1.src[1], c1=s0.src[0])) is not None: return _ret
      if s1.src[0].src[0].op in a0_p9 and s1.src[0].src[1] is s0.src[1]:
        if (_ret:=_fxn9(c0=s1.src[0].src[0], x=s1.src[0].src[1], y=s1.src[1], c1=s0.src[0])) is not None: return _ret
  # p10: x + x*c → x*(c+1)
  if s1op == 37 and len(s1.src) == 2:
    if s1.src[1].op in a0_p10 and s1.src[0] is s0:
      if (_ret:=_fxn10(x=s1.src[0], c=s1.src[1])) is not None: return _ret
    if s1.src[0].op in a0_p10 and s1.src[1] is s0:
      if (_ret:=_fxn10(c=s1.src[0], x=s1.src[1])) is not None: return _ret
  if s0op == 37 and len(s0.src) == 2:
    if s0.src[1].op in a0_p10 and s0.src[0] is s1:
      if (_ret:=_fxn10(x=s0.src[0], c=s0.src[1])) is not None: return _ret
    if s0.src[0].op in a0_p10 and s0.src[1] is s1:
      if (_ret:=_fxn10(c=s0.src[0], x=s0.src[1])) is not None: return _ret
  # p11: (y+x) + x*c
  if s0op == 36 and len(s0.src) == 2 and s1op == 37 and len(s1.src) == 2:
    if s1.src[1].op in a0_p11 and s0.src[1] is s1.src[0]:
      if (_ret:=_fxn11(y=s0.src[0], x=s0.src[1], c=s1.src[1])) is not None: return _ret
    if s1.src[0].op in a0_p11 and s0.src[1] is s1.src[1]:
      if (_ret:=_fxn11(y=s0.src[0], x=s0.src[1], c=s1.src[0])) is not None: return _ret
    if s1.src[1].op in a0_p11 and s0.src[0] is s1.src[0]:
      if (_ret:=_fxn11(x=s0.src[0], y=s0.src[1], c=s1.src[1])) is not None: return _ret
    if s1.src[0].op in a0_p11 and s0.src[0] is s1.src[1]:
      if (_ret:=_fxn11(x=s0.src[0], y=s0.src[1], c=s1.src[0])) is not None: return _ret
  if s0op == 37 and len(s0.src) == 2 and s1op == 36 and len(s1.src) == 2:
    if s0.src[1].op in a0_p11 and s0.src[0] is s1.src[1]:
      if (_ret:=_fxn11(x=s0.src[0], c=s0.src[1], y=s1.src[0])) is not None: return _ret
    if s0.src[1].op in a0_p11 and s0.src[0] is s1.src[0]:
      if (_ret:=_fxn11(x=s0.src[0], c=s0.src[1], y=s1.src[1])) is not None: return _ret
    if s0.src[0].op in a0_p11 and s0.src[1] is s1.src[1]:
      if (_ret:=_fxn11(c=s0.src[0], x=s0.src[1], y=s1.src[0])) is not None: return _ret
    if s0.src[0].op in a0_p11 and s0.src[1] is s1.src[0]:
      if (_ret:=_fxn11(c=s0.src[0], x=s0.src[1], y=s1.src[1])) is not None: return _ret
  # p12: (y + x*c) + x
  if s0op == 36 and len(s0.src) == 2:
    if s0.src[1].op == 37 and len(s0.src[1].src) == 2:
      if s0.src[1].src[1].op in a0_p12 and s0.src[1].src[0] is s1:
        if (_ret:=_fxn12(x=s0.src[1].src[0], c=s0.src[1].src[1], y=s0.src[0])) is not None: return _ret
      if s0.src[1].src[0].op in a0_p12 and s0.src[1].src[1] is s1:
        if (_ret:=_fxn12(c=s0.src[1].src[0], x=s0.src[1].src[1], y=s0.src[0])) is not None: return _ret
    if s0.src[0].op == 37 and len(s0.src[0].src) == 2:
      if s0.src[0].src[1].op in a0_p12 and s0.src[0].src[0] is s1:
        if (_ret:=_fxn12(x=s0.src[0].src[0], c=s0.src[0].src[1], y=s0.src[1])) is not None: return _ret
      if s0.src[0].src[0].op in a0_p12 and s0.src[0].src[1] is s1:
        if (_ret:=_fxn12(c=s0.src[0].src[0], x=s0.src[0].src[1], y=s0.src[1])) is not None: return _ret
  if s1op == 36 and len(s1.src) == 2:
    if s1.src[1].op == 37 and len(s1.src[1].src) == 2:
      if s1.src[1].src[1].op in a0_p12 and s1.src[1].src[0] is s0:
        if (_ret:=_fxn12(x=s1.src[1].src[0], c=s1.src[1].src[1], y=s1.src[0])) is not None: return _ret
      if s1.src[1].src[0].op in a0_p12 and s1.src[1].src[1] is s0:
        if (_ret:=_fxn12(c=s1.src[1].src[0], x=s1.src[1].src[1], y=s1.src[0])) is not None: return _ret
    if s1.src[0].op == 37 and len(s1.src[0].src) == 2:
      if s1.src[0].src[1].op in a0_p12 and s1.src[0].src[0] is s0:
        if (_ret:=_fxn12(x=s1.src[0].src[0], c=s1.src[0].src[1], y=s1.src[1])) is not None: return _ret
      if s1.src[0].src[0].op in a0_p12 and s1.src[0].src[1] is s0:
        if (_ret:=_fxn12(c=s1.src[0].src[0], x=s1.src[0].src[1], y=s1.src[1])) is not None: return _ret
  # p13: x + x → 2*x
  if s0 is s1:
    if (_ret:=_fxn13(x=s0)) is not None: return _ret
  # p14: (y+x) + x
  if s0op == 36 and len(s0.src) == 2:
    if s0.src[1] is s1:
      if (_ret:=_fxn14(y=s0.src[0], x=s0.src[1])) is not None: return _ret
    if s0.src[0] is s1:
      if (_ret:=_fxn14(x=s0.src[0], y=s0.src[1])) is not None: return _ret
  if s1op == 36 and len(s1.src) == 2:
    if s1.src[1] is s0:
      if (_ret:=_fxn14(y=s1.src[0], x=s1.src[1])) is not None: return _ret
    if s1.src[0] is s0:
      if (_ret:=_fxn14(x=s1.src[0], y=s1.src[1])) is not None: return _ret
  # p15: WHERE + WHERE (same cond)
  if s0op == 55 and len(s0.src) == 3 and s1op == 55 and len(s1.src) == 3 and s0.src[0] is s1.src[0]:
    if (_ret:=_fxn15(alu=uop, c=s0.src[0], t=s0.src[1], f=s0.src[2], tt=s1.src[1], ff=s1.src[2])) is not None: return _ret
  # p16: (y + WHERE) + WHERE (same cond)
  if s0op == 36 and len(s0.src) == 2 and s1op == 55 and len(s1.src) == 3:
    if s0.src[1].op == 55 and len(s0.src[1].src) == 3 and s0.src[1].src[0] is s1.src[0]:
      if (_ret:=_fxn16(y=s0.src[0], c=s0.src[1].src[0], t=s0.src[1].src[1], f=s0.src[1].src[2], tt=s1.src[1], ff=s1.src[2])) is not None: return _ret
    if s0.src[0].op == 55 and len(s0.src[0].src) == 3 and s0.src[0].src[0] is s1.src[0]:
      if (_ret:=_fxn16(c=s0.src[0].src[0], t=s0.src[0].src[1], f=s0.src[0].src[2], y=s0.src[1], tt=s1.src[1], ff=s1.src[2])) is not None: return _ret
  if s0op == 55 and len(s0.src) == 3 and s1op == 36 and len(s1.src) == 2:
    if s1.src[1].op == 55 and len(s1.src[1].src) == 3 and s1.src[1].src[0] is s0.src[0]:
      if (_ret:=_fxn16(y=s1.src[0], c=s1.src[1].src[0], t=s1.src[1].src[1], f=s1.src[1].src[2], tt=s0.src[1], ff=s0.src[2])) is not None: return _ret
    if s1.src[0].op == 55 and len(s1.src[0].src) == 3 and s1.src[0].src[0] is s0.src[0]:
      if (_ret:=_fxn16(c=s1.src[0].src[0], t=s1.src[0].src[1], f=s1.src[0].src[2], y=s1.src[1], tt=s0.src[1], ff=s0.src[2])) is not None: return _ret
  # p17: (x + c1) + c2 → x + (c1+c2)
  if s0op == 36 and len(s0.src) == 2 and s1op in a0_p17:
    if s0.src[1].op in a0_p17:
      if (_ret:=_fxn17(x=s0.src[0], c1=s0.src[1], c2=s1, f=uop)) is not None: return _ret
    if s0.src[0].op in a0_p17:
      if (_ret:=_fxn17(c1=s0.src[0], x=s0.src[1], c2=s1, f=uop)) is not None: return _ret
  if s0op in a0_p17 and s1op == 36 and len(s1.src) == 2:
    if s1.src[1].op in a0_p17:
      if (_ret:=_fxn17(x=s1.src[0], c1=s1.src[1], c2=s0, f=uop)) is not None: return _ret
    if s1.src[0].op in a0_p17:
      if (_ret:=_fxn17(c1=s1.src[0], x=s1.src[1], c2=s0, f=uop)) is not None: return _ret
  # p18: (x + c1) + y → (x + y) + c1
  if s0op == 36 and len(s0.src) == 2:
    if s0.src[1].op in a0_p18:
      if (_ret:=_fxn18(x=s0.src[0], c1=s0.src[1], y=s1)) is not None: return _ret
    if s0.src[0].op in a0_p18:
      if (_ret:=_fxn18(c1=s0.src[0], x=s0.src[1], y=s1)) is not None: return _ret
  if s1op == 36 and len(s1.src) == 2:
    if s1.src[1].op in a0_p18:
      if (_ret:=_fxn18(x=s1.src[0], c1=s1.src[1], y=s0)) is not None: return _ret
    if s1.src[0].op in a0_p18:
      if (_ret:=_fxn18(c1=s1.src[0], x=s1.src[1], y=s0)) is not None: return _ret
  # p19: long dtype add
  if (s0.dtype == a0_p19 or s0.dtype._scalar == a0_p19) and (s1.dtype == a0_p19 or s1.dtype._scalar == a0_p19):
    if (_ret:=_fxn19(u=uop, x=s0, y=s1)) is not None: return _ret
  return None
'''

# Fix the globals extraction — _fxns is list of fxn objects, _lookups is list of dicts
mega_globals_fixed = {}
for i in range(len(_fxns)):
    mega_globals_fixed[f"_fxn{i}"] = _fxns[i]
    for k, v in _lookups[i].items():
        mega_globals_fixed[f"{k}_p{i}"] = v

namespace = {}
exec(mega_code, mega_globals_fixed, namespace)
mega_match_ADD = namespace["mega_match_ADD"]

# === Step 4: Correctness test ===

x = UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=("x", 0, 1024))
y = UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=("y", 0, 1024))
c0 = UOp(Ops.CONST, dtypes.weakint, arg=0)
c1 = UOp(Ops.CONST, dtypes.weakint, arg=1)
c2 = UOp(Ops.CONST, dtypes.weakint, arg=2)

test_nodes = [x + c0, x * c1, x + x, (x + y) + c2, x * c2 + y * c2, x // c1, x % c1,
              x + y, x + c1, (x + c1) + c2, x + (y * c2)]

# Force compilation of existing matchers
for n in test_nodes:
    symbolic.rewrite(n)

# Compare mega vs original for ADD nodes
correct = True
for n in test_nodes:
    if n.op != Ops.ADD: continue
    orig = symbolic.rewrite(n)
    mega = mega_match_ADD(n, None)
    if orig is None and mega is None: continue
    if orig is not None and mega is not None and orig is mega: continue
    # Check by value if not by identity
    if str(orig) == str(mega): continue
    print(f"MISMATCH on {n}: orig={orig}, mega={mega}")
    correct = False

if correct:
    print("Correctness: PASS (all ADD nodes match)")
else:
    print("Correctness: FAIL")
    sys.exit(1)

# === Step 5: Benchmark ===

# Warmup
for _ in range(2000):
    for n in test_nodes:
        if n.op == Ops.ADD: symbolic.rewrite(n)

for _ in range(2000):
    for n in test_nodes:
        if n.op == Ops.ADD: mega_match_ADD(n, None)

add_nodes = [n for n in test_nodes if n.op == Ops.ADD]
N = 50000

# Benchmark original (per-pattern dispatch)
start = time.perf_counter()
for _ in range(N):
    for n in add_nodes:
        symbolic.rewrite(n)
t_orig = time.perf_counter() - start

# Benchmark mega-matcher
start = time.perf_counter()
for _ in range(N):
    for n in add_nodes:
        mega_match_ADD(n, None)
t_mega = time.perf_counter() - start

n_calls = N * len(add_nodes)
orig_ns = t_orig / n_calls * 1e9
mega_ns = t_mega / n_calls * 1e9
delta = (mega_ns / orig_ns - 1) * 100

print(f"\nBenchmark ({len(add_nodes)} ADD nodes × {N} iterations = {n_calls} calls):")
print(f"  Original (per-pattern): {orig_ns:.0f}ns/rewrite")
print(f"  Mega-matcher:           {mega_ns:.0f}ns/rewrite")
print(f"  Delta:                  {delta:+.1f}%")
print(f"  JIT: {os.environ.get('PYTHON_JIT', 'N/A')}")
print(f"  Python: {sys.version.split()[0]}")
