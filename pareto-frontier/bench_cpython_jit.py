"""Benchmark PatternMatcher.rewrite on custom CPython with JIT.

Tests just the pattern matching hot path — no device, no realize.
Builds a realistic UOp graph and runs rewrite() in a loop.
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad.uop.ops import PatternMatcher, UPat, UOp, Ops, GroupOp
from tinygrad.uop.symbolic import symbolic
from tinygrad.dtype import dtypes

# Build a realistic PM (the symbolic simplifier — 118 patterns)
pm = symbolic
print(f"PM: {len(pm.patterns)} patterns")

# Build realistic UOps that exercise the pattern matcher
def make_test_graph():
    """Create a softmax-like expression graph."""
    nodes = []
    x = UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=("x", 0, 1024))
    y = UOp(Ops.DEFINE_VAR, dtypes.weakint, arg=("y", 0, 1024))
    c1 = UOp(Ops.CONST, dtypes.weakint, arg=1)
    c2 = UOp(Ops.CONST, dtypes.weakint, arg=2)
    c0 = UOp(Ops.CONST, dtypes.weakint, arg=0)

    # Build expressions that trigger symbolic patterns
    nodes.append(x + c0)        # ADD with CONST 0 → simplifies to x
    nodes.append(x * c1)        # MUL with CONST 1 → simplifies to x
    nodes.append(x + x)         # ADD self → x * 2
    nodes.append((x + y) + c2)  # nested ADD with CONST
    nodes.append(x * c2 + y * c2)  # factoring
    nodes.append(x // c1)       # FLOORDIV by 1
    nodes.append(x % c1)        # FLOORMOD by 1

    return nodes

nodes = make_test_graph()
print(f"Test nodes: {len(nodes)}")

# Warm up patterns (force compilation)
for node in nodes:
    pm.rewrite(node)

# Benchmark
n_iters = 50000
total_rewrites = n_iters * len(nodes)

start = time.perf_counter()
for _ in range(n_iters):
    for node in nodes:
        pm.rewrite(node)
elapsed = time.perf_counter() - start

per_rewrite = elapsed / total_rewrites * 1e9

print(f"\nResults ({n_iters} iterations × {len(nodes)} nodes = {total_rewrites} rewrites):")
print(f"  Total: {elapsed*1e3:.1f}ms")
print(f"  Per rewrite: {per_rewrite:.0f}ns")
print(f"  JIT: {'enabled' if os.environ.get('PYTHON_JIT') == '1' else 'disabled'}")
print(f"  Python: {sys.version}")
