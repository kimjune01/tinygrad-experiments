# Verify: H5 theory transfer on CUDA

## What you're testing

The H5 investigation found that a semantic theory derived from gemm_1024 on Metal ("after TC: UPCAST N by largest divisor, UNROLL K by largest divisor, UPCAST M by largest divisor, LOCAL M by largest divisor") transfers to all 7 tested matmul shapes, beating the heuristic 5-9x.

This needs verification on CUDA (RTX 4080) to confirm it's not Metal-specific.

## Setup

```bash
cd Documents\tinygrad
git remote add fork git@github.com:kimjune01/tinygrad.git
git fetch fork
git checkout fork/post-tc-clean
pip install -e .
```

## Test 1: Theory transfer (heuristic vs adaptive theory)

Run each shape with `DEBUG=2` to see kernel names and timings. Compare the heuristic kernel (default) against manually applying the adaptive theory.

```bash
set IGNORE_BEAM_CACHE=1
set DEBUG=2

REM Square matmuls
python -c "from tinygrad import Tensor; Tensor.randn(1024, 1024).matmul(Tensor.randn(1024, 1024)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(256, 256).matmul(Tensor.randn(256, 256)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(2048, 2048).matmul(Tensor.randn(2048, 2048)).realize()"

REM Non-square
python -c "from tinygrad import Tensor; Tensor.randn(16, 4096).matmul(Tensor.randn(4096, 4096)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(8, 2048).matmul(Tensor.randn(2048, 2048)).realize()"
python -c "from tinygrad import Tensor; Tensor.randn(512, 2048).matmul(Tensor.randn(2048, 256)).realize()"
```

## Test 2: Abduction loop vs heuristic on individual kernels

Run this script to compare the heuristic, a 52-trial abduction loop, and NOOPT on the reduction kernel for each shape:

```python
import os
os.environ['IGNORE_BEAM_CACHE'] = '1'

from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops, AxisType
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.device import Buffer

ren = Device[Device.DEFAULT].renderer

def time_kernel(k, n=8):
    try:
        ast = k.get_optimized_ast(name_override="test")
        prg = to_program(ast, k.ren)
        glbls = sorted([x for x in ast.backward_slice if x.op == Ops.PARAM], key=lambda x: x.arg)
        bufs = [Buffer(ren.target.device, x.ptrdtype.size, x.dtype.base).ensure_allocated() for x in glbls]
        rt = get_runtime(prg.src[1].arg, prg)
        var_vals = {kk.expr: int(kk.vmax+kk.vmin)//2 for kk in ast.variables()}
        global_size, local_size = prg.arg.launch_dims(var_vals)
        raw_bufs = [b._buf for b in bufs]
        tms = []
        for _ in range(n):
            t = rt(*raw_bufs, global_size=global_size, local_size=local_size,
                   vals=prg.arg.vals(var_vals), wait=True)
            if t is not None: tms.append(t)
        return min(tms) * 1e6 if tms else float('inf')
    except Exception:
        return float('inf')

def apply_adaptive_theory(k_orig):
    """Semantic theory: TC, then UPCAST N, UNROLL K, UPCAST M, LOCAL M — each by largest divisor."""
    k = k_orig.copy()
    try:
        rngs = k.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
    except KernelOptError:
        return k_orig.copy()  # TC doesn't apply
    if rngs is None:
        return k
    # UPCAST N (rngs[1]) by largest of [4,3,2]
    for sz in [4,3,2]:
        if rngs[1].src[0].divides(sz) is not None:
            try: rngs[1] = k.apply_opt(Opt(OptOps.UPCAST, k.rngs.index(rngs[1]), sz))[0]
            except KernelOptError: continue
            break
    # UNROLL K by largest of [4,2]
    for sz in [4,2]:
        try:
            k.apply_opt(Opt(OptOps.UNROLL, 0, sz))
            break
        except KernelOptError: continue
    # UPCAST M (rngs[0]) by largest of [4,3,2]
    for sz in [4,3,2]:
        if rngs[0].src[0].divides(sz) is not None:
            try: k.apply_opt(Opt(OptOps.UPCAST, k.rngs.index(rngs[0]), sz))
            except KernelOptError: continue
            break
    # LOCAL M by largest of [4,2]
    for sz in [4,2]:
        if rngs[0].src[0].divides(sz) is not None:
            try: k.apply_opt(Opt(OptOps.LOCAL, k.rngs.index(rngs[0]), sz))
            except KernelOptError: continue
            break
    return k

def get_heaviest_kernel(expr, shape_a, shape_b):
    a = Tensor.rand(*shape_a)
    b = Tensor.rand(*shape_b) if shape_b else None
    ret = eval(expr)
    linear = ret.schedule_linear()
    sinks = [u for u in linear.toposort() if u.op == Ops.SINK]
    best_sink = max(sinks, key=lambda s: len([u for u in s.backward_slice if u.op == Ops.RANGE]))
    k = Scheduler(best_sink, ren)
    k.convert_loop_to_global()
    return k

SHAPES = [
    ('1024x1024',   'a @ b', (1024, 1024), (1024, 1024)),
    ('256x256',     'a @ b', (256, 256),   (256, 256)),
    ('2048x2048',   'a @ b', (2048, 2048), (2048, 2048)),
    ('16x4096',     'a @ b', (16, 4096),   (4096, 4096)),
    ('8x2048',      'a @ b', (8, 2048),    (2048, 2048)),
    ('512x2048',    'a @ b', (512, 2048),  (2048, 256)),
]

print(f"{'shape':14s} {'noopt':>10s} {'heuristic':>10s} {'theory':>10s} {'t/h':>6s}")
print('=' * 55)

for name, expr, sa, sb in SHAPES:
    k = get_heaviest_kernel(expr, sa, sb)
    noopt = time_kernel(k)
    heur = time_kernel(hand_coded_optimizations(k.copy()))
    theory = time_kernel(apply_adaptive_theory(k.copy()))
    ratio = theory / heur if heur > 0 and heur < float('inf') else 0
    print(f"{name:14s} {noopt:8.0f}us {heur:8.0f}us {theory:8.0f}us {ratio:5.2f}x")
```

## What to look for

- If theory/heuristic < 1.0 on all shapes → theory transfer works on CUDA too
- If theory/heuristic > 1.0 on some shapes → CUDA has different optimal strategy, note which
- If theory fails (inf) on some shapes → the adaptive divisor logic needs CUDA-specific handling

## Results

Record results in CUDA_THEORY_TRANSFER.md in this repo.
