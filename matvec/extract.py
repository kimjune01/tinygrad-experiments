"""Extract tinygrad's actual loop ordering for each shape.

Runs a matmul through tinygrad's scheduler and reports:
  - RANGE node ordering (which axis is innermost)
  - stride pattern in the generated kernel
  - generated kernel source (for manual inspection)

Requires tinygrad to be importable.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/documents/tinygrad"))

from tinygrad import Tensor, Device
from tinygrad.helpers import DEBUG, Context, getenv
from tinygrad.codegen import full_rewrite_to_sink, do_to_program
from tinygrad.uop.ops import Ops, UOp
from shapes import SHAPES


def get_matmul_kernel_info(M, K, N):
    """Run a matmul through tinygrad's pipeline and extract the kernel structure."""

    A = Tensor.empty(M, K)
    B = Tensor.empty(K, N)
    C = A @ B

    # get the schedule without executing
    linear = C.schedule_linear()

    kernels = []
    for call in linear.src:
        ast = call.src[0]
        if ast.op not in (Ops.SINK, Ops.PROGRAM):
            continue

        # extract RANGE nodes from the AST to understand loop ordering
        ranges = []
        for u in ast.toposort():
            if u.op == Ops.RANGE:
                ranges.append({
                    "id": u.arg[0] if isinstance(u.arg, tuple) else u.arg,
                    "size": u.src[0].arg if u.src[0].op == Ops.CONST else str(u.src[0]),
                    "axis_type": str(u.arg[-1]) if isinstance(u.arg, tuple) and len(u.arg) > 1 else "unknown",
                })

        # get the program (compile the kernel)
        renderer = Device[Device.DEFAULT].renderer
        try:
            prg = do_to_program(ast, renderer)
            source = None
            for s in prg.src:
                if s.op == Ops.SOURCE:
                    source = s.arg
                    break
        except Exception as e:
            source = f"compilation failed: {e}"
            prg = None

        kernels.append({
            "ranges": ranges,
            "source": source,
            "n_ranges": len(ranges),
        })

    return kernels


def extract_loop_ordering(source):
    """Parse generated kernel source to find the innermost loop axis."""
    if source is None or "compilation failed" in str(source):
        return "unknown"

    lines = source.split('\n')
    loop_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('for ('):
            indent = len(line) - len(line.lstrip())
            loop_lines.append((indent, stripped))

    if not loop_lines:
        return "no loops (fully unrolled)"

    # innermost loop = deepest indent
    innermost = max(loop_lines, key=lambda x: x[0])
    return innermost[1][:60]


if __name__ == "__main__":
    print(f"Backend: {Device.DEFAULT}")
    print("=" * 100)
    print(f"{'Name':<22} {'Shape':<22} {'Kernels':>7} {'Ranges':>7} {'Innermost loop'}")
    print("=" * 100)

    for name, (M, K, N) in SHAPES.items():
        try:
            kernels = get_matmul_kernel_info(M, K, N)

            # find the main matmul kernel (most ranges = the reduce kernel)
            if kernels:
                main = max(kernels, key=lambda k: k["n_ranges"])
                inner = extract_loop_ordering(main["source"])
            else:
                inner = "no kernels"

            print(f"{name:<22} ({M:>5},{K:>5},{N:>5})  "
                  f"{len(kernels):>7} {main['n_ranges'] if kernels else 0:>7} "
                  f"{inner}")

            # dump full source for matvec cases
            if M == 1 and kernels:
                dump_path = f"kernels/{name}.metal"
                os.makedirs("kernels", exist_ok=True)
                with open(dump_path, "w") as f:
                    f.write(main["source"] or "no source")
                print(f"  -> dumped to {dump_path}")

        except Exception as e:
            print(f"{name:<22} ({M:>5},{K:>5},{N:>5})  ERROR: {e}")

    print()
    print("Compare innermost loop axis across matvec vs GEMM shapes.")
    print("If matvec has the reduction axis innermost with large strides, that's the bug.")
