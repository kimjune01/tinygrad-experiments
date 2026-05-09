# Transformation Design: Stride-Aware Loop Ordering in tinygrad

Maps the stride-cost function from `propose.py` onto tinygrad's scheduler.

## The change

When the scheduler orders RANGE nodes for a kernel, it should prefer innermost placement for ranges whose stride in the largest buffer is smallest. Currently, range ordering is determined by the scheduler's generic topological sort with no stride awareness. This produces correct results but walks the weight matrix in the wrong direction for matvec (stride-32768 inner loop instead of stride-4).

## Where it lives in tinygrad

### Current pipeline (what produces the bug)

```
Tensor ops → UOp graph → rangeify.py:get_kernel_graph()
  → rangeify + bufferize → RANGE nodes created
  → schedule/__init__.py:create_schedule() → topological sort of kernels
  → codegen/__init__.py:full_rewrite_to_sink() → apply_opts() → heuristic
  → codegen/late/linearizer.py:linearize() → instruction ordering
```

The RANGE ordering is set during `rangeify` (step 2) and never revisited for stride optimality. The `heuristic.py` matvec path (GROUP/LOCAL/UPCAST) operates on the ranges as-given — it tiles them but doesn't reorder them.

### Where to intervene

**Option A: In `rangeify.py`, when RANGE nodes are created.** The `add_ranges_to_store` function (line 19) creates ranges from shape dimensions. Currently ranges are created in shape order. If we instead sorted by stride cost before creating ranges, the downstream pipeline inherits the right ordering.

- File: `tinygrad/schedule/rangeify.py`
- Function: `add_ranges_to_store` (line 19)
- Current: `idxs = [UOp.range(r, next(ctx), AxisType.LOOP) for r in x.src[0].shape]`
- Change: sort ranges by stride cost in the largest input buffer before assigning loop nesting

**Option B: In `heuristic.py`, inside the matvec path.** The matvec heuristic at line 64-82 already pattern-matches on reduce ops. It could additionally reorder the reduction range to be outer and the output range to be inner.

- File: `tinygrad/codegen/opt/heuristic.py`
- Function: `hand_coded_optimizations` (line 8), matvec branch (line 64)
- Change: add a SWAP opt or equivalent to move the output axis innermost

**Option C: In `linearizer.py`, during instruction ordering.** The linearizer decides which loops are outermost. It could incorporate stride information.

- File: `tinygrad/codegen/late/linearizer.py`
- Change: stride-aware priority in the linearizer's topological sort

### Recommended: Option A

Option A is the most principled — it fixes the problem at the source (range creation) rather than patching it downstream. It matches the "derive, don't optimize" framing: the loop ordering falls out of the data layout, not from a special-case heuristic.

Option B is the safest — it only affects matvec-shaped kernels and doesn't touch the core range creation pipeline. But it's a special case, not a general fix.

Option C is too late — by the time the linearizer sees the ranges, the codegen has already made decisions based on the existing order.

## The stride-cost function

From `propose.py`, adapted for tinygrad's UOp ranges:

```python
def stride_cost(range_node, buffer_uops):
    """Compute the stride cost of making this range the innermost loop.
    Lower cost = should be more inner.
    """
    cost = 0
    for buf in buffer_uops:
        # find this range's stride in the buffer's index expression
        idx = buf.src[1].get_idx()
        if range_node not in idx.backward_slice:
            continue  # range doesn't appear in this buffer's index
        # extract the coefficient of range_node in the index expression
        stride = extract_stride(idx, range_node)  # needs implementation
        buffer_size = prod(buf.shape) * buf.dtype.itemsize
        cost += stride * (buffer_size / sum(prod(b.shape)*b.dtype.itemsize for b in buffer_uops))
    return cost
```

The `extract_stride` function walks the index expression to find the multiplicative coefficient of the range node. For a term like `range_node * 4096`, the stride is 4096.

## What changes in the generated kernel

### Before (current, from `kernels/matvec_attn_proj.metal`)

Inner loop: `Ridx0` (reduction), stride 32768 bytes in weight matrix
```c
for (int Ridx0 = 0; Ridx0 < 512; Ridx0++) {
    int alu5 = (alu0+(lidx0<<12)+(Ridx0<<15));  // stride 32768
    float val1 = (*(data2+alu5));
    float val2 = (*(data2+(alu5+4)));            // +4 = unit stride (output axis)
```

### After (proposed)

Inner loop: output axis, stride 4 bytes in weight matrix
Outer loop: reduction axis, stride 32768 bytes — but now amortized over N inner iterations

The exact generated code depends on how tinygrad's codegen reshapes the loop nest, but the access pattern should reverse: consecutive memory reads within the inner loop instead of 32KB jumps.

## What NOT to change

- **GEMM shapes.** For M,N >> 1, the current ordering works fine — GEMM uses tensor cores with tiled access patterns that don't depend on simple loop ordering. The stride-cost function naturally produces a reasonable ordering for GEMM (propose.py already validates this).
- **Elementwise ops.** Only reduce kernels have this issue. Elementwise ops have no reduction axis.
- **Multi-reduce kernels.** PCONTIG and compound ops need separate handling.

## Verification plan

1. Run `compat.py` after the change to verify numerical equivalence.
2. Run `extract.py` to confirm the generated Metal kernel has reversed loop ordering.
3. Run `validate.py` to confirm the stride-cost function matches BLAS reference for all shapes.
4. Run tinygrad's test suite (`pytest`) to catch regressions.
5. Benchmark matvec bandwidth: `python3 -m tinygrad.llm --benchmark` to measure tok/s improvement.

## Risk

The main risk is regressing non-matvec kernels. The stride-cost function should produce the same ordering as current for GEMM (both options are reasonable), but edge cases in the index expression parsing could produce unexpected orderings. The compatibility suite and test matrix in `validate.py` are designed to catch this.
