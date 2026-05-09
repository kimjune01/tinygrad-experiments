# cython: language_level=3
"""Cython-compiled unified_rewrite — drop-in replacement for RewriteContext.unified_rewrite."""
import collections
from tinygrad.uop.ops import UOp, Ops, unwrap, BottomUpGate

cdef object SENTINEL = object()
cdef set CALL_OPS = {Ops.CALL, Ops.FUNCTION}

def cy_unified_rewrite(self, root):
    cdef dict replace = self.replace
    stack = collections.deque([(root, 0, root)])
    cdef set on_stack = {root}
    cdef dict waitlist = {}
    cdef int stage
    cdef tuple new_src
    cdef list tmp
    cdef bint enter_calls = self.enter_calls
    pm = self.pm
    bpm = self.bpm

    while stack:
        if len(stack) > 250000:
            raise RuntimeError("infinite loop in graph_rewrite (stack too big)")
        n, stage, new_n = stack.pop()
        if n in replace:
            continue
        if stage == 0:
            if bpm is not None:
                test_n = n
                seen = set()
                gate = False
                while test_n is not None:
                    if test_n in seen:
                        raise RuntimeError("infinite loop in fixed_point_rewrite")
                    seen.add(test_n)
                    new_n = test_n
                    try:
                        test_n = self.cached_bpm_rewrite(test_n)
                    except BottomUpGate:
                        replace[n] = unwrap(test_n)
                        if n in waitlist:
                            stack.extend(waitlist.pop(n))
                        gate = True
                        break
                if gate:
                    continue
            stack.append((n, 1, new_n))
            if not enter_calls and new_n.op in CALL_OPS:
                replace[new_n.src[0]] = new_n.src[0]
            for x in reversed(new_n.src):
                if x in on_stack:
                    continue
                stack.append((x, 0, x))
                on_stack.add(x)
        elif stage == 1:
            tmp = []
            broke = False
            for x in new_n.src:
                rx = replace.get(x, SENTINEL)
                if rx is SENTINEL:
                    waitlist.setdefault(x, []).append((n, 1, new_n))
                    broke = True
                    break
                tmp.append(rx)
            if broke:
                continue
            new_src = tuple(tmp)
            if new_src == new_n.src:
                if pm is None:
                    replace[n] = new_n
                    if n in waitlist:
                        stack.extend(waitlist.pop(n))
                    continue
                new_src_n = self.pm_rewrite(new_n)
                if new_src_n is None:
                    replace[n] = new_n
                    if n in waitlist:
                        stack.extend(waitlist.pop(n))
                    continue
            else:
                new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
            stack.append((n, 2, new_src_n))
            stack.append((new_src_n, 0, new_src_n))
        else:
            replaced_new_n = replace.get(new_n, SENTINEL)
            if replaced_new_n is SENTINEL:
                waitlist.setdefault(new_n, []).append((n, 2, new_n))
            else:
                replace[n] = replaced_new_n
                if n in waitlist:
                    stack.extend(waitlist.pop(n))
    return replace[root]
