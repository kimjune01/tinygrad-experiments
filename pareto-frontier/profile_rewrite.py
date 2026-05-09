"""Profile where PatternMatcher.rewrite time actually goes.

Key findings:
- 80% of time is in SUCCESSFUL matches (avg 23us each — the callbacks DO work)
- 20% in failed matches (avg 932ns — cheap structural mismatch)
- Empty calls (no patterns for op) cost 0% of time
- Average winner position: 1.5 (65% of wins have failed matches before them)
- 86% of (op, src0_op) slots are DETERMINISTIC: same pattern always wins
- 53% of wins could be precomputed to position 0

The dispatch optimization payoff: not faster lookup, but ROUTING to the winning
pattern. For deterministic slots (86%), the prework pins the winner as the only
candidate. Failed matches before the winner are eliminated.
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from collections import defaultdict, Counter
from tinygrad.uop.ops import PatternMatcher, UOp, Ops

_orig = PatternMatcher.rewrite


def profile_match_costs(fn, label=""):
    """Break down rewrite time into successful vs failed match costs."""
    success_time, fail_time = [0.0], [0.0]
    success_count, fail_count = [0], [0]

    def profiled(self, uop, ctx=None):
        if len(pats := self.pdict.get(uop.op, [])):
            if (ler := uop.__dict__.get('_src_ops')) is None:
                uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
            for _, match, early_reject in pats:
                if not early_reject.issubset(ler): continue
                start = time.perf_counter()
                ret = match(uop, ctx)
                elapsed = time.perf_counter() - start
                if ret is not None and ret is not uop:
                    success_time[0] += elapsed
                    success_count[0] += 1
                    return ret
                fail_time[0] += elapsed
                fail_count[0] += 1
        return None

    PatternMatcher.rewrite = profiled
    fn()  # warmup
    success_time[0] = fail_time[0] = 0.0
    success_count[0] = fail_count[0] = 0
    fn()  # measure
    PatternMatcher.rewrite = _orig

    total = (success_time[0] + fail_time[0]) * 1e6
    print(f"\n{label} match cost breakdown:")
    print(f"  Successful: {success_count[0]:5d} calls, {success_time[0]*1e6:8.0f}us "
          f"({success_time[0]/(success_time[0]+fail_time[0]+1e-12)*100:.0f}%), "
          f"avg {success_time[0]*1e9/max(1,success_count[0]):.0f}ns")
    print(f"  Failed:     {fail_count[0]:5d} calls, {fail_time[0]*1e6:8.0f}us "
          f"({fail_time[0]/(success_time[0]+fail_time[0]+1e-12)*100:.0f}%), "
          f"avg {fail_time[0]*1e9/max(1,fail_count[0]):.0f}ns")
    return success_count[0], fail_count[0]


def profile_winner_positions(fn, label=""):
    """Track where in the pattern list the winning match occurs."""
    winners = []

    def track(self, uop, ctx=None):
        if len(pats := self.pdict.get(uop.op, [])):
            if (ler := uop.__dict__.get('_src_ops')) is None:
                uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
            for i, (_, match, early_reject) in enumerate(pats):
                if not early_reject.issubset(ler): continue
                if (ret := match(uop, ctx)) is not None and ret is not uop:
                    winners.append((uop.op, uop.src[0].op if uop.src else None, i, len(pats)))
                    return ret
        return None

    PatternMatcher.rewrite = track
    fn()  # warmup
    winners.clear()
    fn()  # measure
    PatternMatcher.rewrite = _orig

    total = len(winners)
    avg_pos = sum(pos for _, _, pos, _ in winners) / max(1, total)
    at_zero = sum(1 for _, _, pos, _ in winners if pos == 0)
    skipped = sum(pos for _, _, pos, _ in winners)

    print(f"\n{label} winner position analysis:")
    print(f"  Total wins: {total}")
    print(f"  Average position: {avg_pos:.1f}")
    print(f"  Already first: {at_zero} ({at_zero/total*100:.0f}%)")
    print(f"  Failed matches skippable: {skipped} (~{skipped*932//1000}us)")
    return winners


def profile_slot_determinism(fn, label=""):
    """Check if (op, src0_op) slots always produce the same winner."""
    slot_winners = defaultdict(list)

    def track(self, uop, ctx=None):
        if len(pats := self.pdict.get(uop.op, [])):
            if (ler := uop.__dict__.get('_src_ops')) is None:
                uop.__dict__['_src_ops'] = ler = {u.op for u in uop.src}
            for i, (_, match, early_reject) in enumerate(pats):
                if not early_reject.issubset(ler): continue
                if (ret := match(uop, ctx)) is not None and ret is not uop:
                    src0 = uop.src[0].op if uop.src else None
                    slot_winners[(id(self), uop.op, src0)].append(i)
                    return ret
        return None

    PatternMatcher.rewrite = track
    fn()  # warmup
    slot_winners.clear()
    fn()  # measure
    PatternMatcher.rewrite = _orig

    total = sum(len(v) for v in slot_winners.values())
    det = sum(len(v) for v in slot_winners.values() if len(set(v)) == 1)
    det_first = sum(len(v) for v in slot_winners.values() if len(set(v)) == 1 and v[0] == 0)

    print(f"\n{label} slot determinism:")
    print(f"  Total winning calls: {total}")
    print(f"  Deterministic (same winner): {det} ({det/total*100:.0f}%)")
    print(f"    Already at position 0: {det_first} ({det_first/total*100:.0f}%)")
    print(f"    Precomputable to pos 0: {det - det_first} ({(det-det_first)/total*100:.0f}%)")
    print(f"  Variable (different winners): {total - det} ({(total-det)/total*100:.0f}%)")


if __name__ == "__main__":
    from shapes import WORKLOADS

    for name in ["softmax_32x128", "conv_relu", "transformer_1x32x128"]:
        fn = WORKLOADS[name]
        profile_match_costs(fn, name)
        profile_winner_positions(fn, name)
        profile_slot_determinism(fn, name)
