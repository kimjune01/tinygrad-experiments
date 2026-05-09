"""
BEAM Search Baseline: measures search efficiency, not just kernel speed.

For each workload, runs BEAM with instrumentation to capture:
  - candidates generated, compiled, timed per round
  - best time per round and improvement trajectory
  - total search wall time
  - final quality vs heuristic

Usage:
  python3 bench_beam_baseline.py
  BEAM=4 python3 bench_beam_baseline.py
  WORKLOAD=gemm_1024 python3 bench_beam_baseline.py
"""
import os, sys, json, time, subprocess

BEAM_AMT = int(os.environ.get("BEAM", "2"))
CNT = int(os.environ.get("CNT", "6"))
SINGLE = os.environ.get("WORKLOAD", None)

WORKLOADS = [
    ("gemm_1024",  "a @ b",                        (1024, 1024), (1024, 1024)),
    ("gemm_256",   "a @ b",                        (256, 256),   (256, 256)),
    ("add_4096",   "a + b",                        (4096, 4096), (4096, 4096)),
    ("mul_sum",    "(a * b).sum()",                (4096, 4096), (4096, 4096)),
    ("relu_4096",  "a.relu()",                     (4096, 4096), None),
    ("exp_2048",   "a.exp()",                      (2048, 2048), None),
    ("sum_4096",   "a.sum()",                      (4096, 4096), None),
    ("permute",    "a.permute(1,0).contiguous()",  (1024, 1024), None),
    ("softmax",    "a.softmax(-1)",                (32, 2048),   None),
    ("layernorm",  "a.layernorm()",                (32, 128, 1024), None),
    ("matvec",     "a @ b",                        (4096,),      (4096, 4096)),
]

if SINGLE:
    WORKLOADS = [(n, e, sa, sb) for n, e, sa, sb in WORKLOADS if n == SINGLE]

# This script runs in a subprocess with tinygrad on the path.
# It instruments beam_search by replacing it with a version that logs rounds.
WORKER_SCRIPT = r'''
import os, sys, json, time, math, multiprocessing
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["IGNORE_BEAM_CACHE"] = "1"
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Context, getenv, prod, flatten, CACHELEVEL
from tinygrad.codegen.opt.search import (
    get_kernel_actions, _try_compile, _time_program,
    _ensure_buffer_alloc, _init_worker, beam_pool
)
from tinygrad.uop.ops import sym_infer

EXPR = """__EXPR__"""
SHAPE_A = __SHAPE_A__
SHAPE_B = __SHAPE_B__
BEAM_AMT = __BEAM_AMT__
CNT = __CNT__

# --- Phase 1: heuristic timing ---
a = Tensor.rand(*SHAPE_A)
b = Tensor.rand(*SHAPE_B) if SHAPE_B else None
for _ in range(3):
    ret = eval(EXPR)
    ret.realize()
    Device[Device.DEFAULT].synchronize()

heur_times = []
for _ in range(CNT):
    a = Tensor.rand(*SHAPE_A)
    if SHAPE_B: b = Tensor.rand(*SHAPE_B)
    a.realize()
    if b is not None: b.realize()
    Device[Device.DEFAULT].synchronize()
    GlobalCounters.reset()
    st = time.perf_counter()
    ret = eval(EXPR)
    ret.realize()
    Device[Device.DEFAULT].synchronize()
    heur_times.append((time.perf_counter() - st) * 1000)
heur_times.sort()
heur_trimmed = heur_times[1:-1] if len(heur_times) > 3 else heur_times
heur_ms = sum(heur_trimmed) / len(heur_trimmed)
heur_kernels = GlobalCounters.kernel_count

# Clear compile cache so instrumented BEAM runs fresh
from tinygrad.codegen import to_program_cache
to_program_cache.clear()

# --- Phase 2: instrumented BEAM search ---
# Must run before regular BEAM timing to avoid cache hits.

import tinygrad.codegen.opt.search as search_mod

original_beam_search = search_mod.beam_search
all_search_logs = []

def instrumented_beam_search(s, rawbufs, amt, allow_test_size=True, disable_cache=True):
    """Clone of beam_search that logs per-round stats."""
    pool = None
    default_parallel = multiprocessing.cpu_count() if s.ren.target.device in {"CUDA", "AMD", "NV", "METAL", "HIP"} else 0
    if (workers := getenv("PARALLEL", default_parallel)):
        pool = multiprocessing.get_context("spawn").Pool(workers, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))

    beam = [(s, float("inf"))]
    seen_libs = set()
    min_progress = getenv("BEAM_MIN_PROGRESS", 0.01) / 1e6
    rawbufs = _ensure_buffer_alloc(rawbufs)
    var_vals = {k.expr: int(k.vmax + k.vmin) // 2 for k in s.ast.variables()}
    dev = Device[s.ren.target.device]

    rounds = []
    search_st = time.perf_counter()
    exiting = False
    round_idx = 0

    while not exiting:
        round_st = time.perf_counter()
        candidates = flatten([get_kernel_actions(si, include_0=False).values() for si, _ in beam])
        n_candidates = len(candidates)

        timed = []
        compiled = 0
        least_compute_ops = math.inf

        mapper = map if pool is None else pool.imap_unordered
        for i, proc in mapper(_try_compile, enumerate(candidates)):
            if proc is None:
                continue
            prg, compile_et = proc
            compiled += 1
            if (lib := prg.src[4].arg) in seen_libs:
                continue
            estimates = prg.src[0].arg.estimates
            least_compute_ops = min(
                this_ops := sym_infer(estimates.ops if estimates is not None else 0, var_vals),
                least_compute_ops
            )
            if least_compute_ops * 1000 < this_ops:
                continue
            seen_libs.add(lib)
            try:
                tms = _time_program(
                    prg, var_vals, rawbufs,
                    early_stop=beam[0][1] * 3 if len(beam) else 1.0,
                    allow_test_size=allow_test_size,
                    clear_l2=hasattr(dev, 'invalidate_caches'),
                    dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1)
                )
            except Exception:
                continue
            timed.append((candidates[i], min(tms)))

        opts = sorted(timed, key=lambda x: x[1])
        best_prev = beam[0][1]
        exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1] - opts[0][1]) < min_progress))

        if not exiting:
            beam = opts[:amt]
        elif len(opts) > 0 and opts[0][1] < beam[0][1]:
            beam = opts[:1]

        rounds.append({
            "round": round_idx,
            "candidates": n_candidates,
            "compiled": compiled,
            "timed": len(timed),
            "best_us": beam[0][1] * 1e6 if beam[0][1] < float("inf") else None,
            "improved": beam[0][1] < best_prev,
            "wall_s": time.perf_counter() - round_st,
            "exiting": exiting,
        })
        round_idx += 1

    if pool is not None:
        pool.close()
        pool.join()

    log = {
        "rounds": rounds,
        "n_rounds": round_idx,
        "total_candidates": sum(r["candidates"] for r in rounds),
        "total_timed": sum(r["timed"] for r in rounds),
        "search_wall_s": time.perf_counter() - search_st,
        "final_us": beam[0][1] * 1e6 if beam[0][1] < float("inf") else None,
        "final_opts": [repr(o) for o in beam[0][0].applied_opts],
    }
    all_search_logs.append(log)

    if CACHELEVEL >= 1:
        from tinygrad.helpers import diskcache_put
        diskcache_put("beam_search", {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size,
                                       "device": s.ren.target.device, "suffix": s.ren.suffix}, beam[0][0].applied_opts)
    return beam[0][0]

# Patch and run instrumented search
search_mod.beam_search = instrumented_beam_search

with Context(BEAM=BEAM_AMT, IGNORE_BEAM_CACHE=1):
    a = Tensor.rand(*SHAPE_A)
    if SHAPE_B: b = Tensor.rand(*SHAPE_B)
    ret = eval(EXPR)
    ret.realize()
    Device[Device.DEFAULT].synchronize()

# Restore original and clear cache for clean BEAM timing
search_mod.beam_search = original_beam_search
to_program_cache.clear()

# --- Phase 3: BEAM execution timing (post-JIT, search cost excluded) ---
beam_times = []
with Context(BEAM=BEAM_AMT, IGNORE_BEAM_CACHE=1):
    for _ in range(3):
        a = Tensor.rand(*SHAPE_A)
        if SHAPE_B: b = Tensor.rand(*SHAPE_B)
        ret = eval(EXPR)
        ret.realize()
        Device[Device.DEFAULT].synchronize()

    for _ in range(CNT):
        a = Tensor.rand(*SHAPE_A)
        if SHAPE_B: b = Tensor.rand(*SHAPE_B)
        a.realize()
        if b is not None: b.realize()
        Device[Device.DEFAULT].synchronize()
        GlobalCounters.reset()
        st = time.perf_counter()
        ret = eval(EXPR)
        ret.realize()
        Device[Device.DEFAULT].synchronize()
        beam_times.append((time.perf_counter() - st) * 1000)
beam_times.sort()
beam_trimmed = beam_times[1:-1] if len(beam_times) > 3 else beam_times
beam_ms = sum(beam_trimmed) / len(beam_trimmed)

# --- Output ---
result = {
    "heur_ms": heur_ms,
    "heur_kernels": heur_kernels,
    "beam_ms": beam_ms,
    "beam_vs_heur": beam_ms / heur_ms if heur_ms > 0 else None,
    "search": all_search_logs,
}
print("RESULT:" + json.dumps(result))
'''


def run_workload(name, expr, shape_a, shape_b):
    script = WORKER_SCRIPT
    script = script.replace("__EXPR__", expr)
    script = script.replace("__SHAPE_A__", repr(shape_a))
    script = script.replace("__SHAPE_B__", repr(shape_b) if shape_b else "None")
    script = script.replace("__BEAM_AMT__", str(BEAM_AMT))
    script = script.replace("__CNT__", str(CNT))

    env = os.environ.copy()
    for k in ("BEAM", "NOOPT", "DEBUG", "IGNORE_BEAM_CACHE"):
        env.pop(k, None)
    env["IGNORE_BEAM_CACHE"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600, env=env
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        stderr_tail = proc.stderr.strip().split("\n")[-5:] if proc.stderr else []
        return {"error": "\n".join(stderr_tail)}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def fmt(us):
    if us is None: return "       —"
    if us < 1000: return f"{us:6.1f}us"
    return f"{us/1000:6.2f}ms"


def main():
    print(f"BEAM Baseline (BEAM={BEAM_AMT}, CNT={CNT})")
    print("=" * 100)
    print()

    all_results = {}

    for name, expr, shape_a, shape_b in WORKLOADS:
        print(f"  {name:14s} running ...", end="", flush=True)
        r = run_workload(name, expr, shape_a, shape_b)

        if "error" in r:
            print(f"\r  {name:14s} ERROR: {r['error'][:80]}")
            all_results[name] = r
            continue

        heur_ms = r["heur_ms"]
        beam_ms = r["beam_ms"]
        bvh = r.get("beam_vs_heur")
        searches = r.get("search", [])

        total_cands = sum(s.get("total_candidates", 0) for s in searches)
        total_timed = sum(s.get("total_timed", 0) for s in searches)
        total_rounds = sum(s.get("n_rounds", 0) for s in searches)
        total_wall = sum(s.get("search_wall_s", 0) for s in searches)
        n_kernels = len(searches)

        bvh_s = f"{bvh:.2f}x" if bvh else "—"

        print(f"\r  {name:14s}  heur={heur_ms:7.2f}ms  beam={beam_ms:7.2f}ms  b/h={bvh_s:>6s}  "
              f"kernels={n_kernels}  rounds={total_rounds}  "
              f"cands={total_cands}  timed={total_timed}  search={total_wall:.1f}s")

        for ki, s in enumerate(searches):
            for rd in s.get("rounds", []):
                print(f"    k{ki} r{rd['round']}: "
                      f"{rd['candidates']:>4d} cands → {rd['compiled']:>4d} compiled → {rd['timed']:>4d} timed  "
                      f"best={fmt(rd.get('best_us'))}  "
                      f"{'▲' if rd['improved'] else '—'}  "
                      f"wall={rd['wall_s']:.1f}s"
                      f"{'  EXIT' if rd['exiting'] else ''}")
            if s.get("final_opts"):
                print(f"    k{ki} final: {s['final_opts']}")

        all_results[name] = r
        print()

    # save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"beam_amt": BEAM_AMT, "cnt": CNT, "results": all_results}, f, indent=2)
    print(f"Raw data → {out_path}")

    # summary
    print()
    print(f"{'workload':14s} {'heur':>8s} {'beam':>8s} {'b/h':>6s} {'rnds':>5s} {'cands':>6s} {'timed':>6s} {'wall':>6s} {'yield':>6s}")
    print("-" * 75)
    for name, _, _, _ in WORKLOADS:
        r = all_results.get(name, {})
        if "error" in r:
            print(f"{name:14s} ERROR")
            continue
        searches = r.get("search", [])
        tc = sum(s.get("total_candidates", 0) for s in searches)
        tt = sum(s.get("total_timed", 0) for s in searches)
        tr = sum(s.get("n_rounds", 0) for s in searches)
        tw = sum(s.get("search_wall_s", 0) for s in searches)
        bvh = r.get("beam_vs_heur")
        bvh_s = f"{bvh:.2f}x" if bvh else "—"
        yld = f"{tt/tc:.0%}" if tc > 0 else "—"
        print(f"{name:14s} {r.get('heur_ms',0):7.2f}ms {r.get('beam_ms',0):7.2f}ms "
              f"{bvh_s:>6s} {tr:>5d} {tc:>6d} {tt:>6d} {tw:>5.1f}s {yld:>6s}")


if __name__ == "__main__":
    main()
