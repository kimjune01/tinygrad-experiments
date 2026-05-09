"""Bench: measure the performance delta between lazy and the refined hack."""
import sys, os, time
os.environ.setdefault("DEV", "METAL")

from tinygrad import Tensor, Device, TinyJit
from tinygrad.llm.model import Transformer
from tinygrad.helpers import fetch, GlobalCounters
from tinygrad import nn
from shapes import QUANT_MODELS, BENCHMARK_TOKENS, WARMUP_TOKENS

def apply_hack(model):
    params = nn.state.get_parameters(model)
    for p in params:
        p.replace(p.contiguous())
    model.rollout_jit = TinyJit(model.forward, prune=True)

def bench_one(model_key, use_hack):
    spec = QUANT_MODELS[model_key]
    model, kv = Transformer.from_gguf(fetch(spec["url"]), 4096, realize=False)
    if use_hack:
        apply_hack(model)

    gen = model.generate(toks:=[0])
    for _ in range(WARMUP_TOKENS):
        next(gen)

    times = []
    dev = os.environ.get("DEV", "METAL")
    for _ in range(BENCHMARK_TOKENS - WARMUP_TOKENS):
        GlobalCounters.reset()
        st = time.perf_counter()
        next(gen)
        Device[dev].synchronize()
        times.append((time.perf_counter() - st) * 1000)

    median_ms = sorted(times)[len(times) // 2]
    return {"ms": median_ms, "toks": 1000 / median_ms}

def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "1b-q6k"
    spec = QUANT_MODELS[key]

    print(f"Benchmarking {key} ({spec['quant']})...")
    print(f"\n--- lazy (default) ---")
    r0 = bench_one(key, use_hack=False)
    print(f"  {r0['ms']:.1f} ms/tok, {r0['toks']:.1f} tok/s")

    print(f"\n--- hack (contiguous + rollout prune) ---")
    r1 = bench_one(key, use_hack=True)
    print(f"  {r1['ms']:.1f} ms/tok, {r1['toks']:.1f} tok/s")

    speedup = r0["ms"] / r1["ms"]
    print(f"\n--- Summary ---")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  tok/s:  {r0['toks']:.1f} -> {r1['toks']:.1f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
