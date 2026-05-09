"""Reference: baseline performance with REALIZE=0 (default) and REALIZE=1."""
import sys, os, time
os.environ.setdefault("DEV", "METAL")

from tinygrad import Tensor, Device
from tinygrad.llm.model import Transformer
from tinygrad.helpers import fetch, GlobalCounters
from tinygrad import nn
from shapes import QUANT_MODELS, BENCHMARK_TOKENS, WARMUP_TOKENS

def benchmark(model_key="1b-q6k", realize=False):
    spec = QUANT_MODELS[model_key]
    print(f"{'='*60}")
    print(f"Model: {model_key} ({spec['quant']}), realize={realize}")
    print(f"{'='*60}")

    model, kv = Transformer.from_gguf(fetch(spec["url"]), 4096, realize=realize)

    params = nn.state.get_parameters(model)
    graph_sizes = [len(p.uop.toposort()) for p in params]
    print(f"UOp graph sizes after load: min={min(graph_sizes)}, max={max(graph_sizes)}")
    print(f"Total param bytes: {sum(p.nbytes() for p in params) / 1e9:.2f} GB")

    print(f"\nWarming up JIT ({WARMUP_TOKENS} tokens)...")
    gen = model.generate(toks:=[0])
    for _ in range(WARMUP_TOKENS):
        next(gen)

    print(f"Benchmarking ({BENCHMARK_TOKENS - WARMUP_TOKENS} tokens)...")
    times = []
    bws = []
    for i in range(BENCHMARK_TOKENS - WARMUP_TOKENS):
        GlobalCounters.reset()
        st = time.perf_counter()
        next(gen)
        Device[os.environ.get("DEV", "METAL")].synchronize()
        et = time.perf_counter()
        ms = (et - st) * 1000
        bw = GlobalCounters.global_mem / 1e9 / (et - st)
        times.append(ms)
        bws.append(bw)

    median_ms = sorted(times)[len(times) // 2]
    median_bw = sorted(bws)[len(bws) // 2]
    toks = 1000 / median_ms

    print(f"\nResults (median of {len(times)} tokens):")
    print(f"  {median_ms:.1f} ms/tok")
    print(f"  {toks:.1f} tok/s")
    print(f"  {median_bw:.1f} GB/s")
    return {"ms": median_ms, "toks": toks, "bw": median_bw, "model": model_key, "realize": realize}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1b-q6k"
    r0 = benchmark(key, realize=False)
    r1 = benchmark(key, realize=True)
    print(f"\n{'='*60}")
    print(f"Speedup: {r0['ms']/r1['ms']:.1f}x  ({r0['toks']:.1f} → {r1['toks']:.1f} tok/s)")
    print(f"Bandwidth: {r0['bw']:.0f} → {r1['bw']:.0f} GB/s")
