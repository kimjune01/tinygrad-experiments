"""Compat: verify that the refined hack produces identical outputs across architectures."""
import sys, os
os.environ.setdefault("DEV", "METAL")

from tinygrad import Tensor, Device, TinyJit
from tinygrad.llm.model import Transformer
from tinygrad.helpers import fetch
from tinygrad import nn
from shapes import QUANT_MODELS

ARCH_MODELS = {
    "llama-q6k": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
    "llama-q4km": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "moe-olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
    "ssm-qwen35": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
}

def apply_hack(model):
    params = nn.state.get_parameters(model)
    for p in params:
        p.replace(p.contiguous())
    model.rollout_jit = TinyJit(model.forward, prune=True)

def generate_tokens(model, prompt, n, seed=42):
    Tensor.manual_seed(seed)
    tokens = list(prompt)
    gen = model.generate(list(prompt), temperature=0.0)
    for _ in range(n):
        tokens.append(int(next(gen)))
    return tokens

def check_one(name, url, num_tokens=15):
    print(f"\n--- {name} ---")

    # Seeded lazy
    m0, _ = Transformer.from_gguf(fetch(url), 4096, realize=False)
    t0 = generate_tokens(m0, [0], num_tokens)
    del m0

    # Seeded lazy again (determinism check)
    m1, _ = Transformer.from_gguf(fetch(url), 4096, realize=False)
    t1 = generate_tokens(m1, [0], num_tokens)
    del m1

    if t0 != t1:
        print(f"  SKIP: baseline non-deterministic even with seed")
        return None

    # Seeded hack
    m2, _ = Transformer.from_gguf(fetch(url), 4096, realize=False)
    apply_hack(m2)
    t2 = generate_tokens(m2, [0], num_tokens)

    match = t0 == t2
    print(f"  Single-turn ({num_tokens} tok): {'PASS' if match else 'FAIL'}")
    if not match:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(t0, t2)) if a != b]
        print(f"  First diff: {diffs[0]}")
        return False

    # Multi-turn: divergent prompt on same model instance
    t3_lazy = generate_tokens(m1 if m1 else Transformer.from_gguf(fetch(url), 4096, realize=False)[0], [1, 2, 3], num_tokens)
    # For hack, reuse m2 which has state from prior generation
    t3_hack = list([1, 2, 3])
    Tensor.manual_seed(42)
    g3 = m2.generate([1, 2, 3], temperature=0.0)
    for _ in range(num_tokens):
        t3_hack.append(int(next(g3)))

    # Fresh lazy for ground truth on divergent prompt
    m3, _ = Transformer.from_gguf(fetch(url), 4096, realize=False)
    t3_fresh = generate_tokens(m3, [1, 2, 3], num_tokens)

    multi_match = t3_hack == t3_fresh
    print(f"  Multi-turn divergent ({num_tokens} tok): {'PASS' if multi_match else 'FAIL'}")
    if not multi_match:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(t3_fresh, t3_hack)) if a != b]
        print(f"  First diff: {diffs[0] if diffs else 'length mismatch'}")

    return match and multi_match

def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    models = {key: ARCH_MODELS[key]} if key else ARCH_MODELS
    results = {}
    for name, url in models.items():
        results[name] = check_one(name, url)

    print(f"\n{'='*40}")
    print("Results:")
    for name, result in results.items():
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        print(f"  {name}: {status}")
    all_tested = [v for v in results.values() if v is not None]
    overall = all(all_tested) if all_tested else False
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1

if __name__ == "__main__":
    sys.exit(main())
