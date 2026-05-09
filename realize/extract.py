"""Extract: confirm that GGUF weights are lazy dequant chains, not realized buffers."""
import sys, os
os.environ.setdefault("DEV", "METAL")

from tinygrad.llm.model import Transformer
from tinygrad.helpers import fetch
from tinygrad import nn
from shapes import QUANT_MODELS

def extract(model_key="1b-q6k"):
    spec = QUANT_MODELS[model_key]
    print(f"Loading {model_key} ({spec['quant']})...")
    model, kv = Transformer.from_gguf(fetch(spec["url"]), 4096)
    params = nn.state.get_parameters(model)

    graph_sizes = [len(p.uop.toposort()) for p in params]
    realized = sum(1 for s in graph_sizes if s <= 5)
    lazy = sum(1 for s in graph_sizes if s > 5)

    print(f"\nParameters: {len(params)}")
    print(f"  Realized (graph_size <= 5): {realized}")
    print(f"  Lazy dequant chains (graph_size > 5): {lazy}")
    print(f"  UOp graph sizes: min={min(graph_sizes)}, max={max(graph_sizes)}, mean={sum(graph_sizes)/len(graph_sizes):.0f}")
    print(f"  Reported dtype: {set(p.dtype for p in params)}")
    print(f"  Total reported bytes: {sum(p.nbytes() for p in params) / 1e9:.2f} GB")

    if lazy > 0:
        print(f"\n  BUG CONFIRMED: {lazy}/{len(params)} parameters are lazy dequant chains.")
        print(f"  These will re-execute {max(graph_sizes)}-UOp dequant graphs on every forward pass.")
        return False
    else:
        print(f"\n  All parameters are realized. No lazy dequant chains found.")
        return True

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1b-q6k"
    ok = extract(key)
    sys.exit(0 if not ok else 1)  # exit 0 = bug confirmed (expected)
