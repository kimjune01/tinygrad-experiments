"""Test matrix: GGUF quantization types × model sizes."""

QUANT_MODELS = {
    "1b-q6k": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
        "quant": "Q6_K",
        "params": 1.24e9,
        "gguf_bytes": 967e6,
        "f16_bytes": 2.48e9,
    },
    "3b-f16": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
        "quant": "F16",
        "params": 3.21e9,
        "gguf_bytes": 5.98e9,
        "f16_bytes": 5.98e9,
    },
    "1b-q4km": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "params": 1.24e9,
        "gguf_bytes": 760e6,
        "f16_bytes": 2.48e9,
    },
}

BENCHMARK_TOKENS = 15
WARMUP_TOKENS = 4  # first N tokens are JIT warmup, excluded from measurement
