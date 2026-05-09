"""Test matrix covering target shapes AND regression shapes."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documents/tinygrad"))

WORKLOADS = {
    # Small / scheduling-dominated
    "softmax_32x128": lambda: _softmax(32, 128),
    "softmax_1x16": lambda: _softmax(1, 16),

    # Matmul variants
    "matmul_64x64": lambda: _matmul(64, 64, 64),
    "matmul_256x256": lambda: _matmul(256, 256, 256),

    # Conv (ResNet building block)
    "conv_relu": lambda: _conv_relu(1, 64, 56, 56, 64, 3),
    "conv_relu_small": lambda: _conv_relu(1, 16, 8, 8, 16, 3),

    # Deep (scaling test)
    "deep_conv_2": lambda: _deep_conv(2),
    "deep_conv_4": lambda: _deep_conv(4),
    "deep_conv_8": lambda: _deep_conv(8),

    # Transformer attention
    "transformer_1x32x128": lambda: _transformer(1, 32, 128),

    # Reduction
    "reduce_sum": lambda: _reduce_sum(64, 256),
    "reduce_max": lambda: _reduce_max(64, 256),

    # Element-wise (should be trivial)
    "elementwise_add": lambda: _elementwise_add(1024),
}


def _softmax(B, C):
    from tinygrad import Tensor
    Tensor.randn(B, C).softmax().realize()

def _matmul(M, K, N):
    from tinygrad import Tensor
    (Tensor.randn(M, K) @ Tensor.randn(K, N)).realize()

def _conv_relu(B, Ci, H, W, Co, K):
    from tinygrad import Tensor
    x = Tensor.randn(B, Ci, H, W)
    w = Tensor.randn(Co, Ci, K, K)
    x.conv2d(w, padding=K // 2).relu().realize()

def _deep_conv(n_layers):
    from tinygrad import Tensor
    x = Tensor.randn(1, 64, 56, 56)
    for _ in range(n_layers):
        w = Tensor.randn(64, 64, 3, 3)
        x = x.conv2d(w, padding=1).relu()
    x.realize()

def _transformer(B, T, C):
    from tinygrad import Tensor
    x = Tensor.randn(B, T, C)
    q = x @ Tensor.randn(C, C)
    k = x @ Tensor.randn(C, C)
    v = x @ Tensor.randn(C, C)
    attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
    (attn.softmax() @ v).realize()

def _reduce_sum(B, C):
    from tinygrad import Tensor
    Tensor.randn(B, C).sum(axis=-1).realize()

def _reduce_max(B, C):
    from tinygrad import Tensor
    Tensor.randn(B, C).max(axis=-1).realize()

def _elementwise_add(N):
    from tinygrad import Tensor
    (Tensor.randn(N) + Tensor.randn(N)).realize()
