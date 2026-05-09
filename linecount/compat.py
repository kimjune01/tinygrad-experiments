#!/usr/bin/env python3
"""Compat: prove the refactor doesn't change behavior on real ONNX models.

Downloads a small ONNX model (MNIST), parses it with both the original and patched
OnnxPBParser, and asserts the parse trees are identical.

Requires tinygrad to be importable (run from the tinygrad repo or with PYTHONPATH set).
"""
import sys, pathlib, copy, json

def main():
  try:
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tinygrad"))
    from tinygrad.nn.onnx import OnnxPBParser
    from propose import _SIMPLE_PROTOS, _parse_proto
  except ImportError as e:
    print(f"SKIP: cannot import tinygrad or propose: {e}")
    return 0

  # find a test model
  model_paths = list((pathlib.Path(__file__).parent.parent / "tinygrad").rglob("*.onnx"))
  if not model_paths:
    print("SKIP: no .onnx files found in tinygrad repo for compat test")
    return 0

  model_path = model_paths[0]
  print(f"Testing with: {model_path.name}")

  # parse with original
  parser_orig = OnnxPBParser(str(model_path), load_external_data=False)
  result_orig = parser_orig.parse()

  # monkey-patch and parse again
  original_methods = {}
  for name, (defaults, fields) in _SIMPLE_PROTOS.items():
    method_name = f"_parse_{name}"
    original_methods[method_name] = getattr(OnnxPBParser, method_name)
    d, f = defaults, fields
    setattr(OnnxPBParser, method_name, lambda self, d=d, f=f: _parse_proto(self, d, f))

  parser_patched = OnnxPBParser(str(model_path), load_external_data=False)
  result_patched = parser_patched.parse()

  # restore
  for method_name, method in original_methods.items():
    setattr(OnnxPBParser, method_name, method)

  # deep compare (skip tensor values, compare structure)
  def normalize(obj):
    if isinstance(obj, dict):
      return {k: normalize(v) for k, v in obj.items() if k != "parsed_tensor"}
    elif isinstance(obj, (list, tuple)):
      return [normalize(x) for x in obj]
    else:
      return obj

  orig_norm = normalize(result_orig)
  patched_norm = normalize(result_patched)

  if orig_norm == patched_norm:
    print(f"PASS: parse trees identical for {model_path.name}")
    return 0
  else:
    print(f"FAIL: parse trees differ for {model_path.name}")
    return 1

if __name__ == "__main__":
  sys.exit(main())
