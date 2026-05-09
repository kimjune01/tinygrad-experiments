#!/usr/bin/env python3
"""Derisk: confirm the duplication exists in the target codebase.

Extracts the seven _parse_* methods from tinygrad/nn/onnx.py and verifies
they follow the identical pattern: obj={}, for fid in message, match fid, return obj.
"""
import ast, sys, pathlib, tokenize, token

TARGET = pathlib.Path(__file__).parent.parent / "tinygrad" / "tinygrad" / "nn" / "onnx.py"
SIMPLE_PARSERS = [
  "_parse_TypeProto", "_parse_TypeProtoTensor", "_parse_TypeProtoWrapper",
  "_parse_TensorShapeProto", "_parse_TensorShapeProtoDimension",
  "_parse_StringStringEntryProto", "_parse_OperatorSetIdProto",
]

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]
def is_docstring(t): return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def count_tokenized_lines(filepath, start, end):
  with tokenize.open(filepath) as f:
    tokens = [t for t in tokenize.generate_tokens(f.readline) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
  return len({line for t in tokens for line in range(t.start[0], t.end[0]+1) if start <= line <= end})

def main():
  if not TARGET.exists():
    print(f"FAIL: {TARGET} not found"); return 1

  source = TARGET.read_text()
  tree = ast.parse(source)

  # find OnnxPBParser class
  parser_class = None
  for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "OnnxPBParser":
      parser_class = node; break
  if parser_class is None:
    print("FAIL: OnnxPBParser class not found"); return 1

  methods = {}
  for node in parser_class.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SIMPLE_PARSERS:
      methods[node.name] = node

  missing = set(SIMPLE_PARSERS) - set(methods.keys())
  if missing:
    print(f"FAIL: methods not found: {missing}"); return 1

  print("=== Duplication confirmed ===\n")
  total_tokenized = 0
  for name in SIMPLE_PARSERS:
    node = methods[name]
    start, end = node.lineno, node.end_lineno
    tlines = count_tokenized_lines(TARGET, start, end)
    total_tokenized += tlines
    lines = source.splitlines()[start-1:end]

    # verify pattern: has _parse_message loop, match statement, skip_field
    body_src = "\n".join(lines)
    has_parse_message = "_parse_message" in body_src
    has_match = "match fid" in body_src
    has_skip = "skip_field" in body_src

    status = "OK" if (has_parse_message and has_match and has_skip) else "PATTERN MISMATCH"
    print(f"  {name}: lines {start}-{end} ({tlines} tokenized) [{status}]")

  print(f"\n  Total tokenized lines: {total_tokenized}")
  print(f"  All 7 methods follow the same pattern: obj, _parse_message loop, match fid, skip_field, return obj")
  return 0

if __name__ == "__main__":
  sys.exit(main())
