#!/usr/bin/env python3
"""Bench: measure line count before and after the refactor.

Uses sz.py's tokenized-line counting methodology to produce an apples-to-apples comparison.
Also counts raw lines for reference.
"""
import tokenize, token, sys, pathlib, tempfile, textwrap

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]
def is_docstring(t): return t.type == token.STRING and t.string.startswith('"""') and t.line.strip().startswith('"""')

def count_tokenized_lines(source_text):
  tokens = [t for t in tokenize.generate_tokens(iter(source_text.splitlines(True)).__next__) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
  return len({line for t in tokens for line in range(t.start[0], t.end[0]+1)})

def count_tokenized_lines_file(filepath):
  with tokenize.open(filepath) as f:
    tokens = [t for t in tokenize.generate_tokens(f.readline) if t.type in TOKEN_WHITELIST and not is_docstring(t)]
  return len({line for t in tokens for line in range(t.start[0], t.end[0]+1)})

# --- before: verbatim from onnx.py lines 298-358 ---
BEFORE = textwrap.dedent("""\
  def _parse_TypeProto(self) -> dict:
    obj: dict[str, Any] = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["tensor_type"] = self._parse_TypeProtoTensor()
        case 4: obj["sequence_type"] = self._parse_TypeProtoWrapper()
        case 9: obj["optional_type"] = self._parse_TypeProtoWrapper()
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_TypeProtoTensor(self) -> dict:
    obj: dict[str, Any] = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["elem_type"] = self.reader.read_int64()
        case 2: obj["shape"] = self._parse_TensorShapeProto()
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_TypeProtoWrapper(self) -> dict:
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["elem_type"] = self._parse_TypeProto()
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_TensorShapeProto(self) -> dict:
    obj: dict[str, Any] = {"dim": []}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["dim"].append(self._parse_TensorShapeProtoDimension())
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_TensorShapeProtoDimension(self) -> dict:
    obj: dict[str, Any] = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["dim_value"] = self.reader.read_int64()
        case 2: obj["dim_param"] = self.reader.read_string()
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_StringStringEntryProto(self) -> dict:
    obj: dict[str, Any] = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["key"] = self.reader.read_string()
        case 2: obj["value"] = self.reader.read_string()
        case _: self.reader.skip_field(wire_type)
    return obj

  def _parse_OperatorSetIdProto(self) -> dict:
    obj: dict[str, Any] = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["domain"] = self.reader.read_string()
        case 2: obj["version"] = self.reader.read_int64()
        case _: self.reader.skip_field(wire_type)
    return obj
""")

# --- after: generic parser + schema + wrappers ---
AFTER = textwrap.dedent("""\
  _SIMPLE_PROTOS: dict[str, tuple[dict, dict[int, tuple[str, str]]]] = {
    "TypeProto": ({}, {1: ("tensor_type", "_parse_TypeProtoTensor"), 4: ("sequence_type", "_parse_TypeProtoWrapper"),
                       9: ("optional_type", "_parse_TypeProtoWrapper")}),
    "TypeProtoTensor": ({}, {1: ("elem_type", "read_int64"), 2: ("shape", "_parse_TensorShapeProto")}),
    "TypeProtoWrapper": ({}, {1: ("elem_type", "_parse_TypeProto")}),
    "TensorShapeProto": ({"dim": []}, {1: ("dim", "_parse_TensorShapeProtoDimension")}),
    "TensorShapeProtoDimension": ({}, {1: ("dim_value", "read_int64"), 2: ("dim_param", "read_string")}),
    "StringStringEntryProto": ({}, {1: ("key", "read_string"), 2: ("value", "read_string")}),
    "OperatorSetIdProto": ({}, {1: ("domain", "read_string"), 2: ("version", "read_int64")}),
  }
  def _parse_proto(self, defaults, fields):
    obj: dict[str, Any] = {k: (list(v) if isinstance(v, list) else v) for k, v in defaults.items()}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      if fid not in fields: self.reader.skip_field(wire_type); continue
      name, action = fields[fid]
      value = getattr(self, action)() if action.startswith("_parse_") else getattr(self.reader, action)()
      if isinstance(obj.get(name), list): obj[name].append(value)
      else: obj[name] = value
    return obj
  def _parse_TypeProto(self): return self._parse_proto(*self._SIMPLE_PROTOS["TypeProto"])
  def _parse_TypeProtoTensor(self): return self._parse_proto(*self._SIMPLE_PROTOS["TypeProtoTensor"])
  def _parse_TypeProtoWrapper(self): return self._parse_proto(*self._SIMPLE_PROTOS["TypeProtoWrapper"])
  def _parse_TensorShapeProto(self): return self._parse_proto(*self._SIMPLE_PROTOS["TensorShapeProto"])
  def _parse_TensorShapeProtoDimension(self): return self._parse_proto(*self._SIMPLE_PROTOS["TensorShapeProtoDimension"])
  def _parse_StringStringEntryProto(self): return self._parse_proto(*self._SIMPLE_PROTOS["StringStringEntryProto"])
  def _parse_OperatorSetIdProto(self): return self._parse_proto(*self._SIMPLE_PROTOS["OperatorSetIdProto"])
""")

def main():
  before_raw = len([l for l in BEFORE.strip().splitlines() if l.strip()])
  after_raw = len([l for l in AFTER.strip().splitlines() if l.strip()])

  before_tok = count_tokenized_lines(BEFORE)
  after_tok = count_tokenized_lines(AFTER)

  # also measure the full file
  target = pathlib.Path(__file__).parent.parent / "tinygrad" / "tinygrad" / "nn" / "onnx.py"
  full_before = count_tokenized_lines_file(target) if target.exists() else "N/A"

  if target.exists():
    source = target.read_text()
    patched = source.replace(BEFORE.rstrip(), AFTER.rstrip())
    if patched == source:
      print("WARNING: could not find BEFORE block in target file (whitespace mismatch?)")
      print("Measuring snippet-level savings only.\n")
      full_after = "N/A"
    else:
      tmp = pathlib.Path(tempfile.mktemp(suffix=".py"))
      tmp.write_text(patched)
      full_after = count_tokenized_lines_file(tmp)
      tmp.unlink()
  else:
    full_after = "N/A"

  print("=== Line Count Benchmark ===\n")
  print(f"{'Metric':<30} {'Before':>8} {'After':>8} {'Delta':>8}")
  print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")
  print(f"{'Raw lines (snippet)':<30} {before_raw:>8} {after_raw:>8} {after_raw - before_raw:>+8}")
  print(f"{'Tokenized lines (snippet)':<30} {before_tok:>8} {after_tok:>8} {after_tok - before_tok:>+8}")
  if isinstance(full_before, int) and isinstance(full_after, int):
    print(f"{'Tokenized lines (full file)':<30} {full_before:>8} {full_after:>8} {full_after - full_before:>+8}")
    print(f"\n  sz.py total impact: {full_after - full_before:+d} lines")
  else:
    print(f"\n  Snippet-level savings: {after_tok - before_tok:+d} tokenized lines")

  return 0

if __name__ == "__main__":
  sys.exit(main())
