#!/usr/bin/env python3
"""Test matrix: all message shapes exercised by the seven simple parsers.

Covers: empty messages, single fields, all fields, unknown fields (skipped),
list-append semantics (TensorShapeProto), recursive nesting (TypeProto chain),
and edge cases (large varints, empty strings).
"""

SHAPES = {
  "OperatorSetIdProto": [
    {"desc": "empty", "fields": {}},
    {"desc": "domain only", "fields": {1: ("ai.onnx", "string")}},
    {"desc": "version only", "fields": {2: (13, "varint")}},
    {"desc": "both", "fields": {1: ("ai.onnx", "string"), 2: (17, "varint")}},
    {"desc": "unknown field skipped", "fields": {1: ("ai.onnx", "string"), 99: (0, "varint"), 2: (1, "varint")}},
    {"desc": "empty domain string", "fields": {1: ("", "string"), 2: (0, "varint")}},
  ],
  "StringStringEntryProto": [
    {"desc": "empty", "fields": {}},
    {"desc": "key-value", "fields": {1: ("location", "string"), 2: ("weights.bin", "string")}},
    {"desc": "unicode", "fields": {1: ("名前", "string"), 2: ("値", "string")}},
    {"desc": "key only", "fields": {1: ("orphan", "string")}},
  ],
  "TensorShapeProtoDimension": [
    {"desc": "empty", "fields": {}},
    {"desc": "dim_value", "fields": {1: (224, "varint")}},
    {"desc": "dim_param", "fields": {2: ("batch_size", "string")}},
    {"desc": "both (unusual but valid)", "fields": {1: (1, "varint"), 2: ("N", "string")}},
    {"desc": "large value", "fields": {1: (2**20, "varint")}},
  ],
  "TensorShapeProto": [
    {"desc": "empty (scalar)", "fields": {}, "expected_defaults": {"dim": []}},
    {"desc": "one dim", "repeat_field": (1, "TensorShapeProtoDimension", 1)},
    {"desc": "four dims (NCHW)", "repeat_field": (1, "TensorShapeProtoDimension", 4)},
  ],
  "TypeProtoTensor": [
    {"desc": "empty", "fields": {}},
    {"desc": "elem_type only", "fields": {1: (1, "varint")}},
    {"desc": "elem_type + shape", "fields": {1: (1, "varint")}, "submessage_field": (2, "TensorShapeProto")},
  ],
  "TypeProtoWrapper": [
    {"desc": "empty", "fields": {}},
    {"desc": "wraps TypeProto", "submessage_field": (1, "TypeProto")},
  ],
  "TypeProto": [
    {"desc": "empty", "fields": {}},
    {"desc": "tensor_type", "submessage_field": (1, "TypeProtoTensor")},
    {"desc": "sequence_type", "submessage_field": (4, "TypeProtoWrapper")},
    {"desc": "optional_type", "submessage_field": (9, "TypeProtoWrapper")},
  ],
}

if __name__ == "__main__":
  total = sum(len(cases) for cases in SHAPES.values())
  print(f"Test matrix: {len(SHAPES)} message types, {total} test cases")
  for msg_type, cases in SHAPES.items():
    print(f"\n  {msg_type}:")
    for case in cases:
      print(f"    - {case['desc']}")
