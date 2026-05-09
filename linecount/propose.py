#!/usr/bin/env python3
"""Candidate fix: data-driven generic parser replacing seven boilerplate methods.

Each message schema is a tuple: (defaults_dict, {fid: (field_name, action_string)}).
action_string is either a reader method ("read_string", "read_int64") or a parser method ("_parse_X").
If defaults[field_name] is a list, the action appends instead of assigning.
"""
from typing import Any

# --- schema definitions ---
# (defaults, {fid: (field_name, action)})

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

def _parse_proto(self, defaults: dict, fields: dict[int, tuple[str, str]]) -> dict:
  """Generic protobuf message parser. Dispatches to reader methods or recursive parsers."""
  obj: dict[str, Any] = {k: (list(v) if isinstance(v, list) else v) for k, v in defaults.items()}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    if fid not in fields:
      self.reader.skip_field(wire_type)
      continue
    name, action = fields[fid]
    value = getattr(self, action)() if action.startswith("_parse_") else getattr(self.reader, action)()
    if isinstance(obj.get(name), list): obj[name].append(value)
    else: obj[name] = value
  return obj

# --- thin wrappers (one line each, preserve call sites) ---
def _parse_TypeProto(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["TypeProto"])
def _parse_TypeProtoTensor(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["TypeProtoTensor"])
def _parse_TypeProtoWrapper(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["TypeProtoWrapper"])
def _parse_TensorShapeProto(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["TensorShapeProto"])
def _parse_TensorShapeProtoDimension(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["TensorShapeProtoDimension"])
def _parse_StringStringEntryProto(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["StringStringEntryProto"])
def _parse_OperatorSetIdProto(self) -> dict: return self._parse_proto(*_SIMPLE_PROTOS["OperatorSetIdProto"])
