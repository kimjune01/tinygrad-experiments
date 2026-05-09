#!/usr/bin/env python3
"""Ground truth: the seven simple parser methods as they exist in tinygrad today.

Copied verbatim from tinygrad/nn/onnx.py lines 298-358 (commit 50aa70a25).
These are methods on OnnxPBParser — self.reader is a PBBufferedReader,
self._parse_message and self._decode_end_pos are inherited.
"""

# --- verbatim copies (indented as methods, shown flat for reference) ---

def _parse_TypeProto(self) -> dict:
  obj = {}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    match fid:
      case 1: obj["tensor_type"] = self._parse_TypeProtoTensor()
      case 4: obj["sequence_type"] = self._parse_TypeProtoWrapper()
      case 9: obj["optional_type"] = self._parse_TypeProtoWrapper()
      case _: self.reader.skip_field(wire_type)
  return obj

def _parse_TypeProtoTensor(self) -> dict:
  obj = {}
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
  obj = {"dim": []}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    match fid:
      case 1: obj["dim"].append(self._parse_TensorShapeProtoDimension())
      case _: self.reader.skip_field(wire_type)
  return obj

def _parse_TensorShapeProtoDimension(self) -> dict:
  obj = {}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    match fid:
      case 1: obj["dim_value"] = self.reader.read_int64()
      case 2: obj["dim_param"] = self.reader.read_string()
      case _: self.reader.skip_field(wire_type)
  return obj

def _parse_StringStringEntryProto(self) -> dict:
  obj = {}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    match fid:
      case 1: obj["key"] = self.reader.read_string()
      case 2: obj["value"] = self.reader.read_string()
      case _: self.reader.skip_field(wire_type)
  return obj

def _parse_OperatorSetIdProto(self) -> dict:
  obj = {}
  for fid, wire_type in self._parse_message(self._decode_end_pos()):
    match fid:
      case 1: obj["domain"] = self.reader.read_string()
      case 2: obj["version"] = self.reader.read_int64()
      case _: self.reader.skip_field(wire_type)
  return obj
