#!/usr/bin/env python3
"""Validate: propose matches reference for all test cases.

Mocks PBBufferedReader and OnnxPBParser, feeds identical byte sequences to both
the reference (verbatim) and proposed (data-driven) implementations, asserts identical output.
"""
import struct, io, sys

# --- mock infrastructure ---
class MockReader:
  def __init__(self, data: bytes):
    self.stream = io.BytesIO(data)
  def tell(self): return self.stream.tell()
  def seek(self, n, whence=0): self.stream.seek(n, whence)
  def read(self, n): return self.stream.read(n)
  def decode_varint(self):
    result, shift = 0, 0
    while True:
      b = self.stream.read(1)
      if not b: raise EOFError
      result |= (b[0] & 0x7F) << shift
      if not (b[0] & 0x80): return result
      shift += 7
  def read_string(self):
    length = self.decode_varint()
    return self.stream.read(length).decode("utf-8")
  def read_int64(self):
    v = self.decode_varint()
    if v >= (1 << 63): v -= (1 << 64)
    return v
  def skip_field(self, wire_type):
    if wire_type == 0: self.decode_varint()
    elif wire_type == 1: self.seek(8, 1)
    elif wire_type == 2: self.seek(self.decode_varint(), 1)
    elif wire_type == 5: self.seek(4, 1)

def encode_varint(value):
  if value < 0: value += (1 << 64)
  result = b""
  while value > 0x7F:
    result += bytes([value & 0x7F | 0x80])
    value >>= 7
  return result + bytes([value])

def encode_field(fid, wire_type, data):
  tag = encode_varint((fid << 3) | wire_type)
  if wire_type == 2:
    return tag + encode_varint(len(data)) + data
  elif wire_type == 0:
    return tag + encode_varint(data) if isinstance(data, int) else tag + data
  return tag + data

def encode_string_field(fid, value):
  encoded = value.encode("utf-8")
  return encode_field(fid, 2, encoded)

def encode_varint_field(fid, value):
  return encode_field(fid, 0, value)

def encode_submessage_field(fid, submessage_bytes):
  return encode_field(fid, 2, submessage_bytes)

# --- reference implementations (from reference.py, bound to mock) ---
class RefParser:
  def __init__(self, data):
    self.reader = MockReader(data)
    self._len = len(data)
  def _parse_message(self, end_pos):
    while self.reader.tell() < end_pos:
      tag = self.reader.decode_varint()
      yield tag >> 3, tag & 0x07
  def _decode_end_pos(self):
    str_len = self.reader.decode_varint()
    return self.reader.tell() + str_len

  def _parse_TypeProto(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["tensor_type"] = self._parse_TypeProtoTensor()
        case 4: obj["sequence_type"] = self._parse_TypeProtoWrapper()
        case 9: obj["optional_type"] = self._parse_TypeProtoWrapper()
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_TypeProtoTensor(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["elem_type"] = self.reader.read_int64()
        case 2: obj["shape"] = self._parse_TensorShapeProto()
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_TypeProtoWrapper(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["elem_type"] = self._parse_TypeProto()
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_TensorShapeProto(self):
    obj = {"dim": []}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["dim"].append(self._parse_TensorShapeProtoDimension())
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_TensorShapeProtoDimension(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["dim_value"] = self.reader.read_int64()
        case 2: obj["dim_param"] = self.reader.read_string()
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_StringStringEntryProto(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["key"] = self.reader.read_string()
        case 2: obj["value"] = self.reader.read_string()
        case _: self.reader.skip_field(wire_type)
    return obj
  def _parse_OperatorSetIdProto(self):
    obj = {}
    for fid, wire_type in self._parse_message(self._decode_end_pos()):
      match fid:
        case 1: obj["domain"] = self.reader.read_string()
        case 2: obj["version"] = self.reader.read_int64()
        case _: self.reader.skip_field(wire_type)
    return obj

# --- proposed implementations (from propose.py, bound to mock) ---
from propose import _SIMPLE_PROTOS, _parse_proto

class PropParser:
  def __init__(self, data):
    self.reader = MockReader(data)
    self._len = len(data)
  def _parse_message(self, end_pos):
    while self.reader.tell() < end_pos:
      tag = self.reader.decode_varint()
      yield tag >> 3, tag & 0x07
  def _decode_end_pos(self):
    str_len = self.reader.decode_varint()
    return self.reader.tell() + str_len
  def _parse_proto(self, defaults, fields): return _parse_proto(self, defaults, fields)
  def _parse_TypeProto(self): return self._parse_proto(*_SIMPLE_PROTOS["TypeProto"])
  def _parse_TypeProtoTensor(self): return self._parse_proto(*_SIMPLE_PROTOS["TypeProtoTensor"])
  def _parse_TypeProtoWrapper(self): return self._parse_proto(*_SIMPLE_PROTOS["TypeProtoWrapper"])
  def _parse_TensorShapeProto(self): return self._parse_proto(*_SIMPLE_PROTOS["TensorShapeProto"])
  def _parse_TensorShapeProtoDimension(self): return self._parse_proto(*_SIMPLE_PROTOS["TensorShapeProtoDimension"])
  def _parse_StringStringEntryProto(self): return self._parse_proto(*_SIMPLE_PROTOS["StringStringEntryProto"])
  def _parse_OperatorSetIdProto(self): return self._parse_proto(*_SIMPLE_PROTOS["OperatorSetIdProto"])

# --- test cases ---
def wrap_submessage(inner_bytes):
  return encode_varint(len(inner_bytes)) + inner_bytes

def test_OperatorSetIdProto():
  inner = encode_string_field(1, "ai.onnx") + encode_varint_field(2, 13)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_OperatorSetIdProto()
  prop = PropParser(data)._parse_OperatorSetIdProto()
  assert ref == prop == {"domain": "ai.onnx", "version": 13}, f"{ref} != {prop}"

def test_StringStringEntryProto():
  inner = encode_string_field(1, "location") + encode_string_field(2, "weights.bin")
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_StringStringEntryProto()
  prop = PropParser(data)._parse_StringStringEntryProto()
  assert ref == prop == {"key": "location", "value": "weights.bin"}, f"{ref} != {prop}"

def test_TensorShapeProtoDimension_value():
  inner = encode_varint_field(1, 224)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TensorShapeProtoDimension()
  prop = PropParser(data)._parse_TensorShapeProtoDimension()
  assert ref == prop == {"dim_value": 224}, f"{ref} != {prop}"

def test_TensorShapeProtoDimension_param():
  inner = encode_string_field(2, "batch_size")
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TensorShapeProtoDimension()
  prop = PropParser(data)._parse_TensorShapeProtoDimension()
  assert ref == prop == {"dim_param": "batch_size"}, f"{ref} != {prop}"

def test_TensorShapeProto_multiple_dims():
  dim1 = encode_varint_field(1, 3)
  dim2 = encode_string_field(2, "seq_len")
  inner = encode_submessage_field(1, dim1) + encode_submessage_field(1, dim2)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TensorShapeProto()
  prop = PropParser(data)._parse_TensorShapeProto()
  assert ref == prop, f"{ref} != {prop}"
  assert len(ref["dim"]) == 2

def test_TypeProtoTensor():
  shape_inner = encode_submessage_field(1, encode_varint_field(1, 64))
  inner = encode_varint_field(1, 1) + encode_submessage_field(2, shape_inner)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TypeProtoTensor()
  prop = PropParser(data)._parse_TypeProtoTensor()
  assert ref == prop, f"{ref} != {prop}"
  assert ref["elem_type"] == 1

def test_TypeProtoWrapper():
  type_tensor_inner = encode_varint_field(1, 7)
  type_proto_inner = encode_submessage_field(1, type_tensor_inner)
  inner = encode_submessage_field(1, type_proto_inner)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TypeProtoWrapper()
  prop = PropParser(data)._parse_TypeProtoWrapper()
  assert ref == prop, f"{ref} != {prop}"

def test_TypeProto_with_tensor():
  tensor_inner = encode_varint_field(1, 1)
  inner = encode_submessage_field(1, tensor_inner)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TypeProto()
  prop = PropParser(data)._parse_TypeProto()
  assert ref == prop, f"{ref} != {prop}"

def test_TypeProto_with_optional():
  tensor_inner = encode_varint_field(1, 10)
  type_proto_inner = encode_submessage_field(1, tensor_inner)
  wrapper_inner = encode_submessage_field(1, type_proto_inner)
  inner = encode_submessage_field(9, wrapper_inner)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_TypeProto()
  prop = PropParser(data)._parse_TypeProto()
  assert ref == prop, f"{ref} != {prop}"

def test_unknown_fields_skipped():
  inner = encode_string_field(1, "ai.onnx") + encode_varint_field(99, 42) + encode_varint_field(2, 17)
  data = wrap_submessage(inner)
  ref = RefParser(data)._parse_OperatorSetIdProto()
  prop = PropParser(data)._parse_OperatorSetIdProto()
  assert ref == prop == {"domain": "ai.onnx", "version": 17}, f"{ref} != {prop}"

def test_empty_message():
  data = wrap_submessage(b"")
  ref = RefParser(data)._parse_OperatorSetIdProto()
  prop = PropParser(data)._parse_OperatorSetIdProto()
  assert ref == prop == {}, f"{ref} != {prop}"

def test_empty_shape():
  data = wrap_submessage(b"")
  ref = RefParser(data)._parse_TensorShapeProto()
  prop = PropParser(data)._parse_TensorShapeProto()
  assert ref == prop == {"dim": []}, f"{ref} != {prop}"

def main():
  tests = [v for k, v in globals().items() if k.startswith("test_")]
  passed = 0
  for t in tests:
    try:
      t()
      print(f"  PASS: {t.__name__}")
      passed += 1
    except Exception as e:
      print(f"  FAIL: {t.__name__}: {e}")
  print(f"\n{passed}/{len(tests)} tests passed")
  return 0 if passed == len(tests) else 1

if __name__ == "__main__":
  sys.exit(main())
