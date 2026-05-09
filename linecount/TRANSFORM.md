# Transformation Design: onnx.py protobuf parser refactor

## Files to touch

1. `tinygrad/nn/onnx.py` — the only file. All changes are within the `OnnxPBParser` class.

## Intervention

Replace lines 298-358 (seven `_parse_*` methods) with:

1. A class-level `_SIMPLE_PROTOS` dict defining the schema for each message type
2. A generic `_parse_proto(self, defaults, fields)` method (~8 lines)
3. Seven one-line wrappers that call `_parse_proto` with their schema

## What NOT to change

- `_parse_ModelProto`, `_parse_GraphProto`, `_parse_NodeProto`, `_parse_TensorProto`, `_parse_AttributeProto`, `_parse_ValueInfoProto` — these have complex post-processing logic (list coercion, parsed_node construction, external data loading) that doesn't fit the simple schema pattern
- `PBBufferedReader` — untouched
- The `OnnxRunner` class — untouched
- All op implementations — untouched

## Option considered and rejected

**Generate methods dynamically via metaclass/`__init_subclass__`**: would save ~7 more lines but loses grep-ability. Every call site does `self._parse_TypeProto()` — if the method doesn't exist in source, tooling breaks. Thin wrappers are the right tradeoff.

## Verification plan

1. `python validate.py` — mock-level equivalence across all test shapes
2. `python compat.py` — real ONNX model parse tree identity
3. `python bench.py` — tokenized line count delta
4. `cd ~/Documents/tinygrad && python -m pytest test/unit/test_onnx.py -x` — existing test suite
5. `cd ~/Documents/tinygrad && python sz.py .` — full sz.py measurement
