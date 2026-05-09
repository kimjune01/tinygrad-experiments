# Integration Manifest

| Artifact | Location | Purpose |
|----------|----------|---------|
| Hypothesis graph | `HYPOTHESIS_GRAPH.md` | Full investigation trail with killed/confirmed nodes |
| Ground truth | `reference.py` | Verbatim copy of the seven methods from onnx.py |
| Candidate fix | `propose.py` | Data-driven generic parser + schema + wrappers |
| Mock validation | `validate.py` | Proves propose == reference on synthetic protobuf data |
| Compat check | `compat.py` | Proves parse tree identity on real ONNX models |
| Line count bench | `bench.py` | Tokenized line count before/after (sz.py methodology) |
| Test matrix | `shapes.py` | All message shapes: empty, full, nested, unknown fields |
| Extraction | `extract.py` | Derisk: confirms the duplication pattern exists in target |
| Transform doc | `TRANSFORM.md` | Maps fix onto target codebase, what to change and what not to |
