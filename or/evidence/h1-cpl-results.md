# H1.1a: CPL Scheduling Results

Date: 2026-05-07
Branch: `or-baseline` + CPL patch in `linearizer.py`
Platform: arm64, macOS 26.4.1, Apple Silicon (Metal)

## Change

Replaced flat `{LOAD:-1, ALU:0, STORE:+1}` priority with Critical Path Length (CPL).
Each computation UOp gets `priority = -cpl[u]` where `cpl[u] = latency(u) + max(cpl(consumers))`.
LOAD latency = 10, everything else = 1. ~15 lines added.

## Definitive evidence: kernel code diff

### matvec (r_64_8_4_16_512)

**Baseline inner loop:**
```metal
for (int Ridx0 = 0; Ridx0 < 512; Ridx0++) {
    float val0 = (*(data1_4096+(lidx0+(Ridx0<<3))));       // LOAD vector
    int alu17 = (alu0+(lidx0<<12)+(Ridx0<<15));             // INDEX matrix
    float val1 = (*(data2_16777216+alu17));                  // LOAD matrix
```

**CPL inner loop:**
```metal
for (int Ridx0 = 0; Ridx0 < 512; Ridx0++) {
    int alu17 = (alu0+(lidx0<<12)+(Ridx0<<15));             // INDEX matrix (moved up)
    float val0 = (*(data1_4096+(lidx0+(Ridx0<<3))));       // LOAD vector
    float val1 = (*(data2_16777216+alu17));                  // LOAD matrix
```

One instruction moved. Kernel time: 957us → 734us (**-23%**).

CPL moved the matrix index computation before the vector load, allowing matrix loads
to issue immediately after their address is computed. Software pipelining effect.

### mul_sum (r_64_8_4_16_512)

CPL grouped all index computations before all loads (baseline interleaved them).
Kernel time: 1571us → 1570us (**0%**). Metal hardware reorders at pipeline level regardless.

## Benchmark-level evidence (unreliable)

bench_perceive has ±20-30% variance on Apple Silicon. Three runs of the SAME code
produced gemm_1024 v_torch of 0.73x, 1.04x, and 1.68x. The metric cannot measure
a 5-15% scheduling effect.

## Scheduling overhead discovery

Softmax (32, 2048) timing breakdown:
- GPU kernel time: 42us (3 kernels: 15 + 18 + 9)
- Total benchmark time: 2243us
- Python scheduling overhead: 2202us (98%)

The "fusion gap" in bench_perceive is scheduling overhead, not kernel quality.

## Regression check

- `test/backend/test_linearizer.py`: 24 passed, 3 skipped, 1 xfailed
- Numerical correctness: GEMM, sum, softmax, layernorm all verified

## Verdict

**H1.1a CONFIRMED.** CPL scheduling produces a genuine 23% speedup on matvec via
software pipelining (INDEX before LOAD). Neutral on all other workloads at the
kernel level. No regressions.

The effect is narrow: CPL only helps when the DAG has independent load chains
that benefit from reordering index computations ahead of data loads. For most
tinygrad kernels, the flat heuristic already produces a reasonable order and
CPL matches it.
