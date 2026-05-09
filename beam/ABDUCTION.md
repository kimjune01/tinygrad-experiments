# Abduction in this repo

## The problem

tinygrad's BEAM search optimizes GPU kernel schedules by trying combinations of compiler knobs (tile sizes, unroll factors, thread mappings, shared memory usage) and timing each candidate on real hardware. It keeps the fastest. This is brute-force search over a combinatorial space.

BEAM can perturb (apply a transformation) and observe (measure wall-clock time). It cannot explain. It sees that schedule A runs in 0.9ms and schedule B runs in 1.2ms but never asks why. Without a why, every untried combination is equally plausible, so the branching factor is the full cross-product of knobs.

## Three modes of reasoning

All reasoning decomposes into three primitives (Peirce, 1903):

- **Deduction**: given a theory, derive what must follow. "This kernel is memory-bound, so increasing ALU work per thread won't help."
- **Induction**: given observations, generalize a pattern. "Schedules with GROUP=2 are faster on average for this kernel shape."
- **Abduction**: given an observation, propose what matters. "The kernel stalled on memory — that's the bottleneck, not ALU."

These form a cycle:

```
         Theory
        ╱      ╲
abduction    deduction
     ╱            ╲
Observation ——→ Experiment
           induction
```

- **Abduction** (Observation → Theory): look at what happened, separate signal from noise, propose an explanation.
- **Deduction** (Theory → Experiment): given the explanation, derive which experiments would confirm or refute it.
- **Induction** (Experiment → Observation): run the experiments, accumulate evidence.

BEAM only runs the bottom edge. It never goes up to Theory.

## What abduction means here

Abduction is figure-ground separation on experimental results.

Given two kernel timings (before and after a transformation), the abductive step is: what *about* this transformation caused the difference? The answer is a theory — "memory-bound," "register pressure crossed an occupancy tier," "bank conflicts on shared memory." The theory is not the timing. The timing is the observation. The theory is what the observation means.

The primitive operation is **diff**: snapshot before, snapshot after, identify what changed (figure) and what didn't (ground).

- **Figure**: the transformation that moved the needle.
- **Ground**: everything else — the knobs that didn't matter, the hardware characteristics that stayed constant.

A system that performs this separation after each measurement can prune the search space. If the theory says "memory-bound," then UPCAST (more ALU per thread) is ground — skip it. Only test transformations the theory predicts will help: LOCAL (more threads to hide latency), GROUP (shared memory to reduce global traffic).

## The hypothesis graph

Each abductive step produces a theory. Each theory generates a small set of deduced experiments. Each experiment produces observations that feed back into the next abductive step. This forms a graph:

1. Observe: kernel has high memory stall cycles, low ALU utilization.
2. Abduce: kernel is memory-bound.
3. Deduce: GROUP should help (reduces global memory traffic). UPCAST won't (ALU isn't the bottleneck).
4. Experiment: test GROUP=2, GROUP=4. Skip UPCAST variations.
5. Observe: GROUP=2 improved 1.4x. GROUP=4 was neutral.
6. Abduce: shared memory bandwidth is now the bottleneck (GROUP=4 saturated it).
7. Deduce: XOR-swizzle on shared memory indices should help. Further GROUP increases won't.
8. Experiment: test swizzle.

Each cycle narrows. The branching factor at each step is 2–3 targeted experiments, not the full knob space.

The graph converges when no new theories are generated — the schedule is locally optimal and the system can explain why.

## Evidence classification

After each experiment, classify the trajectory (not just the endpoint):

| Response | Meaning | Next step |
|----------|---------|-----------|
| **Convergent** | The transformation was absorbed; performance barely changed | This knob doesn't matter for this kernel. Skip it. Test a different knob. |
| **Divergent** | Performance changed significantly in one direction | This knob is load-bearing. Follow its dependencies. |
| **Oscillatory** | Helps some cases, hurts others | Two constraints are fighting (e.g., register pressure vs. memory latency). Split the hypothesis. |
| **Chaotic** | Results are noisy and irreproducible | Measurement is unreliable at this kernel size, or too many interactions. Decompose differently. |

BEAM today treats all results as a flat ranking. The trajectory shape is thrown away.

## What this repo builds

An engine that closes the triangle for BEAM:

1. **Observe** what BEAM observes (kernel timings, and ideally hardware counters).
2. **Abduce** a theory about why the current schedule is slow (figure-ground separation on the observations).
3. **Deduce** which experiments to run next (targeted transformations, not exhaustive search).
4. **Repeat** until the graph converges.

The claim is that this finds better schedules in fewer trials than BEAM's brute-force search.

## Key terms

- **BEAM**: tinygrad's existing kernel schedule optimizer. Brute-force search over compiler knobs.
- **Abduction**: proposing an explanation for an observation. The missing step in BEAM.
- **Figure-ground separation**: identifying what matters (figure) vs. what doesn't (ground) in an observation.
- **Hypothesis graph**: the directed graph of theories and experiments. Nodes are observations; edges are hypotheses generated by classifying the observation's trajectory shape.
- **E-value trajectory**: evidence that accumulates over time and can be classified by shape (convergent, divergent, oscillatory, chaotic). Replaces single-number rankings.
- **Kill condition**: when an experiment disproves a theory. The failure mode names the next hypothesis — this is the edge-generation mechanism.
- **Theory transfer**: caching the theory ("memory-bound, GROUP helps") instead of just the winning schedule. Theories generalize across kernel shapes; specific schedules don't.
