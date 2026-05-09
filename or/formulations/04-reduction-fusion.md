# H5: Reduction Fusion as Algebraic Decomposition

## tinygrad problem

Softmax and layernorm decompose into 3 kernels (reduce → elementwise → reduce) because the scheduler splits at reduction boundaries. Naive fusion (PCONTIG=99) made them 1.8-1.9x SLOWER due to register pressure blowup. PyTorch's fused implementations are 2-6x faster.

The question is not "should we fuse?" but "under what algebraic conditions does fusion have O(1) auxiliary state?"

## OR formulation

**Decomposition of composite functions under resource constraints.**

Given a computation `f = h ∘ g ∘ r` where `r` is a reduction, `g` is elementwise, and `h` is another reduction:
- **Naive materialization:** compute `r(x)`, store intermediate, compute `g(r(x))`, store intermediate, compute `h(g(r(x)))`. Three kernels, two intermediate buffers.
- **Naive fusion:** compute everything in one pass, keeping all intermediates in registers. One kernel, but peak register pressure = O(reduction_dimension).
- **Algebraic fusion:** IF the composition admits a streaming decomposition, compute in one pass with O(1) auxiliary state. One kernel, constant register pressure.

The discriminant is whether the combining function has algebraic structure that enables incremental computation:

### Level 1: Homomorphism (Flashlight)

If the elementwise function `g` is a group homomorphism between the reduction operations, then:
```
h(g(r(x))) can be computed as: h'(r'(x)) with online correction
```
Example: `exp` is a homomorphism `(R,+) → (R⁺,×)`, so `sum(exp(x - max(x)))` can be computed in one pass with running `(max, sum)` state and dynamic rescaling.

This is **Benders decomposition** in disguise: the "complicating variable" (the running max) is handled by the master problem (rescaling), while the subproblems (per-element exp and accumulation) proceed independently.

### Level 2: Separable decomposition (RedFuser)

If `F(x,d) = G(x)·H(d)` (multiplicatively separable), track `G` and `H` independently and combine at the end. O(1) state per tracked quantity.

Example: `Var(X) = E[X²] - E[X]²` — track `sum_x` and `sum_x²` independently.

This is **Dantzig-Wolfe decomposition**: the linking constraint (variance depends on both first and second moments) is decomposed into independent subproblems.

### Level 3: Correction terms (Neptune)

If neither homomorphism nor separability holds, break the dependency and inject a computable correction:
```
h(g(r(x))) = h_approx(x) + correction(x)
```
This works when the correction is cheaper than full materialization.

This is **Lagrangian relaxation**: relax the complicating constraint, solve the relaxed problem, then fix violations.

### Level 4: Accept the split

When no algebraic trick applies, three kernels is the right answer.

## Proof manual lineage

```
Homomorphism (Flashlight)
  kill: combining function not a homomorphism
  └→ Separable decomposition (RedFuser)  [= Dantzig-Wolfe]
     kill: not separable
     └→ Algebraic correction (Neptune)   [= Lagrangian relaxation]
        kill: correction needs full intermediate
        └→ Accept the split               [= no decomposition exists]
```

Each escalation step is the proof manual's kill-condition-names-the-next-technique pattern.

## OR lineage the compiler papers don't cite

| Compiler technique | OR antecedent | Year |
|---|---|---|
| Online softmax / Flashlight | Benders decomposition (Benders, 1962) | 1962 |
| RedFuser separable decomposition | Dantzig-Wolfe decomposition (Dantzig & Wolfe, 1960) | 1960 |
| Neptune correction terms | Lagrangian relaxation (Fisher, 1981) | 1981 |
| SpaceFusion++ moderate fusion | Hypergraph partitioning (Kernighan-Lin, 1970) | 1970 |

## Proof manual validation

- Claim type: construction × algebraic
- Kill condition discriminates by algebraic structure of the combining function
- Each level has a computable test: is `g` a homomorphism? is `F` separable? is the correction bounded?
- Dependency: fused kernels need H1 (APRP-aware scheduling) to avoid register pressure blowup even with O(1) auxiliary state

## Implementation path

1. PatternMatcher rule detects reduce-elementwise-reduce chains
2. Test the elementwise function for homomorphism property (enumerable: exp, log, linear)
3. If homomorphism: apply Flashlight's three confluent rewrites
4. If separable: apply RedFuser's incremental computation transform
5. If neither: check if Neptune-style correction is cheaper than split → if yes, apply; if no, keep 3 kernels
6. All fused kernels pass through H1's APRP check before emission
