# OR Theory Cross-Reference

Maps each tinygrad optimization problem to its OR lineage: the original result, the textbook treatment, and what the compiler community derived independently (often weaker).

---

## Scheduling

### Critical Path Method (CPM)

- **OR original:** Kelley & Walker, "Critical-Path Planning and Scheduling," IRE-AIEE-ACM '59 (Eastern Joint Computer Conference), 1959. Developed at DuPont for chemical plant construction scheduling.
- **Textbook:** Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 6th ed., Springer 2022. Chapter 2 (single machine), Chapter 3 (parallel machines with precedence constraints).
- **Compiler version:** "Critical path length priority" in list scheduling. Shobaki et al. (CGO 2020) use it without citing CPM. LLVM's GenericScheduler uses it as primary heuristic.
- **What the compiler version misses:** CPM includes resource leveling and time-cost tradeoffs (crashing). Compiler list scheduling only uses the priority, not the full CPM framework.
- **tinygrad formulation:** [01-instruction-scheduling.md](01-instruction-scheduling.md) — CPL priority replaces flat `{LOAD:-1, ALU:0, STORE:+1}`.

### Hu's Algorithm

- **OR original:** Hu, T.C., "Parallel Sequencing and Assembly Line Problems," *Operations Research* 9(6), 1961. Proves optimal scheduling for unit-time tasks with tree precedence constraints on identical parallel machines.
- **Textbook:** Pinedo Ch. 3.2. Also: Brucker, *Scheduling Algorithms*, 5th ed., Springer 2007.
- **Compiler version:** Same algorithm, called "longest path priority" or "critical path heuristic." Optimal for expression trees, 2-approximation for general DAGs.
- **Gap:** None — the compiler community correctly adopted this result. But they rarely cite Hu.

### Coffman-Graham Algorithm

- **OR original:** Coffman & Graham, "Optimal Scheduling for Two-Processor Systems," *Acta Informatica* 1(3), 1972. Optimal for unit-time tasks with 2 processors.
- **Textbook:** Pinedo Ch. 3.2. Brucker Ch. 4.
- **Compiler version:** Used in some instruction schedulers for bounded-width scheduling. Less common than CPL.
- **Gap:** Coffman-Graham is optimal for 2 processors but generalizes well as a heuristic for bounded resources. The compiler community prefers CPL even when resource bounds matter.
- **tinygrad relevance:** Register pressure = bounded resource. Coffman-Graham may outperform CPL when register pressure is the binding constraint (H1's kill condition scenario).

### RCPSP (Resource-Constrained Project Scheduling Problem)

- **OR original:** Pritsker, Watters & Wolfe, "Multiproject Scheduling with Limited Resources: A Zero-One Programming Formulation," *Management Science* 16(1), 1969.
- **Textbook:** Hartmann & Briskorn, "A Survey of Variants and Extensions of the Resource-Constrained Project Scheduling Problem," *EJOR* 207(1), 2010. Brucker Ch. 8.
- **Compiler version:** "Register-pressure-aware instruction scheduling." Shobaki et al. (TACO 2022) solve a restricted version with ACO/B&B. LLVM's scheduler has ad-hoc register pressure heuristics.
- **What the compiler version misses:** RCPSP has polyhedral relaxations (Artigues et al., 2003; Koné et al., 2011) that give tight lower bounds. These could evaluate schedule quality without exhaustive search. Also: RCPSP priority rule benchmarks (Kolisch 1996, Hartmann 2002) systematically compared 20+ priority rules — compiler papers test at most 3-4.
- **tinygrad formulation:** [01-instruction-scheduling.md](01-instruction-scheduling.md) — the full problem is RCPSP with renewable resources (registers) and step-function cost (APRP occupancy tiers).

### Sethi-Ullman Numbering

- **OR original:** Sethi & Ullman, "The Generation of Optimal Code for Arithmetic Expressions," *JACM* 17(4), 1970.
- **Textbook:** Aho, Lam, Sethi & Ullman, *Compilers: Principles, Techniques, and Tools*, 2nd ed. (the Dragon Book), Ch. 8.10.
- **Compiler version:** Correctly adopted. One of the few OR results the compiler community cites properly.
- **Gap:** Optimal for trees; NP-hard for DAGs. Chen (arXiv 2023) gives a DAG heuristic within ~17% of optimal. OR has tighter approximation algorithms for special DAG structures (series-parallel, bounded treewidth).

---

## Assignment & Layout

### Assignment Problem / Hungarian Algorithm

- **OR original:** Kuhn, "The Hungarian Method for the Assignment Problem," *Naval Research Logistics* 2(1-2), 1955. Based on König's theorem (1931) and Egerváry's work (1931).
- **Textbook:** Papadimitriou & Steiglitz, *Combinatorial Optimization*, Ch. 11. Schrijver, *Combinatorial Optimization*, Ch. 17.
- **Compiler version:** Bank conflict avoidance is never formulated as an assignment problem. The compiler community uses XOR-swizzle heuristics (CUTLASS) without connecting to the underlying combinatorial structure.
- **Gap:** The Hungarian algorithm is overkill here because the GF(2) structure gives a closed-form. But framing it as an assignment problem makes the optimality proof trivial: XOR-swizzle is optimal because it produces a perfect matching in the bipartite graph (threads × banks).
- **tinygrad formulation:** [02-bank-conflict-avoidance.md](02-bank-conflict-avoidance.md)

### Bin Packing

- **OR original:** Johnson, "Near-Optimal Bin Packing Algorithms," PhD thesis, MIT, 1973. First-Fit Decreasing, Best-Fit Decreasing.
- **Textbook:** Vazirani, *Approximation Algorithms*, Ch. 9. Korte & Vygen, *Combinatorial Optimization*, Ch. 18.
- **Compiler version:** tinygrad's memory planner (`schedule/memory.py`) uses TLSF (Two-Level Segregated Fit) allocation — a real-time variant of best-fit. The compiler community treats memory planning as an engineering problem, not an optimization problem.
- **Gap:** TLSF is designed for real-time guarantees (O(1) alloc/free), not optimal packing. Offline bin packing (which tinygrad's memory planner actually is, since all lifetimes are known at schedule time) has stronger algorithms: First-Fit Decreasing is an 11/9·OPT + 6/9 approximation. Whether this matters depends on whether memory is actually the bottleneck.

---

## Decomposition Methods

### Benders Decomposition

- **OR original:** Benders, "Partitioning Procedures for Solving Mixed-Variables Programming Problems," *Numerische Mathematik* 4, 1962.
- **Textbook:** Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Ch. 6.5. Conforti, Cornuéjols & Zambelli, *Integer Programming*, Ch. 11.
- **Compiler version:** Flashlight's online softmax (You et al., 2025). The "complicating variable" (running max) is handled by the master problem (rescaling), while subproblems (per-element exp and accumulation) proceed independently. Neither the Flashlight paper nor FlashAttention cite Benders.
- **Gap:** Benders decomposition includes a systematic procedure for generating optimality cuts — each time the subproblem is infeasible or suboptimal, a cut is added to the master problem. Flashlight's approach is a single-round Benders: the rescaling is derived analytically, not iteratively. For more complex reduction chains (beyond softmax), the iterative Benders procedure might discover decompositions that algebraic analysis misses.
- **tinygrad formulation:** [04-reduction-fusion.md](04-reduction-fusion.md) — Level 1 (homomorphism).

### Dantzig-Wolfe Decomposition

- **OR original:** Dantzig & Wolfe, "Decomposition Principle for Linear Programs," *Operations Research* 8(1), 1960.
- **Textbook:** Bertsimas & Tsitsiklis Ch. 6.4. Lübbecke & Desrosiers, "Selected Topics in Column Generation," *Operations Research* 53(6), 2005.
- **Compiler version:** RedFuser's `F(x,d) = G(x)·H(d)` separable decomposition (Tang et al., ASPLOS 2026). The linking constraint (the combining function) is decomposed into independent subproblems (track G and H separately). Not cited as Dantzig-Wolfe.
- **Gap:** Dantzig-Wolfe includes column generation — dynamically adding variables to the master problem as needed. For reduction fusion, this would mean dynamically discovering new tracked quantities (beyond sum_x and sum_x²) that enable decomposition. RedFuser's approach is static: the decomposition must be specified a priori.
- **tinygrad formulation:** [04-reduction-fusion.md](04-reduction-fusion.md) — Level 2 (separable decomposition).

### Lagrangian Relaxation

- **OR original:** Fisher, "The Lagrangian Relaxation Method for Solving Integer Programming Problems," *Management Science* 27(1), 1981. (Survey; technique dates to the 1970s.)
- **Textbook:** Bertsimas & Tsitsiklis Ch. 6.3. Wolsey, *Integer Programming*, Ch. 10.
- **Compiler version:** Neptune's algebraic correction terms (Zhao et al., PLDI 2026). Relax the loop-carried dependency, solve the relaxed problem (unfused reduction), then add a correction term that compensates for the relaxation. Not cited as Lagrangian relaxation.
- **Gap:** Lagrangian relaxation includes subgradient optimization for tightening the dual bound. Neptune's corrections are derived analytically for specific patterns. The systematic Lagrangian procedure could handle patterns that resist closed-form correction.
- **tinygrad formulation:** [04-reduction-fusion.md](04-reduction-fusion.md) — Level 3 (correction terms).

---

## Partitioning

### Hypergraph Partitioning (Kernighan-Lin / METIS)

- **OR original:** Kernighan & Lin, "An Efficient Heuristic Procedure for Partitioning Graphs," *Bell System Technical Journal* 49(2), 1970. Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs," *SIAM J. Sci. Comput.* 20(1), 1998 (METIS).
- **Textbook:** Çatalyürek & Aykanat, *Hypergraph Partitioning and Clustering*, 2011.
- **Compiler version:** "Operator fusion heuristics." Every ML compiler (XLA, TVM, tinygrad) has ad-hoc rules for deciding which ops to fuse into a single kernel. SpaceFusion++ (Zhu et al., JSA 2026) is the closest to the OR formulation: it explicitly models the partition cost (inter-kernel data movement) and resource constraints (registers, shared memory per partition).
- **Gap:** The compiler community uses syntactic rules ("fuse elementwise after reduce") instead of solving the partition problem. METIS-style multilevel partitioning with resource constraints would give better fusion decisions, but the compile-time cost may be prohibitive for a JIT compiler.
- **tinygrad formulation:** Implicit in the scheduler (`schedule/__init__.py`). The scheduler splits at reduction boundaries — this is a fixed partitioning rule, not an optimization.

---

## Knapsack

### 0-1 Knapsack / Bounded Knapsack

- **OR original:** Dantzig, "Discrete-Variable Extremum Problems," *Operations Research* 5(2), 1957. Kellerer, Pferschy & Pisinger, *Knapsack Problems*, Springer, 2004.
- **Textbook:** Korte & Vygen Ch. 18. Vazirani Ch. 8 (FPTAS).
- **Compiler version:** BEAM search over kernel optimization actions is an implicit knapsack: each action (UPCAST, UNROLL, LOCAL) consumes resources (registers, shared memory) and provides benefit (throughput). The compiler community doesn't formulate it as knapsack — they use beam search.
- **Gap:** Knapsack has FPTAS (fully polynomial-time approximation scheme). For a fixed register/shared-memory budget, the optimal combination of UPCAST/UNROLL/LOCAL amounts could be computed in pseudo-polynomial time instead of searched.
- **tinygrad formulation:** [03-fused-dequantization.md](03-fused-dequantization.md) — fused dequant adds items (dequant ops) to the knapsack (register budget).

---

## Queuing Theory

### Kingman's Formula / Factory Physics

- **OR original:** Kingman, "The Single Server Queue in Heavy Traffic," *Mathematical Proceedings of the Cambridge Philosophical Society* 57(4), 1961. Hopp & Spearman, *Factory Physics*, 3rd ed., Waveland Press, 2011.
- **Textbook:** Hopp & Spearman Part II. Gross & Harris, *Fundamentals of Queueing Theory*, 4th ed.
- **Compiler version:** Volkov, "Better Performance at Lower Occupancy," GPU Technology Conference, 2010. Rediscovered that maximizing GPU occupancy (= utilization) hurts throughput when it increases queuing delays in the memory system.
- **Gap:** Factory Physics derives the exact relationship: cycle time = process time × (utilization / (1 - utilization)) × variability. The compiler community has the empirical observation (Volkov) but not the queuing-theoretic foundation. This matters because queuing theory predicts *where* the optimal occupancy point is (as a function of memory latency variance), while Volkov's observation only says "it's not 100%."
- **tinygrad relevance:** The APRP step function (Shobaki et al.) is an approximation to the queuing-theoretic optimum. A proper queuing model would give the optimal occupancy target as a function of kernel characteristics (memory intensity, compute intensity, access pattern variance), not as a fixed table.

---

## Summary Table

| OR Field | Key Result | Year | Compiler Rediscovery | Year | Gap |
|---|---|---|---|---|---|
| Project scheduling | CPM | 1959 | List scheduling with CPL | ~1990s | Missing resource leveling |
| Scheduling | Hu's algorithm | 1961 | Same | ~1970s | None (correctly adopted) |
| Scheduling | RCPSP | 1969 | RP-aware scheduling | 2020 | Missing polyhedral relaxations, priority rule benchmarks |
| Scheduling | Sethi-Ullman | 1970 | Same | 1970 | NP-hard DAG extension |
| Assignment | Hungarian | 1955 | XOR-swizzle | ~2018 | Overkill — GF(2) gives closed form |
| Bin packing | FFD | 1973 | TLSF memory planner | ~2020 | Offline vs online distinction |
| Decomposition | Benders | 1962 | Online softmax | 2022 | Missing cut generation |
| Decomposition | Dantzig-Wolfe | 1960 | RedFuser | 2026 | Missing column generation |
| Relaxation | Lagrangian | 1981 | Neptune corrections | 2026 | Missing subgradient optimization |
| Partitioning | Kernighan-Lin | 1970 | Fusion heuristics | ~2015 | Ad-hoc rules vs optimization |
| Knapsack | FPTAS | 1957+ | BEAM search | ~2020 | Search vs closed-form |
| Queuing | Kingman's formula | 1961 | Volkov occupancy | 2010 | Missing variance modeling |
