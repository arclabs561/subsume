# From Hard Boxes to Geometric Embeddings

How the geometric embedding approach evolved across the research literature,
and where each geometry type in subsume fits in the progression.

---

## The progression

### 1. Point embeddings (2013-2019)

TransE (Bordes et al., 2013, NeurIPS) represents entities as points and relations as
translations: `h + r ~ t`. Simple, fast, and surprisingly effective for link
prediction. DistMult (Yang et al., 2015, ICLR) modeled relations as diagonal
matrices. ComplEx (Trouillon et al., 2016, ICML) extended to complex space for
antisymmetric relations. RotatE (Sun et al., 2019, ICLR) represented relations as
rotations in complex space, enabling composition patterns.

**Benchmark results (link prediction):**

| Model | WN18RR MRR | WN18RR H@10 | FB15k-237 MRR | FB15k-237 H@10 |
|-------|-----------|-------------|--------------|----------------|
| TransE | 0.226 | 0.501 | 0.294 | 0.465 |
| DistMult | 0.430 | 0.490 | 0.241 | 0.419 |
| ComplEx | 0.440 | 0.510 | 0.247 | 0.428 |
| RotatE | 0.476 | 0.571 | 0.338 | 0.533 |

Source: RotatE paper (Sun et al., 2019, ICLR) Table 2; confirmed by independent
reproduction in Ali et al. (2021, "Bringing Light Into the Dark").

**Training objective**: margin-based or cross-entropy scoring on `score(h, r, t)`
with negative sampling. TransE minimizes `||h + r - t||`; RotatE minimizes
`||h * r - t||` in complex space (element-wise product = rotation).

**Limitation**: Points have no interior. They cannot encode containment
("dog is-a animal"), volume (how general a concept is), or set operations
(intersection, union, complement). Relations that require modeling set inclusion
(hypernymy, subsumption, part-of) are structurally out of reach.

**Failure modes**: TransE cannot model symmetric relations (`r = 0` collapses
head and tail), 1-to-N relations (multiple tails for same head+relation get
pulled to the same point), or reflexive relations. RotatE fixes symmetry
(rotation by pi) and composition (rotation addition) but still has no notion
of containment.

---

### 2. Hard boxes (2018)

Vilnis et al. (2018, ACL) introduced box lattice measures: represent entities as
axis-aligned hyperrectangles (boxes). Each entity gets two vectors:
min-coordinates and max-coordinates. Containment becomes geometric nesting:
box A inside box B means A is-a B. Volume encodes generality. Intersection
is closed (intersection of two boxes is a box).

**Containment formula**: Box A is contained in box B iff for all dimensions i:
`min_B[i] <= min_A[i]` and `max_A[i] <= max_B[i]`. Probabilistically,
`P(A|B) = vol(A intersect B) / vol(B)`.

**Training objective**: maximize `log P(child | parent)` for positive hypernym
pairs, minimize for negative pairs.

**Results**: On WordNet noun hypernymy (82,114 entities, transitive reduction),
~40% relative F1 improvement over order embeddings (Vendrov et al., 2016).

**Limitation**: Hard boundaries create flat regions in the loss landscape.
If two boxes are disjoint (no overlap), the gradient of containment
probability is exactly zero -- the optimizer receives no signal about which
direction to move. This is the "local identifiability problem": many different
box configurations produce identical loss values.

**Failure modes**: Gradient sparsity at initialization (randomly placed boxes
are usually disjoint). Requires careful initialization or curriculum strategies.
Cannot represent negative correlation between concepts.

---

### 3. Smoothed boxes (2019)

Li et al. (2019, ICLR oral) addressed the zero-gradient problem by convolving
box boundaries with a Gaussian kernel. The step function becomes a smooth sigmoid,
providing gradients everywhere. On WordNet hypernymy and Flickr caption
entailment, smoothed boxes matched or exceeded hard box performance with less
hyperparameter tuning.

**Containment formula**: Replace indicator functions with Gaussian CDFs.
For each dimension, `P(x in [a,b]) = Phi((b-x)/sigma) - Phi((a-x)/sigma)`.
Joint probability is the product over dimensions.

**Limitation**: Gaussian smoothing is not "max-stable" -- the intersection of
two Gaussian-smoothed boxes is not a Gaussian-smoothed box. This breaks
algebraic closure, meaning multi-hop operations accumulate approximation error.

---

### 4. Gumbel boxes (2020)

Dasgupta et al. (2020, NeurIPS) solved both problems simultaneously by
modeling box coordinates as Gumbel random variables instead of Gaussian.
The Gumbel distribution has a critical property: **max-stability**. The maximum
of independent Gumbel variables is still Gumbel. Since box intersection takes
coordinate-wise max/min, the intersection of Gumbel boxes is exactly a Gumbel
box. This gives:

- Dense gradients everywhere (soft boundaries)
- Algebraic closure under intersection
- Analytically tractable volume via the Bessel function K_0

**Containment formula**: `P(A subset B) = prod_i P(min_B[i] <= min_A[i]) * P(max_A[i] <= max_B[i])`,
where probabilities come from the CDF of the difference of two Gumbel
variables (logistic distribution). Volume uses softplus and the modified
Bessel function K_0.

**Training objective**: maximize log-probability of containment for positive
pairs. Temperature (beta) controls softness: low = crisp, high = smooth.
Training anneals from high to low.

**Results**: On WordNet noun hypernymy, Gumbel boxes improved F1 by ~6 points
over smoothed boxes (Li et al., 2019). The improvement was larger without
regularization, confirming that the dense gradients from Gumbel intersection
reduce the need for regularization.

**Failure modes**: K_0 is numerically unstable near zero; implementations
must use the softplus approximation. The Gumbel assumption is exact for
intersection but does not extend to union or complement.

---

### 5. BoxE: boxes for link prediction (2020)

Abboud et al. (2020, NeurIPS) brought box representations to standard link
prediction. BoxE represents entities as points and relations as box-shaped
regions: a triple (h, r, t) is true when the transformed head and tail fall
inside relation-specific boxes. This differs from the containment-based approach
(Vilnis et al.) where the boxes are entities.

**Benchmark results:**

| Model | WN18RR MRR | WN18RR H@10 | FB15k-237 MRR | FB15k-237 H@10 |
|-------|-----------|-------------|--------------|----------------|
| RotatE | 0.476 | 0.571 | 0.338 | 0.533 |
| BoxE | 0.451 | 0.541 | 0.337 | 0.538 |

Source: Abboud et al. (2020, NeurIPS) Table 2.

BoxE is competitive with RotatE on FB15k-237 and slightly behind on WN18RR.
Its advantage is theoretical: it can provably represent any knowledge graph
pattern (1-to-N, symmetric, antisymmetric, composition), some of which are
provably impossible for point-based methods.

---

### 6. Query answering with boxes (2020)

Ren et al. (2020, ICLR) showed that box intersection naturally models logical
conjunction in knowledge graph queries. Query2Box chains projection (relation
traversal) and intersection operations to answer multi-hop conjunctive queries
without enumerating paths. Each query becomes a box; answer entities are those
inside the box.

**Training objective**: minimize distance from answer entities to the query box
(inside = 0; outside = L1 to nearest boundary). Negative samples are non-answers.

**Results**: Query2Box outperformed GQE (Hamilton et al., 2018) by 15-25%
relative MRR on FB15k, FB15k-237, and NELL995 for EPFO queries.

**Limitation**: Box complement is not a box. Disjunction and negation require
a different geometry. Query2Box can only answer existential positive first-order
(EPFO) queries.

---

### 7. BetaE: probabilistic queries with negation (2020)

Ren & Leskovec (2020, NeurIPS) addressed the complement problem by representing
queries and entities as Beta distributions instead of boxes. The Beta distribution
is parameterized by (alpha, beta) on [0,1]^d. The key insight: the reciprocal
of a Beta distribution (swapping alpha and beta) flips high-density regions to
low-density and vice versa, giving a natural negation operator.

**Operations**: conjunction via attention-weighted product of Beta distributions;
negation via reciprocal (swap alpha, beta); disjunction via De Morgan's law.
Scoring uses KL divergence between query and entity distributions.

**Results**: First embedding method to handle all FOL operations. On queries
with negation, BetaE substantially outperformed Query2Box (which cannot handle
negation) and GQE across FB15k, FB15k-237, and NELL995.

**Limitation**: Beta distributions are on bounded [0,1]^d. Conjunction is a
learned approximation (no algebraic closure). Disjunction via De Morgan
accumulates error through double negation.

---

### 8. Cones for negation (2021)

Zhang & Wang (2021, NeurIPS) introduced ConE: represent queries as Cartesian
products of two-dimensional cones (angular sectors). The complement of a cone
is another cone, giving exact geometric negation without approximation.
Intersection of cones is a cone (take the tighter angular bounds).
Union of cones remains a cone (take the wider angular bounds).

**Containment formula**: Entity p is inside a cone with axis a, aperture theta
if `angle(p, a) <= theta`. The full representation is a product of d/2 sectors.

**Training objective**: distance-based scoring. Inside distance = 0; outside
distance proportional to angular gap.

**Results** (average MRR on queries with negation):
- ConE outperformed BetaE by 25.4% relative on FB15k, 9.3% on FB15k-237,
  and 8.5% on NELL.
- ConE was the first geometry-based model to achieve exact closure under all
  three Boolean operations (AND, OR, NOT).

Source: Zhang & Wang (2021, NeurIPS) Table 2.

**Limitation**: The 2D-sector decomposition limits expressiveness per dimension
pair. ConE is designed for query answering, not direct link prediction.

---

### 9. Gaussian boxes for taxonomy expansion (2021-2026)

BEUrRE (Chen et al., 2021, NAACL) added uncertainty to box embeddings by
modeling boundaries as Gaussian distributions. Each entity is a box with
uncertain boundaries, and relations are affine transforms on head/tail boxes.
The probabilistic calibration means that model confidence scores are meaningful,
not just ordinal.

**Containment formula**: KL divergence between Gaussian-boundary boxes gives
asymmetric containment (parent-child). Bhattacharyya coefficient gives
symmetric overlap.

**Training objective**: maximize calibrated probability of observed triples;
minimize for corrupted triples. Includes ranking + calibration terms.

**Results**: On uncertain KGs (CN15k, NL27k), BEUrRE outperformed deterministic
baselines on both confidence prediction and fact ranking.

TaxoBell (Mishra et al., 2026, WWW) combined Gaussian boxes with MLP encoders
for self-supervised taxonomy expansion, using KL + Bhattacharyya + volume
regularization as a joint loss.

**Failure modes**: Extra parameters (mean + variance per boundary) can overfit
on small datasets. Bhattacharyya is symmetric and cannot distinguish direction
without the KL term.

---

### 10. Octagons for tighter geometry (2024)

Charpenay & Schockaert (2024, IJCAI) added diagonal constraints to boxes.
An octagon is a box with additional constraints on coordinate sums and
differences: `a <= x_i + x_j <= b`, `c <= x_i - x_j <= d` for pairs of
dimensions. This cuts unreachable corners, giving tighter volume bounds
while remaining closed under intersection.

**Containment formula**: Same as boxes for axis-aligned constraints, plus checks
on sum/difference constraints. Intersection extends coordinate-wise max/min to
the diagonal constraints.

**Composition**: Octagons support relational composition via explicit parameter
combination. For O_1 compose O_2, result parameters are maxima/minima of
specific input parameter combinations.

**Key advantage over boxes**: BoxE cannot model relational composition. Octagons
capture "closed path rules" `r1(X,Y) AND r2(Y,Z) -> r3(X,Z)`. The geometric
constraints directly encode which logical rules the model has learned, making
them inspectable.

**Results**: Near competitive on standard benchmarks. The value is not raw
MRR but rule transparency and composition support.

**Limitation**: Constraint count grows quadratically with dimension pairs.

---

### 11. EL++ ontology embeddings (2019-2025)

Description Logic (DL) ontology embeddings differ from knowledge graph
embeddings: they represent not just facts (ABox) but also concept-level
axioms (TBox). The target is not link prediction on triples but subsumption
prediction and axiom entailment.

**ELEm** (Kulmanov et al., 2019) was the first: represent EL++ concepts as
n-balls, with subsumption as ball containment. Limited to simple axioms.

**Box2EL** (Jackermeier et al., 2023) represented concepts and roles as boxes.
Subsumption (`C sqsubseteq D`) becomes box containment. Role restrictions
(`exists R.C`) become translated boxes via "bump vectors."

**DELE** (Mashkova et al., 2024) added deductive closure filtering --
incorporating deductive reasoning results into the training process to avoid
learning redundant or contradictory embeddings.

**TransBox** (Yang & Chen, 2024, WWW 2025) achieved EL++-closure: every
expressible axiom in the DL language has a faithful geometric representation.
Box2EL's bump vectors are undefined for conjunctions like `A AND B`; TransBox
handles this through box-based role composition.

**Benchmark results on ontology subsumption** (Hits@100, mean rank):

| Method | GO H@100 | GO MR | Anatomy H@100 | Anatomy MR |
|--------|---------|-------|---------------|------------|
| Box2EL | 0.05 | 4516 | 0.05 | 19744 |
| ELBE | 0.15 | 5217 | 0.10 | 15661 |
| TransBox | 0.65 | 717 | 0.69 | 622 |

Source: Yang & Chen (2024, arXiv:2410.14571) Table 2.

**Containment formula**: concept C subsumed by D iff box_C inside box_D.
Role restriction `exists R.C` maps to `translate(box_C, bump_R)`.

**Training objective**: penalize axiom violations. `C sqsubseteq D`: box_C
must be inside box_D. `C AND D sqsubseteq E`: intersection must be inside
box_E. `C AND D sqsubseteq bot`: intersection must be empty.

**Strong faithfulness** (Lacerda et al., 2024): proved that normalized ELH
ontologies can be faithfully embedded using convex regions -- an embedding
exists that precisely captures all entailed axioms. This is a theoretical
guarantee that box-based approaches are not inherently lossy for ELH.

---

### 12. Hyperbolic embeddings (2017-2019)

Nickel & Kiela (2017, NeurIPS) showed that the Poincare ball model has
exponential capacity: its volume grows exponentially with radius, making it
natural for tree-like hierarchies. A tree with branching factor k and depth d
requires O(k^d) points in Euclidean space but O(d) in hyperbolic space.

**Distance**: `d(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))`.
Entities near the origin are general; near the boundary, specific.

**Training**: minimize hyperbolic distance for connected pairs, maximize for
negatives. Uses Riemannian SGD with the Poincare metric tensor.

**MuRP** (Balazevic et al., 2019, NeurIPS) extended Poincare embeddings to
multi-relational graphs via Mobius matrix-vector multiplication and Mobius
addition.

**Benchmark results:**

| Model | WN18RR MRR | WN18RR H@10 |
|-------|-----------|-------------|
| MuRP (dim=40) | 0.481 | 0.566 |
| MuRE (Euclidean, dim=40) | 0.458 | 0.554 |
| RotatE (dim=500) | 0.476 | 0.571 |

Source: Balazevic et al. (2019, NeurIPS) Table 1.

MuRP matches RotatE on WN18RR with 12.5x fewer parameters (40 vs 500
dimensions), demonstrating the efficiency of hyperbolic space for hierarchical
data. On FB15k-237 (less hierarchical), the advantage disappears.

**RegD** (Yang & Chen, 2025): proved a bijective isometry between Euclidean
ball regions and hyperbolic space H^(n+1), showing region structure matters
more than ambient space. On WordNet noun hierarchy and biomedical ontologies
(GO, GALEN, Anatomy), RegD outperformed both hyperbolic baselines (ShadowCone,
EntailmentCone) and Euclidean baselines (ELBE, ELEm).

**Failure modes**: Riemannian optimization is 2-5x slower than Euclidean SGD.
Numerical precision degrades near the boundary. Non-hierarchical relations
get no benefit from hyperbolic space.

---

### 13. Sheaf networks (2019-2022)

Hansen & Ghrist (2019) developed the spectral theory of cellular sheaves.
Bodnar et al. (2022, NeurIPS) applied this to GNNs, using sheaf Laplacian
diffusion to enforce consistency across graph edges. Each edge carries a
restriction map that defines how features should transform between nodes.

**Core idea**: A cellular sheaf assigns a vector space (stalk) to each node
and a linear map (restriction) to each edge. The sheaf Laplacian generalizes
the graph Laplacian; diffusion with it drives features toward global consistency.

**Training**: learn restriction maps end-to-end with a task-specific loss.

**Key result**: Sheaf neural networks outperform standard GNNs on heterophilic
benchmarks (connected nodes with different labels). Standard message-passing
assumes homophily; sheaf diffusion enforces "consistent difference" through
restriction maps. For KGs, restriction maps encode relation-specific
transformations (TransE's translations generalized to arbitrary linear maps).

**Limitation**: O(d^2) parameters per edge. Practical adoption lags behind
simpler GNN architectures despite strong theoretical foundations.

---

### 14. The box-embeddings library (2021)

Chheda et al. (2021, EMNLP demo) released the `box-embeddings` Python library:
numerically stable hard, smoothed, and Gumbel boxes with PyTorch and TensorFlow
backends. Boratko et al. (2021, NeurIPS) showed box embeddings have higher
capacity than order embeddings for DAGs; (2021, UAI) developed min/max stability
theory.

This is the Python-ecosystem counterpart to `subsume`. The key difference:
`box-embeddings` focuses on the probabilistic box framework, while `subsume`
spans a wider range of geometries with both CPU and GPU backends.

---

## Connections and tradeoffs

Each method addresses a specific limitation but introduces new tradeoffs.

**Key tradeoff axes**:

1. **Expressiveness vs. closure**: Boxes are closed under intersection but not
   complement. Cones are closed under all Boolean operations. Beta distributions
   handle all operations but approximately. More closure = more composable
   multi-hop queries.

2. **Gradient density vs. geometric fidelity**: Hard boxes have exact geometry
   but sparse gradients. Gumbel boxes have dense gradients and exact intersection.
   Gaussian smoothing has dense gradients but inexact intersection. Temperature
   annealing bridges the gap by starting soft and ending crisp.

3. **Capacity vs. parameters**: Hyperbolic space gives exponential capacity
   for hierarchy at the cost of Riemannian optimization. Euclidean boxes give
   volume-based hierarchy with standard optimization. RegD (2025) suggests
   the region structure may matter more than the ambient space.

4. **Rules vs. flexibility**: Octagons and EL++ embeddings make learned rules
   explicit and inspectable. Point embeddings and neural methods are more
   flexible but opaque. When domain rules are known a priori, geometric
   constraints are an advantage. When the structure is unknown, flexibility wins.

---

## Practical guidance: choosing a geometry

### Decision factors

**Dataset structure**: Is the data a flat knowledge graph (FB15k-237), a
hierarchy (WordNet, Gene Ontology), or a full ontology with axioms (GALEN)?

**Task**: Link prediction (which triples are missing)? Subsumption prediction
(which concepts subsume which)? Multi-hop query answering (logical queries
over the graph)? Taxonomy expansion (where does a new concept fit)?

**Computational budget**: Hyperbolic optimization is 2-5x slower than Euclidean
due to Riemannian gradients. Sheaf networks add O(d^2) per edge. Gumbel boxes
require Bessel function evaluation. Points are cheapest.

**Relation types**: Mostly hierarchical? Use hyperbolic or boxes. Mixed
(hierarchy + symmetry + composition)? Use RotatE or BoxE. Requires negation?
Use ConE or BetaE.

### When to use each geometry

| Geometry | Best for | Avoid when |
|----------|----------|------------|
| Points (TransE/RotatE) | General link prediction on large KGs; composition-heavy relations | Subsumption, containment, or hierarchy tasks |
| Hard/Gumbel boxes | Hierarchy modeling; subsumption prediction; hypernymy | Non-hierarchical relations dominate; need negation |
| Cones | Query answering with negation and disjunction | Simple link prediction (overkill); purely hierarchical data |
| BetaE | Probabilistic query answering with uncertainty | Deterministic tasks; small datasets (overfits) |
| Gaussian boxes | Uncertain knowledge graphs; taxonomy expansion | Datasets without confidence scores; standard link prediction |
| Octagons | Rule-aware link prediction; inspectable models | When rules are unknown and you want maximum flexibility |
| EL++ (Box2EL/TransBox) | Ontology completion; subsumption with axiom structure | Standard KGs without TBox axioms |
| Hyperbolic | Deep hierarchies in low dimensions; tree-like data | Flat or cyclic graphs; when you need standard optimizers |
| Sheaf networks | Heterophilic graphs; relation-specific message passing | Homophilic graphs (standard GNNs suffice); very large dense graphs |

### Rules of thumb

1. **Start with RotatE** for general link prediction. It is fast, well-understood,
   and competitive on most benchmarks. Switch to a geometric method only when
   RotatE fails on your specific structure.

2. **Use Gumbel boxes** (not hard or smoothed) when you need containment semantics.
   The algebraic closure under intersection is worth the added complexity.

3. **Use hyperbolic only for deep, narrow hierarchies**. If the hierarchy is
   shallow (depth < 10) or the branching factor is high, Euclidean boxes with
   sufficient dimensions match hyperbolic performance with simpler optimization.
   RegD (2025) showed that Euclidean region embeddings can emulate hyperbolic
   capacity.

4. **Use TransBox over Box2EL** for ontology tasks. TransBox handles the full
   EL++ language; Box2EL breaks on conjunctions and treats roles as implicitly
   transitive.

5. **Use ConE over BetaE** when geometric precision matters for query answering.
   ConE has exact Boolean closure; BetaE has approximate operations. BetaE's
   advantage is probabilistic semantics for uncertain queries.

6. **Dimension sizing**: Point embeddings typically need 200-500 dimensions.
   Gumbel boxes achieve similar expressiveness in 32-64 dimensions (each
   dimension encodes a min-max interval). Hyperbolic embeddings can work in
   5-40 dimensions for pure hierarchies.

---

## Where subsume fits

subsume implements the core geometries as composable Rust types with a shared
evaluation framework:

| Geometry | Type | Paper | What it adds over the previous |
|----------|------|-------|-------------------------------|
| Hard box | `NdarrayBox` | Vilnis 2018 | Containment, volume, intersection |
| Gumbel box | `NdarrayGumbelBox` | Dasgupta 2020 | Dense gradients, algebraic closure |
| Cone | `NdarrayCone` | Zhang & Wang 2021 | Negation (complement closure) |
| Gaussian box | `gaussian` module | BEUrRE 2021, TaxoBell 2026 | Uncertainty, KL containment |
| Octagon | `NdarrayOctagon` | Charpenay 2024 | Tighter volume, diagonal constraints |
| Hyperbolic | `hyperbolic` module | Nickel & Kiela 2017 | Exponential tree capacity |
| Sheaf | `sheaf` module | Hansen 2019, Bodnar 2022 | Edge-level consistency enforcement |
| EL++ | `el` module | Box2EL 2023, TransBox 2024 | Description logic axioms |

The GPU backend (`CandleBox`, `CandleGumbelBox`) provides the same box
operations on Metal/CUDA for training workloads.

---

## Key references

Abboud+ 2020 (BoxE, NeurIPS) -- Balazevic+ 2019 (MuRP, NeurIPS) --
Bodnar+ 2022 (Sheaf diffusion, NeurIPS) -- Bordes+ 2013 (TransE, NeurIPS) --
Boratko+ 2021 (Box capacity, NeurIPS; Min/max stability, UAI) --
Charpenay & Schockaert 2024 (Octagons, IJCAI) --
Chen+ 2021 (BEUrRE, NAACL) -- Chheda+ 2021 (box-embeddings lib, EMNLP) --
Dasgupta+ 2020 (Gumbel boxes, NeurIPS) -- Hansen & Ghrist 2019 (Sheaf theory) --
Jackermeier+ 2023 (Box2EL, ICLR) -- Kulmanov+ 2019 (ELEm) --
Lacerda+ 2024 (Strong faithfulness, ELH) -- Li+ 2019 (Smoothed boxes, ICLR) --
Mashkova+ 2024 (DELE) -- Mishra+ 2026 (TaxoBell, WWW) --
Nickel & Kiela 2017 (Poincare, NeurIPS) --
Ren+ 2020a (Query2Box, ICLR) -- Ren & Leskovec 2020b (BetaE, NeurIPS) --
Sun+ 2019 (RotatE, ICLR) -- Vilnis+ 2018 (Box lattice, ACL) --
Yang & Chen 2024 (TransBox, WWW 2025) -- Yang & Chen 2025 (RegD) --
Zhang & Wang 2021 (ConE, NeurIPS).
