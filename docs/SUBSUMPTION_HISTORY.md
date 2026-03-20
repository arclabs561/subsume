# From Hard Boxes to Geometric Embeddings

How the geometric embedding approach evolved across the research literature,
and where each geometry type in subsume fits in the progression.

## The progression

### 1. Point embeddings (2013-2017)

TransE (Bordes et al., 2013) represents entities as points and relations as
translations: `h + r ~ t`. Simple, fast, and surprisingly effective for link
prediction. RotatE (Sun et al., 2019) improved this with rotations in complex
space, reaching 0.476 MRR on WN18RR.

**Limitation**: Points have no interior. They cannot encode containment
("dog is-a animal"), volume (how general a concept is), or set operations
(intersection, union, complement).

### 2. Hard boxes (2018)

Vilnis et al. (2018) introduced box lattice measures: represent entities as
axis-aligned hyperrectangles. Containment becomes geometric nesting:
box A inside box B means A is-a B. Volume encodes generality. Intersection
is closed (intersection of two boxes is a box).

**Limitation**: Hard boundaries create flat regions in the loss landscape.
If two boxes are disjoint (no overlap), the gradient of containment
probability is exactly zero -- the optimizer receives no signal about which
direction to move. This is the "local identifiability problem": many different
box configurations produce identical loss values.

### 3. Smoothed boxes (2019)

Li et al. (2019, ICLR) addressed the zero-gradient problem by convolving box
boundaries with a Gaussian kernel. The step function becomes a smooth sigmoid,
providing gradients everywhere.

**Limitation**: Gaussian smoothing is not "max-stable" -- the intersection of
two Gaussian-smoothed boxes is not a Gaussian-smoothed box. This breaks
algebraic closure, meaning multi-hop operations accumulate approximation error.

### 4. Gumbel boxes (2020)

Dasgupta et al. (2020, NeurIPS) solved both problems simultaneously by
modeling box coordinates as Gumbel random variables. The Gumbel distribution
has a critical property: **max-stability**. The maximum of independent Gumbel
variables is still Gumbel. Since box intersection takes coordinate-wise max/min,
the intersection of Gumbel boxes is exactly a Gumbel box. This gives:

- Dense gradients everywhere (soft boundaries)
- Algebraic closure under intersection
- Analytically tractable volume via the Bessel function K_0

Temperature (beta) controls boundary softness: low beta = crisp boundaries
(approaching hard boxes), high beta = smooth gradients. Training typically
anneals from high to low.

### 5. Query answering with boxes (2020)

Ren et al. (2020, ICLR) showed that box intersection naturally models logical
conjunction in knowledge graph queries. Query2Box chains projection (relation
traversal) and intersection operations to answer multi-hop conjunctive queries
without enumerating paths.

**Limitation**: Box complement is not a box. Disjunction and negation require
a different geometry.

### 6. Cones for negation (2021)

Zhang & Wang (2021, NeurIPS) introduced ConE: represent entities as angular
sectors (cones) in d-dimensional space. The complement of a cone is another
cone, enabling negation queries. ConE achieves 52.9 MRR on FB15k EPFO+negation
queries vs Query2Box's 41.0.

### 7. Gaussian boxes for taxonomy expansion (2022-2026)

BEUrRE (Chen et al., 2021, ACL) added uncertainty to box embeddings by
modeling boundaries as Gaussian distributions. KL divergence gives asymmetric
containment (parent-child), Bhattacharyya coefficient gives symmetric overlap.

TaxoBell (Mishra et al., 2026, WWW) combined this with MLP encoders for
self-supervised taxonomy expansion, using KL + Bhattacharyya + volume
regularization as a joint loss.

### 8. Octagons for tighter geometry (2024)

Charpenay & Schockaert (2024, IJCAI) added diagonal constraints to boxes.
An octagon is a box with additional constraints on coordinate sums and
differences (`a <= x_i + x_j <= b`, `c <= x_i - x_j <= d`). This cuts
unreachable corners, giving tighter volume bounds while remaining closed under
intersection. The extra parameters (4 per pair of adjacent dimensions) enable
capturing rules that boxes cannot.

### 9. EL++ ontology embeddings (2023-2025)

Box2EL (Jackermeier et al., 2023) and TransBox (Yang & Chen, 2024) use box
embeddings to represent description logic concepts. Subsumption (`C is-a D`)
becomes box containment, role restrictions (`exists R.C`) become translated
boxes. These achieve strong results on Gene Ontology, GALEN, and Anatomy
benchmarks where point embeddings structurally fail.

### 10. Hyperbolic embeddings (2017)

Nickel & Kiela (2017) showed that the Poincare ball model has exponential
capacity: its volume grows exponentially with radius, making it natural for
tree-like hierarchies. A tree with branching factor k and depth d requires
O(k^d) points in Euclidean space but O(d) in hyperbolic space.

### 11. Sheaf networks (2019-2022)

Hansen & Ghrist (2019) developed the spectral theory of cellular sheaves.
Bodnar et al. (2022, ICLR) applied this to GNNs, using sheaf Laplacian
diffusion to enforce consistency across graph edges. Each edge carries a
restriction map that defines how features should transform between nodes.

## Where subsume fits

subsume implements all of these as composable Rust types with a shared
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
