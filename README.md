# subsume

[![crates.io](https://img.shields.io/crates/v/subsume.svg)](https://crates.io/crates/subsume)
[![Documentation](https://docs.rs/subsume/badge.svg)](https://docs.rs/subsume)
[![CI](https://github.com/arclabs561/subsume/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/subsume/actions/workflows/ci.yml)

Geometric region embeddings for subsumption, entailment, and logical query answering. Boxes, cones, octagons, Gaussians, hyperbolic intervals, and sheaf networks. Ndarray and Candle backends.

![Box embedding concepts](docs/box_concepts.png)

*(a) Containment: nested boxes encode taxonomic is-a relationships. (b) Gumbel soft boundary: temperature controls membership sharpness. (c) Octagon: diagonal constraints cut corners for tighter volume bounds.*

## What it provides

### Geometric primitives

| Component | What it does |
|---|---|
| `Box` trait | Axis-aligned hyperrectangle: volume, containment, overlap, distance |
| `GumbelBox` trait | Probabilistic boxes via Gumbel random variables (dense gradients, no flat regions; Dasgupta et al., 2020) |
| `Cone` trait | Angular cones in d-dimensional space: containment via aperture, closed under negation (inspired by Zhang & Wang, NeurIPS 2021) |
| `Octagon` trait | Axis-aligned polytopes with diagonal constraints; tighter volume bounds than boxes (Charpenay & Schockaert, IJCAI 2024) |
| `gaussian` | Diagonal Gaussian boxes: KL divergence (asymmetric containment) and Bhattacharyya coefficient (symmetric overlap) |
| `hyperbolic` | Poincare ball embeddings and hyperbolic box intervals (via `hyperball`) |
| `sheaf` | Sheaf diffusion primitives: stalks, restriction maps, Laplacian (Hansen & Ghrist 2019; Bodnar et al., ICLR 2022) |
| `Region` trait | Generic region interface unifying boxes, cones, and balls |

### Scoring and query answering

| Component | What it does |
|---|---|
| BoxE scoring | Point-entity BoxE model (Abboud et al., 2020) + box-to-box variant |
| `distance` | Depth-based (RegD), boundary, and vector-to-box distance metrics |
| `fuzzy` | Fuzzy t-norms/t-conorms for logical query answering (FuzzQE, Chen et al., AAAI 2022) |
| `el` | EL++ ontology embedding: inclusion loss, role translation/composition, existential boxes, disjointness (Box2EL/TransBox) |

### Taxonomy and training

| Component | What it does |
|---|---|
| `taxonomy` | TaxoBell-format dataset loader: `.terms`/`.taxo` parsing, train/val/test splitting |
| `taxobell` | TaxoBell combined loss: Bhattacharyya triplet + KL containment + volume regularization + sigma clipping |
| Training utilities | Negative sampling, temperature scheduling, AMSGrad, cosine annealing LR |
| Evaluation | MRR, Hits@k, NDCG, calibration (ECE, Brier), reliability diagrams |

### Backends

| Component | What it does |
|---|---|
| `NdarrayBox` / `NdarrayGumbelBox` / `NdarrayCone` / `NdarrayOctagon` | CPU backend using `ndarray::Array1<f32>` |
| `CandleBox` / `CandleGumbelBox` | GPU/Metal backend using `candle_core::Tensor` |

## Usage

```toml
[dependencies]
subsume = { version = "0.1.6", features = ["ndarray-backend"] }
ndarray = "0.16"
```

```rust
use subsume::ndarray_backend::NdarrayBox;
use subsume::Box as BoxTrait;
use ndarray::array;

// Box A: [0,0,0] to [1,1,1] (general concept)
let premise = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;

// Box B: [0.2,0.2,0.2] to [0.8,0.8,0.8] (specific, inside A)
let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

// Containment probability: P(B inside A)
let p = premise.containment_prob(&hypothesis, 1.0)?;
assert!(p > 0.9);
```

## Examples

```bash
cargo run -p subsume --example containment_hierarchy    # taxonomic is-a relationships with nested boxes
cargo run -p subsume --example gumbel_box_exploration   # Gumbel boxes, soft containment, temperature effects
cargo run -p subsume --example cone_training            # training cone embeddings on a taxonomy
cargo run -p subsume --example box_training             # training box embeddings on a 25-entity taxonomy
cargo run -p subsume --example taxobell_demo            # TaxoBell Gaussian box losses on a mini taxonomy
cargo run -p subsume --example query2box                # Query2Box: multi-hop queries, box intersection, distance scoring
cargo run -p subsume --example octagon_demo             # octagon embeddings: diagonal constraints, containment, volume
cargo run -p subsume --example fuzzy_query              # fuzzy query answering: t-norms, De Morgan duality, rankings
cargo run -p subsume --example el_training              # EL++ box embeddings on a biomedical-style ontology
cargo run -p subsume --features candle-backend --example taxobell_training  # TaxoBell MLP encoder training (Candle)
```

See [`examples/README.md`](examples/README.md) for a guide to choosing the right example.

## Tests

```bash
cargo test -p subsume
```

Unit, property, and doc tests covering:

- Box geometry: intersection, union, containment, overlap, distance, volume, truncation
- Gumbel boxes: membership probability, temperature edge cases, Bessel volume
- Cones: angular containment, negation closure, aperture bounds
- Octagon: intersection closure, containment, Sutherland-Hodgman volume
- Fuzzy: t-norm/t-conorm commutativity, associativity, De Morgan duality
- Gaussian boxes, EL++ ontology losses, sheaf networks, hyperbolic geometry, quasimetrics
- Training: MRR, Hits@k, NDCG, calibration, negative sampling, AMSGrad

## Choosing a geometry

| Geometry | When to use it | Negation? | Key tradeoff |
|---|---|---|---|
| Box / GumbelBox | Axis-aligned containment hierarchies, each dimension independent | No | Simple, fast; Gumbel variant adds dense gradients |
| Cone | Multi-hop reasoning with NOT; FOL queries requiring negation | Yes | Closed under complement, but angular parameterization is harder to initialize |
| Octagon | Rule-aware KG completion; need tighter volume than boxes | No | Tighter bounds via diagonal constraints; more parameters per entity |
| Gaussian | Taxonomy expansion with uncertainty; TaxoBell-style training | No | KL gives asymmetric containment for free; Bhattacharyya gives symmetric overlap |
| Hyperbolic | Tree-like hierarchies with low distortion | No | Exponential capacity in limited dimensions; numerical care near boundary |

## Why Gumbel boxes?

![Gumbel noise robustness](docs/gumbel_robustness.png)

*Containment loss under increasing coordinate noise for Gumbel, Gaussian, and hard boxes. Gumbel boxes remain stable at perturbation levels where other formulations fail.*

Gumbel boxes model coordinates as Gumbel random variables, creating soft boundaries
that provide dense gradients throughout training. Hard boxes create flat regions where
gradients vanish; Gumbel boxes solve this *local identifiability problem*
(Dasgupta et al., 2020). As shown above, this also makes containment robust to
coordinate noise -- Gumbel containment loss stays near zero even at high perturbation
levels where Gaussian boxes fail completely.

## Training convergence

![Training convergence](docs/training_convergence.png)

*25-entity taxonomy learned over 200 epochs. Left: total violation drops 3 orders of magnitude. Right: containment probabilities converge to 1.0 at different rates depending on hierarchy depth. Reproduce: `cargo run --example box_training` or `uv run scripts/plot_training.py`.*

## References

- Vilnis et al. (2018). "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Nickel & Kiela (2017). "Poincare Embeddings for Learning Hierarchical Representations"
- Abboud et al. (2020). "BoxE: A Box Embedding Model for Knowledge Base Completion"
- Dasgupta et al. (2020). "Improving Local Identifiability in Probabilistic Box Embeddings"
- Ren et al. (2020). "Query2Box: Reasoning over Knowledge Graphs using Box Embeddings"
- Hansen & Ghrist (2019). "Toward a Spectral Theory of Cellular Sheaves"
- Bodnar et al. (2022). "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"
- Zhang & Wang (2021). "ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs"
- Chen et al. (2022). "Fuzzy Logic Based Logical Query Answering on Knowledge Graphs"
- Jackermeier et al. (2023). "Dual Box Embeddings for the Description Logic EL++"
- Yang, Chen & Sattler (2024). "TransBox: EL++-closed Ontology Embedding"
- Charpenay & Schockaert (2024). "Capturing Knowledge Graphs and Rules with Octagon Embeddings"
- Huang et al. (2023). "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs"
- Yang & Chen (2025). "Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions"
- Xiong et al. (2026). "TaxoBell: Taxonomy Expansion via Bell-Curve Gaussian Box Embeddings"

## See also

- [`innr`](https://crates.io/crates/innr) -- SIMD-accelerated vector similarity primitives
- [`kuji`](https://crates.io/crates/kuji) -- stochastic sampling (Gumbel-max uses the same distribution)
- [`anno`](https://crates.io/crates/anno) -- information extraction; uses subsume's box embeddings for coreference resolution

## License

MIT OR Apache-2.0
