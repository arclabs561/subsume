# subsume

[![crates.io](https://img.shields.io/crates/v/subsume.svg)](https://crates.io/crates/subsume)
[![Documentation](https://docs.rs/subsume/badge.svg)](https://docs.rs/subsume)
[![CI](https://github.com/arclabs561/subsume/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/subsume/actions/workflows/ci.yml)

Geometric box embeddings: containment, entailment, overlap. Ndarray and Candle backends.

![Box embedding concepts](docs/box_concepts.png)

## What it provides

| Component | What it does |
|---|---|
| `Box` trait | Framework-agnostic axis-aligned hyperrectangle: volume, containment, overlap, distance |
| `GumbelBox` trait | Probabilistic boxes via Gumbel random variables (dense gradients, no flat regions) |
| `NdarrayBox` / `NdarrayGumbelBox` | CPU backend using `ndarray::Array1<f32>` |
| `CandleBox` / `CandleGumbelBox` | GPU/Metal backend using `candle_core::Tensor` |
| Training utilities | Negative sampling, volume regularization, temperature scheduling, AMSGrad |
| Evaluation | Mean rank, MRR, Hits@k, NDCG, calibration, reliability diagrams |
| Sheaf networks | Sheaf neural networks for transitivity (Hansen & Ghrist 2019) |
| Hyperbolic boxes | Box embeddings in Poincare ball (via `hyperball`) |

## Usage

```toml
[dependencies]
subsume = { version = "0.1.1", features = ["ndarray-backend"] }
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
cargo run -p subsume --example containment_hierarchy  # taxonomic is-a relationships with nested boxes
cargo run -p subsume --example gumbel_box_training    # Gumbel boxes, soft containment, temperature effects
```

## Tests

```bash
cargo test -p subsume
```

380+ unit tests + doc tests covering box operations (intersection, union, containment, overlap, distance, truncation), Gumbel box membership and temperature edge cases, serialization round-trips, training metrics (MRR, Hits@k, NDCG), calibration diagnostics, negative sampling, sheaf networks, hyperbolic geometry, quasimetric properties, and more.

## Why Gumbel boxes?

![Gumbel noise robustness](docs/gumbel_robustness.png)

Gumbel boxes model coordinates as Gumbel random variables, creating soft boundaries
that provide dense gradients throughout training. Hard boxes create flat regions where
gradients vanish; Gumbel boxes solve this *local identifiability problem*
(Dasgupta et al., 2020). As shown above, this also makes containment robust to
coordinate noise -- Gumbel containment loss stays near zero even at high perturbation
levels where Gaussian boxes fail completely.

## References

- Vilnis et al. (2018). "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Dasgupta et al. (2020). "Improving Local Identifiability in Probabilistic Box Embeddings"
- Ren et al. (2020). "Query2Box: Reasoning over Knowledge Graphs using Box Embeddings"

## License

MIT OR Apache-2.0
