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
| Gumbel operations | Softplus Bessel volume, LSE intersection (Dasgupta et al., 2020) |
| BoxE scoring | Point-entity BoxE model (Abboud et al., 2020) + box-to-box variant |
| `NdarrayBox` / `NdarrayGumbelBox` | CPU backend using `ndarray::Array1<f32>` |
| `CandleBox` / `CandleGumbelBox` | GPU/Metal backend using `candle_core::Tensor` |
| Training utilities | Negative sampling, volume regularization, temperature scheduling, AMSGrad |
| Evaluation | Mean rank, MRR, Hits@k, NDCG, calibration, reliability diagrams |
| Sheaf networks | Sheaf neural networks for transitivity (Hansen & Ghrist 2019) |
| Hyperbolic boxes | Box embeddings in Poincare ball (via `hyperball`) |
| `gaussian` | Diagonal Gaussian box embeddings: KL divergence (asymmetric containment) and Bhattacharyya coefficient (symmetric overlap) |
| `el` | EL++ ontology embedding primitives: inclusion loss, role translation/composition, existential boxes, disjointness (Box2EL/TransBox) |
| `taxonomy` | TaxoBell-format taxonomy dataset loader: `.terms`/`.taxo` parsing, train/val/test splitting, conversion to `Triple`s |
| `taxobell` | TaxoBell combined training loss: symmetric (Bhattacharyya triplet), asymmetric (KL containment), volume regularization, sigma clipping |

## Usage

```toml
[dependencies]
subsume = { version = "0.1.4", features = ["ndarray-backend"] }
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
```

See [`examples/README.md`](examples/README.md) for a guide to choosing the right example.

## Tests

```bash
cargo test -p subsume
```

614 unit tests + property tests + doc tests covering box operations (intersection, union, containment, overlap, distance, truncation), Gumbel box membership and temperature edge cases, serialization round-trips, training metrics (MRR, Hits@k, NDCG), calibration diagnostics, negative sampling, sheaf networks, hyperbolic geometry, quasimetric properties, Gaussian box KL/Bhattacharyya scoring, EL++ ontology losses, taxonomy dataset loading, and TaxoBell combined training losses.

## Why Gumbel boxes?

![Gumbel noise robustness](docs/gumbel_robustness.png)

Gumbel boxes model coordinates as Gumbel random variables, creating soft boundaries
that provide dense gradients throughout training. Hard boxes create flat regions where
gradients vanish; Gumbel boxes solve this *local identifiability problem*
(Dasgupta et al., 2020). As shown above, this also makes containment robust to
coordinate noise -- Gumbel containment loss stays near zero even at high perturbation
levels where Gaussian boxes fail completely.

## Training convergence

![Training convergence](docs/training_convergence.png)

*Box embeddings learning a 25-entity containment hierarchy over 200 epochs. Run `cargo run --example box_training` to reproduce, or `uv run scripts/plot_training.py` to regenerate the plot.*

## References

- Vilnis et al. (2018). "Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures"
- Abboud et al. (2020). "BoxE: A Box Embedding Model for Knowledge Base Completion"
- Dasgupta et al. (2020). "Improving Local Identifiability in Probabilistic Box Embeddings"
- Ren et al. (2020). "Query2Box: Reasoning over Knowledge Graphs using Box Embeddings"

## See also

- [`innr`](https://crates.io/crates/innr) -- SIMD-accelerated vector similarity primitives
- [`kuji`](https://crates.io/crates/kuji) -- stochastic sampling (Gumbel-max uses the same distribution)
- [`anno`](https://crates.io/crates/anno) -- information extraction with optional box-embedding coreference

## License

MIT OR Apache-2.0
