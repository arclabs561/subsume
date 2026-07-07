# subsume

[![crates.io](https://img.shields.io/crates/v/subsume.svg)](https://crates.io/crates/subsume)
[![Documentation](https://docs.rs/subsume/badge.svg)](https://docs.rs/subsume)

Region embeddings for entailment and set containment.

`subsume` represents concepts as geometric regions. A general concept contains
the regions for its more specific concepts, so containment becomes the scoring
operation for hierarchy, ontology, and set-query tasks.

![Box embedding concepts](docs/box_concepts.png)

*(a) Containment: nested boxes encode is-a relationships. (b) Gumbel soft boundary: temperature controls membership sharpness.*

## Install

```toml
[dependencies]
subsume = "0.17.0"
ndarray = "0.16"
```

The default features include the ndarray backend and knowledge-graph dataset
helpers. GPU training examples use Burn through `burn-ndarray` or `burn-wgpu`.

## Usage

```rust
use ndarray::array;
use subsume::{ndarray_backend::NdarrayBox, HyperBox};

// A is the general concept.
let premise = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;

// B is the specific concept inside A.
let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

let p = premise.containment_prob(&hypothesis)?;
assert!(p > 0.9);
```

Triple convention: the head box contains the tail box. For datasets where
triples are `(child, hypernym, parent)`, pass `reverse=True` to the Python
loader or reverse the triples before training.

## Training

```rust,ignore
use std::path::Path;
use subsume::{dataset::load_dataset, BoxEmbeddingTrainer, TrainingConfig};

let dataset = load_dataset(Path::new("data/wn18rr"))?;
let interned = dataset.into_interned();
let train: Vec<_> = interned.train.iter().map(|t| (t.head, t.relation, t.tail)).collect();

let config = TrainingConfig { learning_rate: 0.01, epochs: 50, ..Default::default() };
let mut trainer = BoxEmbeddingTrainer::new(config, 32);
let result = trainer.fit(&train, None, None)?;
println!("MRR: {:.3}", result.final_results.mrr);
```

Python bindings are published as `subsumer`:

```bash
pip install subsumer
```

See [`subsume-python/README.md`](subsume-python/README.md) for Python examples.

## Choosing A Geometry

| Task | Start with | Notes |
| --- | --- | --- |
| Containment hierarchy | `NdarrayBox` or `NdarrayGumbelBox` | Boxes have volume and intersection; Gumbel boxes give dense gradients |
| Logical queries with negation | Cone or subspace | Cones and subspaces support complement-like operations |
| Taxonomy expansion with uncertainty | Gaussian boxes | KL gives asymmetric containment; Bhattacharyya gives overlap |
| EL++ ontology completion | `el`, `transbox` | Uses axiom losses rather than plain triple scoring |
| Tree-like hierarchies in low dimension | Hyperbolic intervals or balls | Useful when depth is the main structure |

The full geometry table is in [`docs/geometries.md`](docs/geometries.md).
Scores are monotone within a geometry but not calibrated across geometries; see
`cargo run --example region_generic`.

## Why Regions

Point embeddings such as TransE, RotatE, and ComplEx work well for ordinary link
prediction. Regions are useful when the task needs structure that points do not
have:

| Need | Point embeddings | Region embeddings |
| --- | --- | --- |
| Containment | No interior | Box nesting |
| Generality | No volume | Larger region = broader concept |
| Intersection | No set operation | Box intersection |
| Negation | No complement | Cone or subspace complement |
| Uncertainty | Extra model-specific machinery | Region size or Gaussian variance |

For background, see [Why Regions, Not Points](https://attobop.net/posts/region-embeddings/)
and [`docs/SUBSUMPTION_HISTORY.md`](docs/SUBSUMPTION_HISTORY.md).

## Examples

```bash
cargo run --example containment_hierarchy
cargo run --example box_training
cargo run --example el_training
```

The full example map is in [`examples/README.md`](examples/README.md).

## Benchmarks

EL++ ontology completion results and reproduction commands are in
[`docs/benchmarks.md`](docs/benchmarks.md). The current strongest results are on
NF3 existential restrictions, with MRR 0.21-0.37 across GALEN, GO, and Anatomy in
the recorded single-run Burn benchmark.

## Limits

- For ordinary link prediction, point embeddings are often simpler.
- Region scores from different geometries are not directly comparable.
- Several geometry trainers are research paths, not recommended defaults.
- GPU examples depend on Burn backend features and dataset files under `data/`.

## Documentation

- [Geometry table](docs/geometries.md)
- [EL++ benchmarks](docs/benchmarks.md)
- [CLQA evaluation](docs/CLQA_EVAL.md)
- [Research history](docs/SUBSUMPTION_HISTORY.md)
- [Python bindings](subsume-python/README.md)

## License

MIT OR Apache-2.0
