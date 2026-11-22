# subsume

[![crates.io](https://img.shields.io/crates/v/subsume.svg)](https://crates.io/crates/subsume)
[![Documentation](https://docs.rs/subsume/badge.svg)](https://docs.rs/subsume)

Region embeddings for entailment and set containment.

`subsume` represents concepts as geometric regions. A general concept contains
the regions for its more specific concepts. It scores containment for
hierarchies, ontologies, and set queries.
In Python, UMass IESL's `box-embeddings` (PyTorch and TensorFlow) implements
hard and Gumbel boxes; use `subsume` for the same box scoring from Rust, or for
its other region geometries (cones, balls, Gaussians and more).

![Box embedding concepts](docs/box_concepts.png)

Nested boxes encode an `is-a` relationship. The right panel shows how Gumbel
boxes replace a hard boundary with a temperature-controlled one.

## Install

Add these dependencies to your Rust project’s `Cargo.toml`.

```toml
[dependencies]
subsume = "0.18"
ndarray = "0.16"
```

The default features include the ndarray backend and knowledge-graph dataset
helpers. Burn training examples require `burn-ndarray` for CPU or `burn-wgpu`
for a WGPU device.

## Usage

Save this as `src/main.rs` and run `cargo run`.

```rust
use ndarray::array;
use subsume::{ndarray_backend::NdarrayBox, BoxError, HyperBox};

fn main() -> Result<(), BoxError> {
    // `general` contains `specific`.
    let general = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;
    let specific = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

    let probability = general.containment_prob(&specific)?;
    println!("P(specific in general) = {probability:.1}");
    let reverse = specific.containment_prob(&general)?;
    println!("P(general in specific) = {reverse:.3}");
    Ok(())
}
```

```text
P(specific in general) = 1.0
P(general in specific) = 0.216
```

For hard boxes, the denominator is the volume of the argument. The smaller
box lies entirely inside the larger one and covers 0.6³ = 0.216 of its volume.
Using these geometric scores as probabilities of observed labels requires calibration. The triple convention is `head contains tail`; reverse datasets
that instead store `(child, hypernym, parent)`.

## Choose a geometry

| Task | Start with | Notes |
| --- | --- | --- |
| Containment hierarchy | `NdarrayBox` or `NdarrayGumbelBox` | Boxes have volume and intersection; Gumbel boxes give dense gradients |
| Logical queries with negation | Cone or subspace | Cones and subspaces support complement-like operations |
| Taxonomy expansion with distributional spread | Gaussian boxes | KL gives asymmetric containment; Bhattacharyya gives overlap |
| EL++ ontology completion | `el`, `transbox` | Uses axiom losses rather than plain triple scoring |
| Tree-like hierarchies in low dimension | Hyperbolic points or balls | Useful when depth is the main structure |

The full geometry table is in [`docs/geometries.md`](docs/geometries.md).
Scores are meaningful within one geometry, but are not calibrated across
geometries. The `region_generic` example ranks candidates within each geometry.

## Examples

Run the bundled examples from a repository checkout.

```bash
git clone https://github.com/arclabs561/subsume.git
cd subsume
cargo run --example containment_hierarchy
cargo run --example region_generic
```

The first example prints containment and overlap scores for a small hierarchy.
The second ranks three hand-built candidates per geometry. To train boxes and
evaluate held-out links, run `cargo run --release --example dataset_training`.
It reports head and tail MRR separately and identifies the known answers
removed at validation and test time. The [example guide](examples/README.md)
also covers ontology, query, and data-gated benchmarks.

Python bindings are published as `subsumer`; see
[`subsume-python/README.md`](subsume-python/README.md).

## Benchmarks

EL++ ontology completion results and reproduction commands are in
[`docs/benchmarks.md`](docs/benchmarks.md). The current strongest results are on
NF3 existential restrictions, with MRR 0.21-0.37 across GALEN, GO, and Anatomy in
recorded single-run Burn runs.

## Limits

- For ordinary link prediction, point embeddings are often simpler.
- Region scores from different geometries are not directly comparable.
- Region volume expresses geometric generality; it is not calibrated epistemic
  or target uncertainty. Gaussian geometry represents distributional spread,
  with assumptions set by the caller.
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
