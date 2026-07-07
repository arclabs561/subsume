# Geometries

`subsume` exposes several region parameterizations. Start with boxes unless the
task needs a specific operation, such as negation, uncertainty, or EL++ axiom
closure.

Scores are meaningful within a geometry. Do not compare raw scores across
geometries without calibration.

| Geometry | When to use it | Negation? | Main tradeoff |
| --- | --- | --- | --- |
| `NdarrayBox` / `NdarrayGumbelBox` | Containment hierarchies, each dimension independent | No | Simple and fast; Gumbel boxes add dense gradients |
| Cone | Multi-hop queries requiring negation | Yes | Closed under complement; angular parameterization is harder to initialize |
| Octagon | Rule-aware KG completion; tighter containment than boxes | No | Diagonal constraints add parameters |
| Gaussian box | Taxonomy expansion with uncertainty | No | KL is asymmetric containment; Bhattacharyya is symmetric overlap |
| Hyperbolic interval | Tree-like hierarchies with exponential branching | No | Low-dimensional capacity; numerical care near the Poincare boundary |
| Ball | Spherical containment | No | Fewer parameters than boxes; analytical gradients available |
| Spherical cap | Directional containment | No | May need more epochs than boxes |
| Subspace | Conjunction, disjunction, and negation via projection | Yes | Closed under set operations; finite-difference gradients are slow at high dimension |
| Ellipsoid | Full-covariance containment via Cholesky | No | More expressive than boxes; O(d^2) parameters |
| TransBox | EL++-closed ontology embedding | No | Designed for description-logic semantics; requires box parameterization |
| Annular sector | Angular position plus spread for KGE | No | Experimental; behavior depends on dataset shape |

## Examples

```bash
cargo run --example containment_hierarchy
cargo run --example gumbel_box_exploration
cargo run --example cone_training
cargo run --example octagon_demo
cargo run --features hyperbolic --example hyperbolic_demo
cargo run --features kge --example geometry_comparison
```

See [`../examples/README.md`](../examples/README.md) for the full example list
and captured output.
