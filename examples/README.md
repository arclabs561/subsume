# subsume examples

## Examples

| Example | What it demonstrates |
|---------|---------------------|
| `containment_hierarchy` | Hand-constructed box hierarchy; containment/overlap probabilities; temperature effects on sharpness |
| `gumbel_box_exploration` | Gumbel box properties: soft containment, membership probability, temperature annealing (no training) |
| `cone_training` | Training cone embeddings on an 18-entity taxonomy (3+ levels); aperture reflects generality |
| `box_training` | Training box embeddings with direct coordinate updates on a 25-entity taxonomy; volume reflects generality |
| `query2box` | Query2Box-style compositional query answering: multi-hop KG queries via box intersection, containment ranking, and alpha-weighted distance scoring |
| `octagon_demo` | Octagon embeddings: diagonal constraints, point containment, intersection (closure), volume comparison, soft containment/overlap |
| `fuzzy_query` | Fuzzy query answering with t-norms (Min/Product/Lukasiewicz), t-conorms, negation, and De Morgan duality on a small KG |
| `taxobell_demo` | TaxoBell Gaussian box losses on a mini taxonomy (no training, loss inspection only) |
| `el_training` | End-to-end EL++ box embedding training on a biomedical-style ontology |
| `taxobell_training` | TaxoBell MLP encoder training with Candle autograd (requires `--features candle-backend`) |

## Decision tree

- **Want to understand box geometry (containment, overlap, volume)?**
  Start with `containment_hierarchy`.

- **Want to understand Gumbel temperature effects and soft membership?**
  Start with `gumbel_box_exploration`.

- **Want to answer multi-hop knowledge graph queries with box intersection?**
  Start with `query2box`.

- **Want to understand octagon embeddings (boxes + diagonal constraints)?**
  Start with `octagon_demo`.

- **Want to explore fuzzy logic operators for query answering?**
  Start with `fuzzy_query`.

- **Want to train embeddings on a hierarchy?**
  - Cone model (angular containment, supports negation): `cone_training`
  - Box model (axis-aligned hyperrectangles, volume-based): `box_training`
  - EL++ ontology (subsumption + roles): `el_training`
  - TaxoBell (Gaussian boxes + MLP encoder, Candle): `taxobell_training`

- **Want to see TaxoBell losses without training?**
  Start with `taxobell_demo`.

## Running

```bash
cargo run -p subsume --example containment_hierarchy
cargo run -p subsume --example gumbel_box_exploration
cargo run -p subsume --example cone_training
cargo run -p subsume --example box_training
cargo run -p subsume --example query2box
cargo run -p subsume --example octagon_demo
cargo run -p subsume --example fuzzy_query
cargo run -p subsume --example taxobell_demo
cargo run -p subsume --example el_training
cargo run -p subsume --features candle-backend --example taxobell_training
```

## Visualization scripts

The `scripts/` directory contains Python plotting scripts (PEP 723, run with `uv run`):

- `scripts/plot_box_concept.py` -- generates `docs/box_concepts.png` (containment, Gumbel soft boundary, octagon vs box)
- `scripts/plot_training.py` -- generates `docs/training_convergence.png` (loss curve + containment probability convergence)
- `scripts/plot_gumbel_robustness.py` -- generates `docs/gumbel_robustness.png` (Gumbel vs Gaussian vs hard-box noise robustness)
- `scripts/plot_temperature_sensitivity.py` -- generates `docs/temperature_sensitivity.png` (containment probability vs Gumbel temperature)
- `scripts/plot_fuzzy_tnorms.py` -- generates `docs/fuzzy_tnorms.png` (t-norm contour plots and 1D slices)
