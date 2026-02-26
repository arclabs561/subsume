# subsume examples

## Examples

| Example | What it demonstrates |
|---------|---------------------|
| `containment_hierarchy` | Hand-constructed box hierarchy; containment/overlap probabilities; temperature effects on sharpness |
| `gumbel_box_exploration` | Gumbel box properties: soft containment, membership probability, temperature annealing (no training) |
| `cone_training` | Training cone embeddings on an 18-entity taxonomy (3+ levels); aperture reflects generality |
| `box_training` | Training box embeddings with direct coordinate updates on a 25-entity taxonomy; volume reflects generality |
| `query2box` | Query2Box-style compositional query answering: multi-hop KG queries via box intersection and containment ranking |

## Decision tree

- **Want to understand box geometry (containment, overlap, volume)?**
  Start with `containment_hierarchy`.

- **Want to understand Gumbel temperature effects and soft membership?**
  Start with `gumbel_box_exploration`.

- **Want to answer multi-hop knowledge graph queries with box intersection?**
  Start with `query2box`.

- **Want to train embeddings on a hierarchy?**
  - Cone model (angular containment, supports negation): `cone_training`
  - Box model (axis-aligned hyperrectangles, volume-based): `box_training`

## Running

```bash
cargo run -p subsume --example containment_hierarchy
cargo run -p subsume --example gumbel_box_exploration
cargo run -p subsume --example cone_training
cargo run -p subsume --example box_training
cargo run -p subsume --example query2box
```

## Visualization scripts

The `scripts/` directory contains Python plotting scripts (PEP 723, run with `uv run`):

- `scripts/plot_box_concept.py` -- generates `docs/box_concepts.png` (containment, overlap, Gumbel soft boundary diagrams)
- `scripts/plot_gumbel_robustness.py` -- generates `docs/gumbel_robustness.png` (Gumbel vs Gaussian vs hard-box noise robustness)
- `scripts/plot_temperature_sensitivity.py` -- generates `docs/temperature_sensitivity.png` (containment probability vs Gumbel temperature)
