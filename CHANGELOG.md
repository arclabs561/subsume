# Changelog

## [Unreleased]

## [0.15.1] - 2026-07-06

### Added

- GALEN CLQA diagnostics now report DCA-set targets, candidate-rank and
  candidate-pool recall, rank fusion, direct-frontier candidate pools,
  symbolic common-ancestor pools, and conformal set sizes inside those
  symbolic pools.

### Fixed

- Burn ellipsoid InfoNCE loss clamps the positive score before `log`, avoiding
  non-finite batch losses when a large KL underflows the sigmoid score to zero.

## [0.15.0] - 2026-07-05

### Added

- `clqa` module: conjunctive least-common-ancestor queries over faithful EL++
  box embeddings. `BoxClqa` answers "X such that A ⊑ X and B ⊑ X" by
  containment-gated proximity to the join box, the readout that recovers the
  LCA where plain containment saturates. On GALEN (dim 200, 300 conjunctive
  queries) it reaches top-1 LCA accuracy 0.60, up from 0.42 for proximity
  alone and 0.44 for containment, and beats a plain-KGE point baseline on the
  same queries.
- EL++ trainer evaluation now reports the full Box2EL metric suite
  (Hits@1/10/100, MRR, mean and median rank, AUC) with a filtered NF2
  protocol, plus a containment-based NF2 scorer that recovers ranking signal
  center-distance discards.
- `offset_clamp` training knob: a hard cap on box offset L1 to diagnose and
  suppress offset blowup.
- L2-normalized initialization for the burn box, transbox, ellipsoid, and
  octagon trainers, and relation-aware burn box, cone, and octagon trainers.
- Examples: a real-GALEN faithful-vs-plain CLQA head-to-head, a
  closure-grounded conjunctive-query benchmark, and graded and gated CLQA
  readouts over trained boxes with box-model diagnostics.

### Changed

- Link-prediction evaluation excludes the query entity from its own candidate
  pool.

## [0.14.1] - 2026-07-03

### Deprecated

- `cone_query::ConeQuery` / `cone_query::cone_containment_score` and the
  `fuzzy` t-norm/t-conorm items: superseded by the geometry-generic query
  engine and typed `Truth` algebras in the `heyting` crate. Removal planned
  for the next breaking release; this crate stays the geometry layer.

### Fixed

- `--no-default-features` builds again (the `Region` impls for the ndarray
  box backends are now feature-gated); CI gained a bare-build lane.

## [0.14.0] - 2026-06-28

### Added

- `Region` trait: a unified, geometry-generic subsumption interface (`dim` + `subsumption_score`) implemented by the box backends, `Ball`, `Ellipsoid`, `Subspace`, `GaussianBox`, `SphericalCap`, and `AnnularSector`, so retrieval/ranking code can be written once and reused across geometries. The score is monotone WITHIN a geometry (more-contained ranks higher) but is not calibrated across geometries (a box's volume-ratio 1.0 and a ball's sigmoid-of-margin 0.5 are different scales) -- documented on the trait. Method named `subsumption_score` (not `containment_prob`) to avoid colliding with `HyperBox::containment_prob` on box types. New `region_generic` example. Cones, octagons, TransBox, and the feature-gated hyperbolic/sheaf/density geometries are intentionally not covered: their relations are not symmetric nested containment (`P(inner ⊆ self)`), so they don't share this contract.
- `gaussian::containment_prob(child, parent, k)`: soft containment probability for diagonal Gaussians via `sigmoid(-k * KL)`, parallel to `ellipsoid::containment_prob`.

### Changed

- Renamed the core `Box` trait to `HyperBox` so it no longer shadows `std::boxed::Box`. "Box" in the geometric-embedding literature means an n-dimensional hyperrectangle, which is exactly what the trait models, so `HyperBox` keeps the literature term while removing the std collision the trait's own docs previously worked around. All internal code and tests use `HyperBox`.

### Removed

- The `candle-backend` and `cuda` features and the candle backend (`CandleBox`/`CandleGumbelBox`, the candle box/EL/cone trainers, the candle TaxoBell encoder, and the `subsumer` python `Candle*Trainer` bindings). Burn covers every training path now, and `burn-wgpu` provides GPU/Metal. **Breaking.**

### Deprecated

- `Box` is now a deprecated re-export aliased to `HyperBox`; it still compiles (with a deprecation warning) but will be removed in a future major. Switch `use subsume::Box` to `use subsume::HyperBox`. The non-shadowing `BoxRegion` alias is retained.

## [0.12.1] - 2026-06-10

### Deprecated

- `candle-backend` feature is deprecated and will be removed in 0.13.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-ndarray` (CPU) or `burn-wgpu` (cross-platform GPU). NVIDIA users on the candle CUDA path should follow the burn-cuda roadmap.
