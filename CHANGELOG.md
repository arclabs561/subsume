# Changelog

## [Unreleased]

### Added

- `Region` trait: a unified, geometry-generic subsumption interface (`dim` + `subsumption_score`) implemented by the box backends, `Ball`, `Ellipsoid`, `Subspace`, `GaussianBox`, `SphericalCap`, and `AnnularSector`, so retrieval/ranking code can be written once and reused across geometries. The score is monotone WITHIN a geometry (more-contained ranks higher) but is not calibrated across geometries (a box's volume-ratio 1.0 and a ball's sigmoid-of-margin 0.5 are different scales) -- documented on the trait. Method named `subsumption_score` (not `containment_prob`) to avoid colliding with `HyperBox::containment_prob` on box types. New `region_generic` example. Cones, octagons, TransBox, and the feature-gated hyperbolic/sheaf/density geometries are intentionally not covered: their relations are not symmetric nested containment (`P(inner ⊆ self)`), so they don't share this contract.
- `gaussian::containment_prob(child, parent, k)`: soft containment probability for diagonal Gaussians via `sigmoid(-k * KL)`, parallel to `ellipsoid::containment_prob`.

### Changed

- Renamed the core `Box` trait to `HyperBox` so it no longer shadows `std::boxed::Box`. "Box" in the geometric-embedding literature means an n-dimensional hyperrectangle, which is exactly what the trait models, so `HyperBox` keeps the literature term while removing the std collision the trait's own docs previously worked around. All internal code and tests use `HyperBox`.

### Notes

- The `candle-backend`, deprecated in 0.12.1 with a stated 0.13.0 removal, is NOT removed in 0.13.0. burn does not yet provide equivalents for GPU box training (`CandleBoxTrainer`), cone training (`candle_cone_trainer`), or the TaxoBell encoder, so removing candle now would regress those geometries. The direction remains burn-everywhere (`burn-wgpu` gives Metal on Mac); candle removal is deferred until burn box/cone/TaxoBell trainers land. burn already covers EL, ball, ellipsoid, subspace, annular, transbox, and spherical-cap training.

### Deprecated

- `Box` is now a deprecated re-export aliased to `HyperBox`; it still compiles (with a deprecation warning) but will be removed in a future major. Switch `use subsume::Box` to `use subsume::HyperBox`. The non-shadowing `BoxRegion` alias is retained.

## [0.12.1] - 2026-06-10

### Deprecated

- `candle-backend` feature is deprecated and will be removed in 0.13.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-ndarray` (CPU) or `burn-wgpu` (cross-platform GPU). NVIDIA users on the candle CUDA path should follow the burn-cuda roadmap.
