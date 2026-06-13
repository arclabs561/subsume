# Changelog

## [Unreleased]

### Added

- `Region` trait: a unified, geometry-generic subsumption interface (`dim` + `subsumption_score`) implemented by the box backends, `Ball`, `Ellipsoid`, and `Subspace`, so retrieval/ranking code can be written once and reused across geometries. The score is monotone WITHIN a geometry (more-contained ranks higher) but is not calibrated across geometries (a box's volume-ratio 1.0 and a ball's sigmoid-of-margin 0.5 are different scales) -- documented on the trait. Method named `subsumption_score` (not `containment_prob`) to avoid colliding with `HyperBox::containment_prob` on box types. New `region_generic` example. `AnnularSector`, `GaussianBox`, and `SphericalCap` are not yet covered (the first lacks a `dim()`; the others need their containment orientation verified).

### Changed

- Renamed the core `Box` trait to `HyperBox` so it no longer shadows `std::boxed::Box`. "Box" in the geometric-embedding literature means an n-dimensional hyperrectangle, which is exactly what the trait models, so `HyperBox` keeps the literature term while removing the std collision the trait's own docs previously worked around. All internal code and tests use `HyperBox`.

### Deprecated

- `Box` is now a deprecated re-export aliased to `HyperBox`; it still compiles (with a deprecation warning) but will be removed in a future major. Switch `use subsume::Box` to `use subsume::HyperBox`. The non-shadowing `BoxRegion` alias is retained.

## [0.12.1] - 2026-06-10

### Deprecated

- `candle-backend` feature is deprecated and will be removed in 0.13.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-ndarray` (CPU) or `burn-wgpu` (cross-platform GPU). NVIDIA users on the candle CUDA path should follow the burn-cuda roadmap.
