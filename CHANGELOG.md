# Changelog

## [Unreleased]

### Changed

- Renamed the core `Box` trait to `HyperBox` so it no longer shadows `std::boxed::Box`. "Box" in the geometric-embedding literature means an n-dimensional hyperrectangle, which is exactly what the trait models, so `HyperBox` keeps the literature term while removing the std collision the trait's own docs previously worked around. All internal code and tests use `HyperBox`.

### Deprecated

- `Box` is now a deprecated re-export aliased to `HyperBox`; it still compiles (with a deprecation warning) but will be removed in a future major. Switch `use subsume::Box` to `use subsume::HyperBox`. The non-shadowing `BoxRegion` alias is retained.

## [0.12.1] - 2026-06-10

### Deprecated

- `candle-backend` feature is deprecated and will be removed in 0.13.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-ndarray` (CPU) or `burn-wgpu` (cross-platform GPU). NVIDIA users on the candle CUDA path should follow the burn-cuda roadmap.
