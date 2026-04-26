# Changelog

## Unreleased

### Deprecated

- `candle-backend` feature is deprecated and will be removed in 0.13.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-ndarray` (CPU) or `burn-wgpu` (cross-platform GPU). NVIDIA users on the candle CUDA path should follow the burn-cuda roadmap.
