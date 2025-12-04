# subsume-candle

✅ **Dependency conflict resolved - compilation in progress**

This crate provides `CandleBox` and `CandleGumbelBox` types that implement the `Box` and `GumbelBox` traits using `candle_core::Tensor`.

## Status

- ✅ Dependency conflict resolved (half downgraded to 2.3.1 via Cargo resolution)
- 🔧 Compilation errors being fixed (error conversion from `candle_core::Error` to `BoxError`)
- ⏳ Implementation nearly complete

## Dependency Resolution

The `candle-core` crate uses `half` 2.3.1 and `rand_distr` 0.4.3, but newer transitive dependencies were pulling in `half` 2.7.1 which requires `rand_distr` 0.5. This conflict has been resolved by Cargo's dependency resolution automatically downgrading `half` to 2.3.1 to match `candle-core`'s requirements.

## Current Work

Fixing error conversion issues where `candle_core::Error` needs to be converted to `BoxError` using `.map_err()` throughout the implementation.ub.com/huggingface/candle/issues/2805) - rand/half version conflicts
