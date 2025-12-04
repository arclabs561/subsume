# subsume-candle

✅ **Fully functional - dependency conflict resolved**

This crate provides `CandleBox` and `CandleGumbelBox` types that implement the `Box` and `GumbelBox` traits using `candle_core::Tensor`.

## Status

- ✅ Dependency conflict resolved (half downgraded to 2.3.1 via Cargo resolution)
- ✅ All compilation errors fixed
- ✅ Implementation complete and tested

## Dependency Resolution

The `candle-core` crate uses `half` 2.3.1 and `rand_distr` 0.4.3, but newer transitive dependencies were pulling in `half` 2.7.1 which requires `rand_distr` 0.5. This conflict was resolved by Cargo's dependency resolution automatically downgrading `half` to 2.3.1 to match `candle-core`'s requirements.

## Implementation Notes

- Error conversion from `candle_core::Error` to `BoxError` is done via `.map_err()` throughout
- Uses `std::result::Result` instead of `candle_core::Result` (which only takes 1 type parameter)
- Tensor product computed manually (no `prod_all()` method in candle-core 0.4)ub.com/huggingface/candle/issues/2805) - rand/half version conflicts
