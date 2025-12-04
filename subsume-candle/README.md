# subsume-candle

⚠️ **Currently disabled due to dependency conflicts**

This crate provides `CandleBox` and `CandleGumbelBox` types that implement the `Box` and `GumbelBox` traits using `candle_core::Tensor`.

## Known Issue

The `candle-core` crate has internal dependency conflicts between `rand` 0.8/0.9 and `half` 2.7.1 that prevent compilation. This is an upstream issue in `candle-core` itself, not in this crate.

## Status

- ✅ Implementation complete
- ❌ Compilation blocked by `candle-core` dependency conflicts
- ⏳ Waiting for upstream fix in `candle-core`

## Workaround

If you need candle support, you can:
1. Manually add `subsume-candle` to workspace members in root `Cargo.toml`
2. Use `subsume-ndarray` backend instead (fully functional)
3. Wait for `candle-core` to resolve the `rand`/`half` version conflicts

## Related Issues

- [candle-core issue #2805](https://github.com/huggingface/candle/issues/2805) - rand/half version conflicts
