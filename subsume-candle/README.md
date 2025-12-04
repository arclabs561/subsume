# subsume-candle

Candle implementation of `subsume-core` traits for box embeddings.

This crate provides `CandleBox` and `CandleGumbelBox` types that implement the `Box` and `GumbelBox` traits using `candle_core::Tensor`.

## Example

```rust
use subsume_candle::CandleBox;
use subsume_core::Box;
use candle_core::{Device, Tensor};

let device = Device::Cpu;
let min = Tensor::new(&[0.0f32, 0.0, 0.0], &device)?;
let max = Tensor::new(&[1.0f32, 1.0, 1.0], &device)?;

let box_a = CandleBox::new(min, max, 1.0)?;
let volume = box_a.volume(1.0)?;
```

