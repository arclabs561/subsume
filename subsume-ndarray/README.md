# subsume-ndarray

Ndarray implementation of `subsume-core` traits for box embeddings.

This crate provides `NdarrayBox` and `NdarrayGumbelBox` types that implement the `Box` and `GumbelBox` traits using `ndarray::Array1<f32>`.

## Example

```rust
use subsume_ndarray::NdarrayBox;
use subsume_core::Box;
use ndarray::array;

let min = array![0.0, 0.0, 0.0];
let max = array![1.0, 1.0, 1.0];

let box_a = NdarrayBox::new(min, max, 1.0)?;
let volume = box_a.volume(1.0)?;
```

