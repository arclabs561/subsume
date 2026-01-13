# subsume

Geometric box embeddings for modeling containment ("is-a") and entailment relationships.

Dual-licensed under MIT or Apache-2.0.

```rust
use subsume_ndarray::NdarrayBox;
use ndarray::array;

// Box A: [0,0,0] to [1,1,1]
let premise = NdarrayBox::new(array![0., 0., 0.], array![1., 1., 1.], 1.0)?;

// Box B: [0.2,0.2,0.2] to [0.8,0.8,0.8] (inside A)
let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

// Probability that A contains B
let p = premise.containment_prob(&hypothesis, 1.0)?;
println!("P(B ⊆ A) = {:.2}", p);
```

## Features

- **Gumbel Box**: Probabilistic boxes for training stability
- **Backends**: `ndarray` (CPU) and `candle` (GPU/Metal)
- **Training**: Volume regularization, temperature scheduling
- **Inference**: Fast containment and overlap scoring

See [`docs/`](docs/) for mathematical foundations and research details.
