# subsume

Geometric box embeddings for containment / entailment.

## Overview

Represent a concept as an axis-aligned hyperrectangle. Subsumption is modeled by containment.

```text
B ⊆ A  ⇔  A subsumes B
```

## Example

```rust
use subsume_ndarray::NdarrayBox;
use subsume_core::Box;
use ndarray::array;

let premise = NdarrayBox::new(array![0.0, 0.0, 0.0], array![1.0, 1.0, 1.0], 1.0)?;
let hypothesis = NdarrayBox::new(array![0.2, 0.2, 0.2], array![0.8, 0.8, 0.8], 1.0)?;

let p = premise.containment_prob(&hypothesis, 1.0)?;
println!("P(hypothesis ⊆ premise) = {p:.3}");
```

## Crates

- `subsume-core`: traits and shared types
- `subsume-ndarray`: `ndarray` backend
- `subsume-candle`: `candle` backend

## Documentation

Start at `docs/READING_GUIDE.md`.

## References

- Vilnis et al. (2018): Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures
- Dasgupta et al. (2020): Improving Local Identifiability in Probabilistic Box Embeddings
- Boratko et al. (2020): BoxE

## License

MIT OR Apache-2.0
