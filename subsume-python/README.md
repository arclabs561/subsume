# subsume-python

Python bindings for subsume: geometric region embeddings for knowledge graph subsumption.

## Install

```bash
pip install maturin
cd subsume-python
maturin develop
```

## Usage

```python
import subsume

box_a = subsume.NdarrayBox([0.0, 0.0], [1.0, 1.0], 1.0)
box_b = subsume.NdarrayBox([0.2, 0.2], [0.8, 0.8], 1.0)
prob = box_a.containment_prob(box_b, 1.0)
print(f"P(B inside A) = {prob:.4f}")
```
