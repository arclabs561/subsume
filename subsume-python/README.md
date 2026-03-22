# subsume-python

Python bindings for subsume: geometric region embeddings for knowledge graph subsumption.

## Install

```bash
pip install subsumer
```

### From source

```bash
pip install maturin
cd subsume-python
maturin develop
```

## Quick Start

### Box embeddings

```python
import subsumer

box_a = subsumer.NdarrayBox([0.0, 0.0], [1.0, 1.0], 1.0)
box_b = subsumer.NdarrayBox([0.2, 0.2], [0.8, 0.8], 1.0)
prob = box_a.containment_prob(box_b)
print(f"P(B inside A) = {prob:.4f}")
```

### Training box embeddings on a knowledge graph

```python
import subsumer

# From string triples (head contains tail)
triples = [("animal", "hypernym", "dog"), ("animal", "hypernym", "cat")]
trainer, ids = subsumer.BoxEmbeddingTrainer.from_triples(triples)
result = trainer.fit(ids)
print(f"MRR: {result['mrr']:.3f}")

# With full configuration
config = subsumer.TrainingConfig(dim=32, epochs=50, learning_rate=0.01)
trainer, ids = subsumer.BoxEmbeddingTrainer.from_triples(triples, config=config)
result = trainer.fit(ids)

# From files (WN18RR format: train.txt, valid.txt, test.txt)
train, val, test, ents, rels = subsumer.load_dataset("data/wn18rr")
config = subsumer.TrainingConfig(dim=32, epochs=50, learning_rate=0.01)
trainer = subsumer.BoxEmbeddingTrainer.from_config(config)
result = trainer.fit(train, val_triples=val, num_entities=len(ents))
print(f"MRR: {result['mrr']:.3f}, Hits@10: {result['hits_at_10']:.3f}")

# Export learned embeddings as numpy arrays
entity_ids, min_bounds, max_bounds = trainer.export_embeddings()
# min_bounds and max_bounds are (n_entities, dim) numpy arrays
```

### Cone embeddings

```python
import subsumer

# ConE-style cone embeddings for DAG / partial-order relations
triples = [("animal", "hypernym", "dog"), ("animal", "hypernym", "cat")]
trainer, ids = subsumer.ConeEmbeddingTrainer.from_triples(triples)
result = trainer.fit(ids)
print(f"MRR: {result['mrr']:.3f}")

# Export learned cones (axes + apertures) as numpy arrays
entity_ids, axes, apertures = trainer.export_embeddings()
# axes and apertures are (n_entities, dim) numpy arrays
```

## License

MIT OR Apache-2.0
