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

box_a = subsumer.BoxEmbedding([0.0, 0.0], [1.0, 1.0], 1.0)
box_b = subsumer.BoxEmbedding([0.2, 0.2], [0.8, 0.8], 1.0)
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

# Score individual triples after training
score = trainer.predict(head_id=0, relation_id=0, tail_id=1)

# Score all entities as tails
scores = trainer.score_tails(head_id=0, relation_id=0)

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

# Evaluate on separate test set
test_result = trainer.evaluate(test_triples)

# Export learned cones (axes + apertures) as numpy arrays
entity_ids, axes, apertures = trainer.export_embeddings()
# axes and apertures are (n_entities, dim) numpy arrays
```

### EL++ ontology embedding

```python
import subsumer

# Load normalized EL++ axioms (GALEN, Gene Ontology, Anatomy formats)
axioms = subsumer.load_el_axioms("data/go_normalized.tsv")
print(f"Classes: {axioms['num_classes']}, Roles: {axioms['num_roles']}")
print(f"NF2 (C ⊑ D): {len(axioms['nf2'])} axioms")

# Compute inclusion loss: how much box A fails to fit inside box B
loss = subsumer.el_inclusion_loss(
    center_a=[0.0, 0.0], offset_a=[0.5, 0.5],
    center_b=[0.0, 0.0], offset_b=[2.0, 2.0],
)
print(f"Inclusion loss: {loss:.4f}")  # 0.0 (A inside B)

# Compute intersection loss: C1 AND C2 should be inside D
loss = subsumer.el_intersection_loss(
    center_c1=[0.0], offset_c1=[2.0],
    center_c2=[1.0], offset_c2=[2.0],
    center_d=[0.5], offset_d=[2.0],
)
print(f"Intersection loss: {loss:.4f}")  # 0.0 (intersection fits)
```

## License

MIT OR Apache-2.0
