# subsumer

Use boxes and cones to represent containment in a knowledge graph. The Python package exposes the Rust trainers and exports learned coordinates as NumPy arrays.

## Install

With Python 3.9 or newer, install the published package into your environment:

```sh
python -m pip install subsumer
```

To build from a source checkout, install Rust and run `python -m pip install ./subsume-python` from the repository root in a virtual environment.

## Measure containment

The receiver is the containing box. Here the inner square fits entirely inside the outer square and covers one quarter of its area:

```python
from subsumer import BoxEmbedding

outer = BoxEmbedding([0.0, 0.0], [2.0, 2.0], 1.0)
inner = BoxEmbedding([0.0, 0.0], [1.0, 1.0], 1.0)
assert outer.containment_prob(inner) == 1.0
assert inner.containment_prob(outer) == 0.25
```

## Train on triples

Give the trainer triples in parent-to-child order. For data in child-to-parent order, pass `reverse=True` to `from_triples`. This example fits two edges and evaluates the implied third edge using the same entity IDs:

```python
from subsumer import BoxEmbeddingTrainer, TrainingConfig

train = [("animal", "contains", "mammal"), ("mammal", "contains", "dog")]
held_out = [("animal", "contains", "dog")]
config = TrainingConfig(dim=8, epochs=50, learning_rate=0.01)
trainer, ids = BoxEmbeddingTrainer.from_triples(train + held_out, config=config)
trainer.fit(ids[:len(train)])
metrics = trainer.evaluate(ids[len(train):])
print(f"Held-out mean reciprocal rank: {metrics['mrr']:.3f}")

head, relation, tail = ids[-1]
score = trainer.predict(head, relation, tail)
scores = trainer.score_tails(head, relation)
entity_ids, lower, upper = trainer.export_embeddings()
assert lower.shape == upper.shape == (3, 8)
```

To train cones on the same two edges, use the cone trainer. Both exported arrays have one row per entity:

```python
from subsumer import ConeEmbeddingTrainer, TrainingConfig

triples = [("animal", "contains", "mammal"), ("mammal", "contains", "dog")]
trainer, ids = ConeEmbeddingTrainer.from_triples(triples, config=TrainingConfig(dim=8, epochs=50))
trainer.fit(ids)
entity_ids, axes, apertures = trainer.export_embeddings()
assert axes.shape == apertures.shape == (3, 8)
```

For WN18RR, run `python3 scripts/download_wn18rr.py` from the repository root. Then `subsumer.load_dataset("data/WN18RR")` returns the training, validation and test triples followed by the entity and relation names. Pass `reverse=True` when the relations you want to model point from child to parent.

## EL++ ontology embeddings

Compute the inclusion loss for a box centered inside a larger box, then check whether the intersection of two intervals fits inside a third:

```python
import subsumer

assert subsumer.el_inclusion_loss(
    center_a=[0.0, 0.0], offset_a=[0.5, 0.5],
    center_b=[0.0, 0.0], offset_b=[2.0, 2.0],
) == 0.0

assert subsumer.el_intersection_loss(
    center_c1=[0.0], offset_c1=[2.0],
    center_c2=[1.0], offset_c2=[2.0],
    center_d=[0.5], offset_d=[2.0],
) == 0.0
```

From the repository root, load the bundled Gene Ontology subset:

```python
import subsumer

axioms = subsumer.load_el_axioms("data/go_subset/go_normalized.tsv")
print(f"{axioms['num_classes']} classes, {len(axioms['nf2'])} subclass axioms")
```

For a complete RDF example, run `uv run subsume-python/examples/ontology_embedding.py` from the repository root. It builds a small taxonomy, trains boxes and ranks candidate parents for “Dog”. [RDF example](https://github.com/arclabs561/subsume/blob/main/subsume-python/examples/ontology_embedding.py)

## License

MIT or Apache-2.0.
