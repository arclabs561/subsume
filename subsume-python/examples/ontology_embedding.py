# /// script
# requires-python = ">=3.10"
# dependencies = ["subsumer>=0.7.1", "rdflib>=7.0"]
# ///
"""Embed an OWL-like ontology built with rdflib into box embeddings.

Demonstrates the biomedical/ontology use case:
  1. Build a small taxonomy using rdflib with rdfs:subClassOf triples
  2. Extract triples from the RDF graph
  3. Train box embeddings with subsumer
  4. Verify learned containment matches the ontology hierarchy
  5. Predict new subClassOf relations (ontology completion)
"""

from rdflib import Graph, Namespace, RDF, RDFS, URIRef

import subsumer


def build_ontology() -> Graph:
    """Build a small animal taxonomy as an RDF graph.

    Hierarchy:
        Animal
        +-- Mammal
        |   +-- Dog
        |   +-- Cat
        +-- Bird
            +-- Eagle
    """
    EX = Namespace("http://example.org/ontology#")
    g = Graph()

    # Declare classes
    classes = ["Animal", "Mammal", "Bird", "Dog", "Cat", "Eagle"]
    for name in classes:
        g.add((EX[name], RDF.type, RDFS.Class))

    # rdfs:subClassOf means "child subClassOf parent"
    g.add((EX.Mammal, RDFS.subClassOf, EX.Animal))
    g.add((EX.Bird, RDFS.subClassOf, EX.Animal))
    g.add((EX.Dog, RDFS.subClassOf, EX.Mammal))
    g.add((EX.Cat, RDFS.subClassOf, EX.Mammal))
    g.add((EX.Eagle, RDFS.subClassOf, EX.Bird))

    return g


def extract_subclass_triples(g: Graph) -> list[tuple[str, str, str]]:
    """Extract (child, subClassOf, parent) triples from the graph.

    Returns short names (fragment after #) for readability.
    """
    triples = []
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        child = str(s).rsplit("#", 1)[-1]
        parent = str(o).rsplit("#", 1)[-1]
        triples.append((child, "subClassOf", parent))
    return triples


def main() -> None:
    # -- 1. Build ontology ------------------------------------------------
    g = build_ontology()
    print(f"RDF graph: {len(g)} triples total")

    # -- 2. Extract subClassOf triples ------------------------------------
    triples = extract_subclass_triples(g)
    print(f"Extracted {len(triples)} subClassOf triples:")
    for h, r, t in triples:
        print(f"  {h} {r} {t}")

    # -- 3. Train box embeddings ------------------------------------------
    # rdfs:subClassOf gives (child, subClassOf, parent), but subsumer expects
    # (head_contains_tail) ordering, i.e. (parent, rel, child). Setting
    # reverse=True swaps head and tail internally.
    config = subsumer.TrainingConfig(
        dim=16,
        learning_rate=0.01,
        epochs=600,
        batch_size=5,
        margin=1.0,
        negative_samples=3,
        gumbel_beta=3.0,
        gumbel_beta_final=15.0,
        warmup_epochs=30,
        early_stopping_patience=None,
        regularization=0.0001,
    )
    trainer, train_ids = subsumer.BoxEmbeddingTrainer.from_triples(
        triples, config=config, reverse=True
    )
    result = trainer.fit(train_ids)

    final_loss = result["loss_history"][-1]
    print(f"\nTraining complete (final loss: {final_loss:.4f})")

    # -- 4. Verify containment -------------------------------------------
    # Build a name -> id map from the entity list
    entity_names = trainer.entity_names()
    name_to_id = {name: i for i, name in enumerate(entity_names)}

    print("\nContainment probabilities (P(child inside parent)):")
    pairs = [
        ("Dog", "Mammal"),
        ("Cat", "Mammal"),
        ("Mammal", "Animal"),
        ("Eagle", "Bird"),
        ("Bird", "Animal"),
        ("Dog", "Animal"),  # transitive: Dog inside Animal
    ]
    for child, parent in pairs:
        child_box = trainer.get_box(name_to_id[child])
        parent_box = trainer.get_box(name_to_id[parent])
        prob = parent_box.containment_prob(child_box)
        print(f"  P({child} in {parent}) = {prob:.3f}")

    # Negative examples: unrelated classes should have low containment
    print("\nNegative containment (should be low):")
    neg_pairs = [
        ("Eagle", "Mammal"),
        ("Dog", "Bird"),
    ]
    for child, parent in neg_pairs:
        child_box = trainer.get_box(name_to_id[child])
        parent_box = trainer.get_box(name_to_id[parent])
        prob = parent_box.containment_prob(child_box)
        print(f"  P({child} in {parent}) = {prob:.3f}")

    # -- 5. Ontology completion -------------------------------------------
    # Predict whether a new subClassOf relation holds by checking containment.
    # A high P(candidate_child inside candidate_parent) suggests a missing
    # subClassOf edge.
    print("\nOntology completion -- rank candidate parents for 'Dog':")
    dog_id = name_to_id["Dog"]
    dog_box = trainer.get_box(dog_id)

    candidates = []
    for name, eid in name_to_id.items():
        if name == "Dog":
            continue
        candidate_box = trainer.get_box(eid)
        prob = candidate_box.containment_prob(dog_box)
        candidates.append((name, prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    print("  Candidate parent     P(Dog inside candidate)")
    for name, prob in candidates:
        marker = "<--" if name in ("Mammal", "Animal") else ""
        print(f"  {name:<20s} {prob:.3f}  {marker}")


if __name__ == "__main__":
    main()
