# /// script
# requires-python = ">=3.10"
# dependencies = ["subsumer>=0.7.1", "pykeen>=1.10", "torch"]
# ///
"""PyKEEN interop: train subsumer box embeddings on a PyKEEN dataset.

Loads the Nations dataset (~200 triples) from PyKEEN, trains box embeddings
with subsumer, then trains PyKEEN's BoxE on the same data for comparison.

Usage:
    uv run subsume-python/examples/pykeen_interop.py
"""

from __future__ import annotations

import time


def extract_triples_from_pykeen(
    triples_factory,
) -> list[tuple[str, str, str]]:
    """Extract (head, relation, tail) string triples from a PyKEEN TriplesFactory."""
    triples = []
    for row in triples_factory.triples:
        h, r, t = row[0], row[1], row[2]
        triples.append((str(h), str(r), str(t)))
    return triples


def train_subsumer(
    train_triples: list[tuple[str, str, str]],
    test_triples: list[tuple[str, str, str]],
) -> dict:
    """Train subsumer box embeddings and evaluate."""
    import subsumer

    # Merge all triples to build a shared vocabulary, then split back.
    all_triples = train_triples + test_triples
    trainer, all_ids = subsumer.BoxEmbeddingTrainer.from_triples(all_triples)

    train_ids = all_ids[: len(train_triples)]
    test_ids = all_ids[len(train_triples) :]

    num_entities = max(max(h, t) for h, _, t in all_ids) + 1

    # Reconfigure with a custom config for better training.
    config = subsumer.TrainingConfig(
        dim=32,
        learning_rate=0.01,
        epochs=100,
        batch_size=128,
        margin=2.0,
        negative_samples=5,
        gumbel_beta=10.0,
        gumbel_beta_final=50.0,
        warmup_epochs=5,
        early_stopping_patience=20,
    )
    trainer, all_ids = subsumer.BoxEmbeddingTrainer.from_triples(
        all_triples, config=config
    )
    train_ids = all_ids[: len(train_triples)]
    test_ids = all_ids[len(train_triples) :]

    # Train
    t0 = time.time()
    result = trainer.fit(train_ids)
    train_time = time.time() - t0

    # Evaluate on test set
    eval_result = trainer.evaluate(test_ids, num_entities)

    return {
        "mrr": eval_result["mrr"],
        "hits_at_1": eval_result["hits_at_1"],
        "hits_at_3": eval_result["hits_at_3"],
        "hits_at_10": eval_result["hits_at_10"],
        "mean_rank": eval_result["mean_rank"],
        "train_time_s": train_time,
        "final_loss": result["loss_history"][-1] if result["loss_history"] else None,
    }


def train_pykeen_boxe(training, testing) -> dict:
    """Train PyKEEN BoxE and evaluate."""
    from pykeen.pipeline import pipeline

    t0 = time.time()
    result = pipeline(
        training=training,
        testing=testing,
        model="BoxE",
        model_kwargs={"embedding_dim": 32},
        training_kwargs={"num_epochs": 100, "batch_size": 128},
        optimizer_kwargs={"lr": 0.01},
        evaluation_kwargs={"batch_size": 128},
        random_seed=42,
    )
    train_time = time.time() - t0

    # Extract metrics via get_metric (stable across PyKEEN versions).
    mr = result.metric_results
    return {
        "mrr": mr.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        "hits_at_1": mr.get_metric("both.realistic.hits_at_1"),
        "hits_at_3": mr.get_metric("both.realistic.hits_at_3"),
        "hits_at_10": mr.get_metric("both.realistic.hits_at_10"),
        "mean_rank": mr.get_metric("both.realistic.arithmetic_mean_rank"),
        "train_time_s": train_time,
    }


def main():
    # -- Load PyKEEN dataset --
    try:
        from pykeen.datasets import Nations
    except ImportError:
        print("PyKEEN not installed. Install with: pip install pykeen")
        print("Skipping PyKEEN comparison. Running subsumer-only demo.")
        run_subsumer_only()
        return

    print("Loading Nations dataset from PyKEEN...")
    dataset = Nations()
    print(
        f"  Train: {dataset.training.num_triples} triples, "
        f"Test: {dataset.testing.num_triples} triples"
    )
    print(
        f"  Entities: {dataset.training.num_entities}, "
        f"Relations: {dataset.training.num_relations}"
    )

    train_triples = extract_triples_from_pykeen(dataset.training)
    test_triples = extract_triples_from_pykeen(dataset.testing)

    # -- Train subsumer --
    print("\n--- subsumer (Rust box embeddings) ---")
    subsumer_results = train_subsumer(train_triples, test_triples)
    print(f"  MRR:       {subsumer_results['mrr']:.4f}")
    print(f"  Hits@1:    {subsumer_results['hits_at_1']:.4f}")
    print(f"  Hits@3:    {subsumer_results['hits_at_3']:.4f}")
    print(f"  Hits@10:   {subsumer_results['hits_at_10']:.4f}")
    print(f"  Mean Rank: {subsumer_results['mean_rank']:.1f}")
    print(f"  Time:      {subsumer_results['train_time_s']:.2f}s")

    # -- Train PyKEEN BoxE --
    print("\n--- PyKEEN BoxE (PyTorch) ---")
    try:
        pykeen_results = train_pykeen_boxe(dataset.training, dataset.testing)
        print(f"  MRR:       {pykeen_results['mrr']:.4f}")
        print(f"  Hits@1:    {pykeen_results['hits_at_1']:.4f}")
        print(f"  Hits@3:    {pykeen_results['hits_at_3']:.4f}")
        print(f"  Hits@10:   {pykeen_results['hits_at_10']:.4f}")
        print(f"  Mean Rank: {pykeen_results['mean_rank']:.1f}")
        print(f"  Time:      {pykeen_results['train_time_s']:.2f}s")
    except Exception as e:
        print(f"  PyKEEN training failed: {e}")
        pykeen_results = None

    # -- Side-by-side comparison --
    if pykeen_results:
        print("\n--- Comparison ---")
        print(f"{'Metric':<12} {'subsumer':>10} {'PyKEEN BoxE':>12}")
        print("-" * 36)
        for key in ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]:
            s = subsumer_results[key]
            p = pykeen_results[key]
            print(f"{key:<12} {s:>10.4f} {p:>12.4f}")
        print(
            f"{'mean_rank':<12} {subsumer_results['mean_rank']:>10.1f} "
            f"{pykeen_results['mean_rank']:>12.1f}"
        )
        print(
            f"{'time (s)':<12} {subsumer_results['train_time_s']:>10.2f} "
            f"{pykeen_results['train_time_s']:>12.2f}"
        )


def run_subsumer_only():
    """Fallback: demonstrate subsumer without PyKEEN."""
    import subsumer

    # Minimal Nations-like triples for demo purposes.
    triples = [
        ("usa", "exports_to", "uk"),
        ("usa", "exports_to", "france"),
        ("uk", "exports_to", "france"),
        ("france", "exports_to", "usa"),
        ("usa", "ally_of", "uk"),
        ("uk", "ally_of", "france"),
    ]

    config = subsumer.TrainingConfig(dim=16, epochs=50, learning_rate=0.01)
    trainer, ids = subsumer.BoxEmbeddingTrainer.from_triples(triples, config=config)
    num_entities = max(max(h, t) for h, _, t in ids) + 1

    result = trainer.fit(ids)
    eval_result = trainer.evaluate(ids, num_entities)

    print("\n--- subsumer results (demo triples) ---")
    print(f"  MRR:       {eval_result['mrr']:.4f}")
    print(f"  Hits@10:   {eval_result['hits_at_10']:.4f}")
    print(f"  Mean Rank: {eval_result['mean_rank']:.1f}")
    print(f"  Final loss: {result['loss_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
