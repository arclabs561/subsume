# /// script
# requires-python = ">=3.10"
# dependencies = ["subsumer>=0.8.2", "numpy>=1.20"]
# ///
"""Train EL++ ontology embeddings with CandleElTrainer (GPU-capable).

End-to-end pipeline:
  1. Load normalized EL++ axioms from TSV
  2. Train box embeddings with CandleElTrainer
  3. Evaluate per normal form (NF1-NF4)
  4. Save and reload a checkpoint
  5. Export embeddings for downstream use

Requires the candle-backend feature (included in published wheels).
For GPU acceleration, build with the cuda feature.

Usage:
  uv run examples/el_ontology.py
  uv run examples/el_ontology.py --axioms data/GALEN/train.tsv --test data/GALEN/test.tsv
  uv run examples/el_ontology.py --device cuda --epochs 500 --dim 200
"""

import argparse
from pathlib import Path

import numpy as np

import subsumer


def main() -> None:
    parser = argparse.ArgumentParser(description="EL++ ontology embedding")
    parser.add_argument(
        "--axioms",
        default="../data/go_subset/go_normalized.tsv",
        help="Training axiom TSV file",
    )
    parser.add_argument("--test", default=None, help="Test axiom TSV file")
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--neg-samples", type=int, default=2, help="Negatives per positive")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or metal")
    parser.add_argument("--save", default=None, help="Save checkpoint to this path")
    args = parser.parse_args()

    # -- 1. Load axioms and create trainer ------------------------------------
    axiom_path = str(Path(args.axioms))
    print(f"Loading axioms from {axiom_path}")

    trainer = subsumer.CandleElTrainer.from_axiom_file(
        axiom_path,
        dim=args.dim,
        device=args.device,
    )
    print(f"  {trainer}")

    # -- 2. Train -------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs on {args.device}...")
    losses = trainer.fit(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        neg_samples=args.neg_samples,
    )
    print(f"  Final loss: {losses[-1]:.4f}")

    # -- 3. Evaluate (if test set provided, or self-evaluate on train) --------
    test_path = args.test or axiom_path
    print(f"\nEvaluating on {test_path}...")
    metrics = trainer.evaluate(test_path)

    for nf, m in sorted(metrics.items()):
        print(f"  {nf}: H@1={m['h1']:.3f}  H@10={m['h10']:.3f}  MRR={m['mrr']:.3f}  (n={m['count']})")

    # -- 4. Save / load checkpoint -------------------------------------------
    ckpt = args.save or "/tmp/el_example.safetensors"
    trainer.save(ckpt)
    print(f"\nCheckpoint saved to {ckpt}")

    # Reload into a fresh trainer to verify
    trainer2 = subsumer.CandleElTrainer.from_axiom_file(
        axiom_path, dim=args.dim, device=args.device
    )
    trainer2.load(ckpt)
    c1 = trainer.concept_embeddings()["centers"]
    c2 = trainer2.concept_embeddings()["centers"]
    assert np.allclose(c1, c2), "Checkpoint mismatch"
    print("  Reload verified: embeddings match")

    # -- 5. Export embeddings -------------------------------------------------
    emb = trainer.concept_embeddings()
    print(f"\nConcept embeddings: {emb['centers'].shape}")
    print(f"  First 5 concepts: {emb['concept_names'][:5]}")

    roles = trainer.role_embeddings()
    print(f"Role embeddings: head={roles['head_centers'].shape}, tail={roles['tail_centers'].shape}")
    print(f"  Roles: {roles['role_names']}")

    bumps = trainer.bump_vectors()
    print(f"Bump vectors: {bumps['bumps'].shape}")

    # Show nearest neighbors for a concept (by center L2 distance)
    centers = emb["centers"]
    names = emb["concept_names"]
    if len(names) > 1:
        query_idx = 0
        dists = np.linalg.norm(centers - centers[query_idx], axis=1)
        nearest = np.argsort(dists)[1:6]
        print(f"\nNearest to '{names[query_idx]}':")
        for i in nearest:
            print(f"  {names[i]:40s} dist={dists[i]:.4f}")


if __name__ == "__main__":
    main()
