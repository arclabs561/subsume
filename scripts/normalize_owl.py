# /// script
# /// requires-python = ">=3.9"
# /// dependencies = ["mowl-borg>=1.0", "jpype1"]
# ///
"""Normalize an OWL ontology to EL++ normal forms (TSV).

Produces the TSV format consumed by subsume's el_dataset module:
  NF1  C1  C2  D      (C1 ⊓ C2 ⊑ D)
  NF2  C   D          (C ⊑ D)
  NF3  C   r   D      (C ⊑ ∃r.D)
  NF4  r   C   D      (∃r.C ⊑ D)
  RI6  r   s           (r ⊑ s)
  RI7  r   s   t       (r ∘ s ⊑ t)
  DISJ C   D          (C ⊓ D ⊑ ⊥)

Usage:
  uv run scripts/normalize_owl.py ontology.owl -o train.tsv
  uv run scripts/normalize_owl.py ontology.owl --split 0.1 -o data/MYONTO/
"""

import argparse
import random
import sys
from pathlib import Path


def normalize(owl_path: str) -> list[tuple[str, ...]]:
    """Normalize an OWL file to EL++ normal forms via mOWL/jcel."""
    from mowl.ontology.normalize import ELNormalizer
    from org.semanticweb.owlapi.apibinding import OWLManager
    import java.io

    manager = OWLManager.createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(java.io.File(owl_path))
    normalizer = ELNormalizer()
    gcis = normalizer.normalize(ontology)

    axioms: list[tuple[str, ...]] = []

    # GCI0 -> NF2: C ⊑ D
    for ax in gcis.get("gci0", []):
        sub = str(ax.getSubClass().toStringID())
        sup = str(ax.getSuperClass().toStringID())
        axioms.append(("NF2", sub, sup))

    # GCI1 -> NF1: C1 ⊓ C2 ⊑ D
    for ax in gcis.get("gci1", []):
        ops = list(ax.getSubClass().getOperands())
        if len(ops) == 2:
            c1 = str(ops[0].toStringID())
            c2 = str(ops[1].toStringID())
            d = str(ax.getSuperClass().toStringID())
            axioms.append(("NF1", c1, c2, d))

    # GCI2 -> NF3: C ⊑ ∃r.D
    for ax in gcis.get("gci2", []):
        sub = str(ax.getSubClass().toStringID())
        svf = ax.getSuperClass()
        role = str(svf.getProperty().toStringID())
        filler = str(svf.getFiller().toStringID())
        axioms.append(("NF3", sub, role, filler))

    # GCI3 -> NF4: ∃r.C ⊑ D
    for ax in gcis.get("gci3", []):
        svf = ax.getSubClass()
        role = str(svf.getProperty().toStringID())
        filler = str(svf.getFiller().toStringID())
        sup = str(ax.getSuperClass().toStringID())
        axioms.append(("NF4", role, filler, sup))

    # GCI1_BOT -> DISJ: C ⊓ D ⊑ ⊥
    for ax in gcis.get("gci1_bot", []):
        ops = list(ax.getSubClass().getOperands())
        if len(ops) == 2:
            c1 = str(ops[0].toStringID())
            c2 = str(ops[1].toStringID())
            axioms.append(("DISJ", c1, c2))

    # Extract role hierarchy axioms directly from ontology
    for ax_type in ontology.getAxioms():
        ax_str = str(type(ax_type).__name__)
        if "SubObjectPropertyOf" in ax_str:
            try:
                sub_prop = str(ax_type.getSubProperty().toStringID())
                sup_prop = str(ax_type.getSuperProperty().toStringID())
                axioms.append(("RI6", sub_prop, sup_prop))
            except Exception:
                pass
        elif "SubPropertyChainOf" in ax_str:
            try:
                chain = list(ax_type.getPropertyChain())
                sup_prop = str(ax_type.getSuperProperty().toStringID())
                if len(chain) == 2:
                    r = str(chain[0].toStringID())
                    s = str(chain[1].toStringID())
                    axioms.append(("RI7", r, s, sup_prop))
            except Exception:
                pass

    return axioms


def write_tsv(axioms: list[tuple[str, ...]], path: Path) -> None:
    """Write axioms to TSV file."""
    with open(path, "w") as f:
        for ax in axioms:
            f.write("\t".join(ax) + "\n")
    nf_counts = {}
    for ax in axioms:
        nf_counts[ax[0]] = nf_counts.get(ax[0], 0) + 1
    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(nf_counts.items()))
    print(f"Wrote {len(axioms)} axioms to {path} ({counts_str})")


def main():
    parser = argparse.ArgumentParser(description="Normalize OWL to EL++ TSV")
    parser.add_argument("owl_file", help="Path to OWL ontology file")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file or directory (with --split)")
    parser.add_argument("--split", type=float, default=0.0,
                        help="Test split ratio (e.g. 0.1 for 90/10 train/test). Output becomes a directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    print(f"Normalizing {args.owl_file}...")
    axioms = normalize(args.owl_file)
    print(f"Total: {len(axioms)} normalized axioms")

    if args.split > 0:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        random.shuffle(axioms)
        n_test = max(1, int(len(axioms) * args.split))
        test = axioms[:n_test]
        train = axioms[n_test:]
        write_tsv(train, out_dir / "train.tsv")
        write_tsv(test, out_dir / "test.tsv")
    else:
        write_tsv(axioms, Path(args.output))


if __name__ == "__main__":
    main()
