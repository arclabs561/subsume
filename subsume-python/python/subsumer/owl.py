"""OWL ontology normalization to EL++ TSV format.

Requires mOWL (``pip install mowl-borg``) which depends on Java/JPype.
This module is optional -- the core subsumer package works without it.
"""

from __future__ import annotations

import random
from pathlib import Path


def normalize_owl(owl_path: str | Path) -> list[tuple[str, ...]]:
    """Normalize an OWL file to EL++ normal forms.

    Args:
        owl_path: Path to an OWL ontology file (.owl, .rdf, .xml).

    Returns:
        List of axiom tuples. Each tuple starts with the normal form tag
        (NF1, NF2, NF3, NF4, RI6, RI7, DISJ) followed by concept/role IRIs.

    Raises:
        ImportError: If mOWL or JPype is not installed.

    Example::

        from subsumer.owl import normalize_owl, write_tsv
        axioms = normalize_owl("ontology.owl")
        write_tsv(axioms, "train.tsv")

        # Then load and train:
        import subsumer
        trainer = subsumer.CandleElTrainer.from_axiom_file("train.tsv")
        trainer.fit()
    """
    from mowl.ontology.normalize import ELNormalizer  # type: ignore[import-untyped]
    from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore[import-untyped]
    import java.io  # type: ignore[import-untyped]

    manager = OWLManager.createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(
        java.io.File(str(owl_path))
    )
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

    # Role hierarchy from ontology axioms
    for ax_obj in ontology.getAxioms():
        ax_type = str(type(ax_obj).__name__)
        if "SubObjectPropertyOf" in ax_type:
            try:
                sub_prop = str(ax_obj.getSubProperty().toStringID())
                sup_prop = str(ax_obj.getSuperProperty().toStringID())
                axioms.append(("RI6", sub_prop, sup_prop))
            except Exception:
                pass
        elif "SubPropertyChainOf" in ax_type:
            try:
                chain = list(ax_obj.getPropertyChain())
                sup_prop = str(ax_obj.getSuperProperty().toStringID())
                if len(chain) == 2:
                    r = str(chain[0].toStringID())
                    s = str(chain[1].toStringID())
                    axioms.append(("RI7", r, s, sup_prop))
            except Exception:
                pass

    return axioms


def write_tsv(axioms: list[tuple[str, ...]], path: str | Path) -> dict[str, int]:
    """Write normalized axioms to a TSV file.

    Args:
        axioms: Output from ``normalize_owl()``.
        path: Output file path.

    Returns:
        Dict of NF type counts.
    """
    path = Path(path)
    with open(path, "w") as f:
        for ax in axioms:
            f.write("\t".join(ax) + "\n")
    counts: dict[str, int] = {}
    for ax in axioms:
        counts[ax[0]] = counts.get(ax[0], 0) + 1
    return counts


def normalize_and_split(
    owl_path: str | Path,
    output_dir: str | Path,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Normalize an OWL file and split into train/test TSV files.

    Args:
        owl_path: Path to OWL ontology file.
        output_dir: Directory for train.tsv and test.tsv.
        test_ratio: Fraction of axioms for the test set.
        seed: Random seed for reproducibility.

    Returns:
        Dict with "train" and "test" counts.
    """
    axioms = normalize_owl(owl_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(axioms)
    n_test = max(1, int(len(axioms) * test_ratio))

    test_counts = write_tsv(axioms[:n_test], out / "test.tsv")
    train_counts = write_tsv(axioms[n_test:], out / "train.tsv")

    return {
        "train": sum(train_counts.values()),
        "test": sum(test_counts.values()),
        "total": len(axioms),
    }
