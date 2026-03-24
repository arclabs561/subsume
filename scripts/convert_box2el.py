#!/usr/bin/env -S uv run --with numpy --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Convert Box2EL numpy datasets to subsume TSV format.

Usage:
    uv run scripts/convert_box2el.py data/box2el/data/GALEN/prediction data/GALEN
    uv run scripts/convert_box2el.py data/box2el/data/GO/prediction data/GO_full
    uv run scripts/convert_box2el.py data/box2el/data/ANATOMY/prediction data/ANATOMY
"""

import json
import sys
from pathlib import Path

import numpy as np


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class and relation mappings (index -> name)
    classes: dict[str, int] = json.loads((input_dir / "classes.json").read_text())
    relations: dict[str, int] = json.loads((input_dir / "relations.json").read_text())

    # Invert: index -> name
    idx_to_class = {v: k for k, v in classes.items()}
    idx_to_rel = {v: k for k, v in relations.items()}

    for split in ["train", "val", "test"]:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        lines = []

        # Box2EL NF1: C ⊑ D (subsumption, 2 cols) -- our NF2
        nf1_path = split_dir / "nf1.npy"
        if nf1_path.exists():
            nf1 = np.load(nf1_path)
            for row in nf1:
                c, d = int(row[0]), int(row[1])
                lines.append(f"NF2\t{idx_to_class[c]}\t{idx_to_class[d]}")

        # Box2EL NF2: C1 ⊓ C2 ⊑ D (conjunction, 3 cols) -- our NF1
        nf2_path = split_dir / "nf2.npy"
        if nf2_path.exists():
            nf2 = np.load(nf2_path)
            for row in nf2:
                c1, c2, d = int(row[0]), int(row[1]), int(row[2])
                lines.append(f"NF1\t{idx_to_class[c1]}\t{idx_to_class[c2]}\t{idx_to_class[d]}")

        # NF3: C ⊑ ∃r.D -- (c, r, d)
        nf3_path = split_dir / "nf3.npy"
        if nf3_path.exists():
            nf3 = np.load(nf3_path)
            for row in nf3:
                c, r, d = int(row[0]), int(row[1]), int(row[2])
                lines.append(f"NF3\t{idx_to_class[c]}\t{idx_to_rel[r]}\t{idx_to_class[d]}")

        # NF4: ∃r.C ⊑ D -- (r, c, d)
        nf4_path = split_dir / "nf4.npy"
        if nf4_path.exists():
            nf4 = np.load(nf4_path)
            for row in nf4:
                r, c, d = int(row[0]), int(row[1]), int(row[2])
                lines.append(f"NF4\t{idx_to_rel[r]}\t{idx_to_class[c]}\t{idx_to_class[d]}")

        # Role inclusion and chains
        for fname, prefix in [("role_inclusion.npy", "RI6"), ("role_chain.npy", "RI7")]:
            path = split_dir / fname
            if path.exists():
                arr = np.load(path)
                if arr.ndim > 0 and len(arr) > 0:
                    for row in arr:
                        if prefix == "RI6":
                            r, s = int(row[0]), int(row[1])
                            lines.append(f"RI6\t{idx_to_rel[r]}\t{idx_to_rel[s]}")
                        else:
                            r, s, t = int(row[0]), int(row[1]), int(row[2])
                            lines.append(f"RI7\t{idx_to_rel[r]}\t{idx_to_rel[s]}\t{idx_to_rel[t]}")

        # Disjointness
        disj_path = split_dir / "disjoint.npy"
        if disj_path.exists():
            disj = np.load(disj_path)
            if disj.ndim > 0 and len(disj) > 0:
                for row in disj:
                    a, b = int(row[0]), int(row[1])
                    lines.append(f"DISJ\t{idx_to_class[a]}\t{idx_to_class[b]}")

        out_file = output_dir / f"{split}.tsv"
        out_file.write_text("\n".join(lines) + "\n")
        print(f"  {split}: {len(lines)} axioms -> {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_box2el.py <input_dir> <output_dir>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
