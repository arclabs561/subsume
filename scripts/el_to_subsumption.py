#!/usr/bin/env python3
"""Convert EL++ normalized axioms to WN18RR-format atomic-subsumption triples.

The EL++ datasets (GALEN/GO/ANATOMY, in `data/<NAME>/{train,val,test}.tsv`) store
one type-tagged axiom per tab-separated line. The `NF2` rows are atomic concept
subsumptions `C sqsubseteq D` (two concepts). This extracts those and emits
WN18RR-format triples so the generic region-geometry trainers and the
`wn18rr_geometry_comparison_burn` harness can be evaluated on subsumption
(subsume's actual niche) rather than only on general link prediction.

Direction: `C sqsubseteq D` means C is a subclass of D, so D (general) contains
C (specific). subsume's convention is head-contains-tail, so each triple is
`D<TAB>subsumes<TAB>C`. Only NF2 rows are used; NF1/NF3/NF4 involve roles or
intersections the plain (head, relation, tail) trainers cannot consume.

Usage:
    python3 scripts/el_to_subsumption.py            # converts GALEN/GO/ANATOMY under data/
    python3 scripts/el_to_subsumption.py data       # explicit data root

Writes `data/<NAME>_sub/{train,valid,test}.txt`. The `data/` tree is gitignored;
these are derived files, regenerate on demand.
"""
import sys
from pathlib import Path


def localname(uri: str) -> str:
    """Strip an OWL/OBO URI down to its local concept name."""
    s = uri.strip().strip("<>")
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def convert(src_dir: Path, dst_dir: Path) -> "dict[str, int]":
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for src_split, dst_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        src = src_dir / f"{src_split}.tsv"
        if not src.exists():
            continue
        out = []
        for line in src.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3 or parts[0] != "NF2":
                continue
            c, d = localname(parts[1]), localname(parts[2])
            if c and d and c != d:
                out.append(f"{d}\tsubsumes\t{c}")
        (dst_dir / f"{dst_split}.txt").write_text("\n".join(out) + "\n")
        counts[dst_split] = len(out)
    return counts


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    any_done = False
    for name in ("GALEN", "GO", "ANATOMY"):
        src = base / name
        if src.exists():
            counts = convert(src, base / f"{name}_sub")
            print(f"{name} -> {name}_sub: {counts}")
            any_done = True
    if not any_done:
        print(f"No EL++ datasets found under {base}/ (expected GALEN/GO/ANATOMY).")
        sys.exit(1)


if __name__ == "__main__":
    main()
