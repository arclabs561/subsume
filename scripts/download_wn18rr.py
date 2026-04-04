#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://ndownloader.figshare.com/files/21844185"
DEFAULT_OUTPUT = Path("data/WN18RR")


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract_archive(archive: Path, dest: Path) -> Path:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive}")

    if (dest / "train.txt").exists():
        return dest

    train_files = list(dest.rglob("train.txt"))
    if not train_files:
        raise RuntimeError("Archive extracted but train.txt was not found")

    return train_files[0].parent


def promote_contents(source: Path, dest: Path) -> None:
    if source == dest:
        return

    for item in source.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the WN18RR dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    output = args.output
    if (
        (output / "train.txt").exists()
        and (output / "valid.txt").exists()
        and (output / "test.txt").exists()
    ):
        print(f"WN18RR already present at {output}")
        return 0

    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "wn18rr.zip"
        print(f"Downloading WN18RR from {args.url}")
        download(args.url, archive)

        extracted_root = extract_archive(archive, Path(tmpdir) / "extracted")
        promote_contents(extracted_root, output)

    for name in ("train.txt", "valid.txt", "test.txt"):
        if not (output / name).exists():
            raise RuntimeError(
                f"Missing expected file after extraction: {output / name}"
            )

    print(f"WN18RR ready at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
