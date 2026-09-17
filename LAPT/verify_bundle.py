#!/usr/bin/env python3
"""Create or verify the portable LAPT bundle SHA-256 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
MANIFEST = STEP_ROOT / "MANIFEST_SHA256.csv"
INCLUDED_ROOTS = (PROJECT_ROOT / "data", PROJECT_ROOT / "src", STEP_ROOT)
EXCLUDED_DIRS = {"outputs", "__pycache__", ".git"}
EXCLUDED_FILES = {MANIFEST.name}


def included_files() -> list[Path]:
    files: list[Path] = []
    for included_root in INCLUDED_ROOTS:
        if not included_root.exists():
            continue
        for path in included_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(PROJECT_ROOT)
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if path.name in EXCLUDED_FILES or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(
        set(files),
        key=lambda value: value.relative_to(PROJECT_ROOT).as_posix().lower(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_manifest() -> int:
    rows = []
    for path in included_files():
        rows.append((path.relative_to(PROJECT_ROOT).as_posix(), path.stat().st_size, sha256(path)))
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)
    print(f"Manifest written: {MANIFEST}")
    print(f"Files recorded: {len(rows)}")
    return 0


def verify_manifest() -> int:
    if not MANIFEST.exists():
        print(f"[FAIL] Missing manifest: {MANIFEST}")
        return 2
    failures: list[str] = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = PROJECT_ROOT / Path(row["path"])
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(row["size_bytes"]):
            failures.append(f"size mismatch: {row['path']}")
            continue
        if sha256(path) != row["sha256"].upper():
            failures.append(f"hash mismatch: {row['path']}")
    if failures:
        print("Bundle verification: FAILED")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("Bundle verification: PASS")
    print(f"Files verified: {len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="create or replace MANIFEST_SHA256.csv; default action is verification",
    )
    args = parser.parse_args()
    return write_manifest() if args.write else verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
