"""Verify that the portable HyDRO/LAPT bundle can run on this computer."""

from __future__ import annotations

import importlib
from pathlib import Path
import platform
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent


def check_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"LAPT step: {STEP_ROOT}")
    print(f"Shared project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"System: {platform.platform()}")
    modules = (
        "torch", "torch_geometric", "torch_scatter", "torch_sparse", "geoopt",
        "numpy", "scipy", "sklearn", "networkx", "dgl", "ogb", "pygsp",
        "gensim", "pandas", "matplotlib", "tqdm", "click",
    )
    failed = False
    for name in modules:
        ok, detail = check_import(name)
        failed |= not ok
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA runtime: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            failed = True
    except Exception:
        pass
    files = (
        PROJECT_ROOT / "src" / "run" / "train_all.py",
        PROJECT_ROOT / "src" / "run" / "select_adaptive_rank_lapt.py",
        PROJECT_ROOT / "src" / "graphslim" / "condensation" / "paper_lapt.py",
        PROJECT_ROOT / "data" / "Computers" / "raw" / "amazon_electronics_computers.npz",
        PROJECT_ROOT / "data" / "Photo" / "raw" / "amazon_electronics_photo.npz",
        PROJECT_ROOT / "data" / "cora" / "raw" / "ind.cora.graph",
    )
    for path in files:
        ok = path.is_file() and path.stat().st_size > 0
        failed |= not ok
        print(f"[{'OK' if ok else 'FAIL'}] {path.relative_to(PROJECT_ROOT)}")
    print("\nResult: " + ("FAILED" if failed else "READY"))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
