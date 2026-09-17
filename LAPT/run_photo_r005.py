"""Run the current Photo R=0.05 HyDRO + LAPT experiment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULTS = [
    "--dataset", "photo", "--rate", "0.05",
    "--seeds", "1,2,3,4", "--stage", "all", "--gpu", "0",
    "--split-seed", "1", "--eval-seed", "1",
    "--validation-runs", "3", "--eval-runs", "10",
]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-u", str(ROOT / "run_pipeline.py"), *DEFAULTS, *sys.argv[1:]],
        cwd=ROOT,
        check=True,
    )
