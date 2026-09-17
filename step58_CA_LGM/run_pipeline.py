"""Portable launcher for the Step58 CA-LGM gradient audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / STEP_ROOT.name
DEFAULT_RATES = {"computers": 0.005, "photo": 0.01, "cora": 0.5}


def _rate(value: float) -> str:
    return f"{float(value):.12g}"


def _safe_print(line: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(
        line.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        ),
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULT_RATES), default="computers")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--adj-points", default="0,50,100")
    parser.add_argument("--feat-points", default="10,60,110")
    parser.add_argument("--pair-limit", type=int, default=2048)
    parser.add_argument("--syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--distance-alpha", type=float, default=0.50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.rate = DEFAULT_RATES[args.dataset] if args.rate is None else args.rate
    output = (
        OUTPUT_ROOT
        / args.dataset
        / f"r_{_rate(args.rate)}"
        / "gradient_audit"
        / f"seed_{args.seed}"
    )
    command = [
        sys.executable,
        "-u",
        str(STEP_ROOT / "run_gradient_audit.py"),
        "-M",
        "hydro",
        "-D",
        args.dataset,
        "-R",
        _rate(args.rate),
        "--setting",
        "trans",
        "-G",
        str(args.gpu),
        "-S",
        str(args.seed),
        "-E",
        str(args.epochs),
        "--load_path",
        str(DATA_ROOT),
        "--save_path",
        str(output / "unused_reduced_graph"),
        "--step58-output",
        str(output),
        "--step58-adj-points",
        args.adj_points,
        "--step58-feat-points",
        args.feat_points,
        "--step58-pair-limit",
        str(args.pair_limit),
        "--step58-syn-pair-fraction",
        str(args.syn_pair_fraction),
        "--step58-distance-alpha",
        str(args.distance_alpha),
    ]
    print(f"Step: {STEP_ROOT}")
    print(f"Shared source: {SOURCE_ROOT}")
    print(f"Shared data: {DATA_ROOT}")
    print(f"Output: {output}")
    print("Command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "gradient_audit.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=SOURCE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _safe_print(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
