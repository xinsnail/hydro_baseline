"""Portable launcher for Step58-V1.1 audit and gradient-safe training."""

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


def _number(value: float) -> str:
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
    parser.add_argument("--stage", choices=("audit", "train"), default="train")
    parser.add_argument("--dataset", choices=tuple(DEFAULT_RATES), default="computers")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--max-scale", type=float, default=5.0)
    parser.add_argument("--audit-lambda-max", type=float, default=0.05)
    parser.add_argument("--pair-limit", type=int, default=2048)
    parser.add_argument("--syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--eval-interval", type=int, default=60)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.rate = DEFAULT_RATES[args.dataset] if args.rate is None else args.rate
    experiment_root = OUTPUT_ROOT / args.dataset / f"r_{_number(args.rate)}"

    if args.stage == "audit":
        output = experiment_root / "full_gradient_audit" / f"seed_{args.seed}"
        command = [
            sys.executable,
            "-u",
            str(STEP_ROOT / "run_trajectory_audit.py"),
            "-M", "hydro", "-D", args.dataset, "-R", _number(args.rate),
            "--setting", "trans", "-G", str(args.gpu), "-S", str(args.seed),
            "-E", str(args.epochs),
            "--load_path", str(DATA_ROOT),
            "--save_path", str(output / "unused_reduced_graph"),
            "--trajectory-output", str(output),
            "--trajectory-rho", str(args.rho),
            "--trajectory-lambda-max", str(args.audit_lambda_max),
            "--trajectory-pair-limit", str(args.pair_limit),
            "--trajectory-syn-pair-fraction", str(args.syn_pair_fraction),
        ]
        log_name = "trajectory_audit.log"
    else:
        output = (
            experiment_root
            / f"pilot_{args.epochs}"
            / f"gradient_safe_rho_{_number(args.rho)}"
            / f"seed_{args.seed}"
        )
        command = [
            sys.executable,
            "-u",
            str(STEP_ROOT / "run_train_v1_1.py"),
            "-M", "hydro", "-D", args.dataset, "-R", _number(args.rate),
            "--setting", "trans", "-G", str(args.gpu), "-S", str(args.seed),
            "-E", str(args.epochs),
            "--eval_interval", str(args.eval_interval),
            "--run_eval", str(args.eval_runs),
            "--lp_hidden", "128", "--lp_lr", "0.001",
            "--lp_epochs", "100",
            "--lp_syn_thresholds", "q55,q65,q75,q85,q95",
            "--lp_val_interval", "10",
            "--lp_decision_threshold", "0.5",
            "--lp_neg_ratio", "1.0",
            "--lp_f1_average", "macro",
            "--load_path", str(DATA_ROOT),
            "--save_path", str(output),
            "--v1-1-output", str(output),
            "--v1-1-rho", str(args.rho),
            "--v1-1-max-scale", str(args.max_scale),
            "--v1-1-pair-limit", str(args.pair_limit),
            "--v1-1-syn-pair-fraction", str(args.syn_pair_fraction),
        ]
        log_name = "v1_1_training.log"

    print(f"Step: {STEP_ROOT}")
    print(f"Shared source: {SOURCE_ROOT}")
    print(f"Shared data: {DATA_ROOT}")
    print(f"Output: {output}")
    print("Command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / log_name).open(
        "w", encoding="utf-8", errors="replace"
    ) as log:
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
