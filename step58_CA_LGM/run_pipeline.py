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
    parser.add_argument(
        "--stage",
        choices=("audit", "baseline", "v1", "pilot"),
        default="audit",
    )
    parser.add_argument("--lp-weight", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=60)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.rate = DEFAULT_RATES[args.dataset] if args.rate is None else args.rate
    experiment_root = OUTPUT_ROOT / args.dataset / f"r_{_rate(args.rate)}"
    audit_output = experiment_root / "gradient_audit" / f"seed_{args.seed}"
    audit_command = [
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
        str(audit_output / "unused_reduced_graph"),
        "--step58-output",
        str(audit_output),
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
    # Condensation seeds are independent experimental units.  Keep every
    # paired baseline/V1 run in its own directory so later seeds cannot
    # overwrite logs, diagnostics, or selected condensed graphs from seed 1.
    paired_output = (
        experiment_root / f"pilot_{args.epochs}" / f"seed_{args.seed}"
    )
    baseline_output = paired_output / "baseline"
    v1_output = paired_output / "v1_lgm_w005"
    common_training = [
        "-M", "hydro", "-D", args.dataset, "-R", _rate(args.rate),
        "--setting", "trans", "-G", str(args.gpu), "-S", str(args.seed),
        "-E", str(args.epochs), "--eval_interval", str(args.eval_interval),
        "--run_eval", str(args.eval_runs),
        "--lp_hidden", "128", "--lp_lr", "0.001", "--lp_epochs", "100",
        "--lp_syn_thresholds", "q55,q65,q75,q85,q95",
        "--lp_val_interval", "10", "--lp_decision_threshold", "0.5",
        "--lp_neg_ratio", "1.0", "--lp_f1_average", "macro",
        "--load_path", str(DATA_ROOT),
    ]
    baseline_command = [
        sys.executable, "-u", str(SOURCE_ROOT / "run" / "train_all.py"),
        *common_training, "--save_path", str(baseline_output),
    ]
    v1_command = [
        sys.executable, "-u", str(STEP_ROOT / "run_train_v1.py"),
        *common_training, "--save_path", str(v1_output),
        "--step58-output", str(v1_output),
        "--step58-lp-weight", str(args.lp_weight),
        "--step58-pair-limit", str(args.pair_limit),
        "--step58-syn-pair-fraction", str(args.syn_pair_fraction),
        "--step58-distance-alpha", str(args.distance_alpha),
    ]
    commands = {
        "audit": [("gradient_audit", audit_output, audit_command)],
        "baseline": [("baseline", baseline_output, baseline_command)],
        "v1": [("v1", v1_output, v1_command)],
        "pilot": [
            ("baseline", baseline_output, baseline_command),
            ("v1", v1_output, v1_command),
        ],
    }[args.stage]
    print(f"Step: {STEP_ROOT}")
    print(f"Shared source: {SOURCE_ROOT}")
    print(f"Shared data: {DATA_ROOT}")
    for label, output, command in commands:
        print(f"Output ({label}): {output}")
        print("Command:", subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"{label}.log"
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
