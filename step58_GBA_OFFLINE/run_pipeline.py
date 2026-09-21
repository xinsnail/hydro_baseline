"""Portable launcher for Step58-GBA-v2 snapshot and offline audit."""

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
DEFAULT_RATES = {"computers": 0.005, "photo": 0.05, "cora": 1.0}


def _rate(value: float) -> str:
    return f"{float(value):.12g}"


def _safe_print(line: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(line.encode(encoding, errors="replace").decode(
        encoding, errors="replace"), end="", flush=True)


def _run(label, command, output, dry_run):
    print(f"Output ({label}): {output}")
    print("Command:", subprocess.list2cmdline(command))
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"{label}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command, cwd=SOURCE_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _safe_print(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("trace", "audit", "all"), default="all")
    parser.add_argument("--dataset", choices=tuple(DEFAULT_RATES), default="computers")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lp-weight", type=float, default=0.05)
    parser.add_argument("--pair-limit", type=int, default=2048)
    parser.add_argument("--syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--distance-alpha", type=float, default=0.50)
    parser.add_argument(
        "--trace-points",
        default="0,1,51,101,151,201,251,301,351,401,451,501,551",
    )
    parser.add_argument("--eval-interval", type=int, default=60)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rate = DEFAULT_RATES[args.dataset] if args.rate is None else args.rate
    root = OUTPUT_ROOT / args.dataset / f"r_{_rate(rate)}" / f"seed_{args.seed}"
    trace_root = root / "trace"
    audit_root = root / "offline_audit"
    common = [
        "-M", "hydro", "-D", args.dataset, "-R", _rate(rate),
        "--setting", "trans", "-G", str(args.gpu), "-S", str(args.seed),
        "-E", str(args.epochs), "--eval_interval", str(args.eval_interval),
        "--run_eval", str(args.eval_runs),
        "--lp_hidden", "128", "--lp_lr", "0.001", "--lp_epochs", "100",
        "--lp_syn_thresholds", "q55,q65,q75,q85,q95",
        "--lp_val_interval", "10", "--lp_decision_threshold", "0.5",
        "--lp_neg_ratio", "1.0", "--lp_f1_average", "macro",
        "--condensed_selection_metric", "lp",
        "--load_path", str(DATA_ROOT), "--save_path", str(trace_root),
    ]
    trace_command = [
        sys.executable, "-u", str(STEP_ROOT / "run_trace.py"), *common,
        "--offline-output", str(trace_root),
        "--offline-trace-points", args.trace_points,
        "--offline-lp-weight", str(args.lp_weight),
        "--offline-pair-limit", str(args.pair_limit),
        "--offline-syn-pair-fraction", str(args.syn_pair_fraction),
        "--offline-distance-alpha", str(args.distance_alpha),
    ]
    audit_command = [
        sys.executable, "-u", str(STEP_ROOT / "run_offline_audit.py"), *common,
        "--offline-trace-root", str(trace_root),
        "--offline-audit-output", str(audit_root),
        "--offline-lp-weight", str(args.lp_weight),
        "--offline-pair-limit", str(args.pair_limit),
        "--offline-syn-pair-fraction", str(args.syn_pair_fraction),
        "--offline-distance-alpha", str(args.distance_alpha),
    ]
    print(f"Step: {STEP_ROOT}")
    print(f"Shared source: {SOURCE_ROOT}")
    print(f"Shared data: {DATA_ROOT}")
    if args.stage in ("trace", "all"):
        _run("trace", trace_command, trace_root, args.dry_run)
    if args.stage in ("audit", "all"):
        _run("offline_audit", audit_command, audit_root, args.dry_run)


if __name__ == "__main__":
    main()
