"""Show portable HyDRO/LAPT experiment progress without changing files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent


def parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def artifact_rate(value: float) -> str:
    return str(float(value))


def complete(root: Path, dataset: str, rate: float, seed: int) -> bool:
    directory = root / "reduced_graph" / "hydro"
    stem = f"{dataset}_{artifact_rate(rate)}_{seed}.pt"
    return all((directory / f"{kind}_{stem}").is_file() for kind in ("adj", "feat", "label"))


def last_progress(log: Path) -> str:
    if not log.is_file():
        return "not started"
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(\d+)%[^\r\n]*?(\d+)/600", text)
    if matches:
        percent, epoch = matches[-1]
        return f"training epoch {epoch}/600 ({percent}%)"
    if "100%" in text or "LP test mean F1" in text or "Fixed-LP report" in text:
        return "stage log completed"
    return "started; inspect log tail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("computers", "photo", "cora"), required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1"))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    exp = args.output_root.resolve() / args.dataset / f"r_{args.rate:.12g}"
    print(f"Experiment: {exp}")
    for seed in args.seeds:
        hydro = complete(exp / "hydro", args.dataset, args.rate, seed)
        lapt = complete(exp / "lapt", args.dataset, args.rate, seed)
        active = last_progress(exp / "logs" / (f"lapt_seed_{seed}.txt" if hydro else f"hydro_seed_{seed}.txt"))
        print(f"seed {seed}: HyDRO={'done' if hydro else 'pending'}, LAPT={'done' if lapt else 'pending'}, {active}")
    for method in ("hydro", "lapt"):
        report = exp / "evaluation" / method / "fixed_seed_graph_report.csv"
        print(f"{method} evaluation: {'done' if report.is_file() else 'pending'}")
    summary = exp / "summary" / "paired_summary.txt"
    print(f"paired summary: {'done' if summary.is_file() else 'pending'}")


if __name__ == "__main__":
    main()
