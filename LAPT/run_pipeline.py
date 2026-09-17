"""Portable end-to-end HyDRO and LAPT experiment runner.

Pipeline:
1. train one HyDRO condensed graph per condensation seed;
2. build one validation-selected LAPT graph from each HyDRO graph;
3. evaluate both graph sets under exactly the same fixed LP protocol;
4. write paired per-seed and aggregate CSV/TXT reports.

The LAPT step lives in its own directory while ``src``, ``data`` and
``outputs`` are shared by all experiment steps in the parent project.  All
paths are therefore resolved relative to this file and its parent directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RATES = {"computers": 0.005, "photo": 0.01, "cora": 0.5}


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)) or any(seed <= 0 for seed in seeds):
        raise argparse.ArgumentTypeError("Seeds must be distinct positive integers")
    return seeds


def rate_text(value: float) -> str:
    return f"{float(value):.12g}"


def artifact_rate_text(value: float) -> str:
    """Match the filename produced after Click parses -R as a Python float."""
    return str(float(value))


def graph_paths(root: Path, dataset: str, rate: float, seed: int) -> tuple[Path, ...]:
    directory = root / "reduced_graph" / "hydro"
    stem = f"{dataset}_{artifact_rate_text(rate)}_{seed}.pt"
    return tuple(directory / f"{kind}_{stem}" for kind in ("adj", "feat", "label"))


def graph_complete(root: Path, dataset: str, rate: float, seed: int) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in graph_paths(root, dataset, rate, seed))


def safe_print(line: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"), end="", flush=True)


def run_logged(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n[RUN] {printable}\n[LOG] {log_path}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"COMMAND: {printable}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe_print(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def common_lp_args(dataset: str, rate: float, seed: int, gpu: int, runs: int) -> list[str]:
    return [
        "-M", "hydro", "-D", dataset, "-R", rate_text(rate),
        "--setting", "trans", "-G", str(gpu), "-S", str(seed),
        "--run_eval", str(runs), "--lp_hidden", "128",
        "--lp_lr", "0.001", "--lp_epochs", "100",
        "--lp_syn_thresholds", "q55,q65,q75,q85,q95",
        "--lp_val_interval", "10", "--lp_decision_threshold", "0.5",
        "--lp_neg_ratio", "1.0", "--lp_f1_average", "macro",
    ]


def experiment_root(output_root: Path, dataset: str, rate: float) -> Path:
    return output_root / dataset / f"r_{rate_text(rate)}"


def train_hydro(args: argparse.Namespace, root: Path) -> None:
    target = root / "hydro"
    for seed in args.seeds:
        if graph_complete(target, args.dataset, args.rate, seed):
            print(f"[SKIP] Complete HyDRO graph: seed {seed}", flush=True)
            continue
        command = [
            sys.executable, "-u", str(SOURCE_ROOT / "run" / "train_all.py"),
            "-M", "hydro", "-D", args.dataset, "-R", rate_text(args.rate),
            "--setting", "trans", "-G", str(args.gpu), "-S", str(seed),
            "--load_path", str(DATA_ROOT), "--save_path", str(target),
        ]
        run_logged(command, SOURCE_ROOT, root / "logs" / f"hydro_seed_{seed}.txt", args.dry_run)
        if not args.dry_run and not graph_complete(target, args.dataset, args.rate, seed):
            raise FileNotFoundError(f"HyDRO did not produce a complete seed-{seed} graph")


def select_lapt(args: argparse.Namespace, root: Path) -> None:
    source = root / "hydro"
    target = root / "lapt"
    for seed in args.seeds:
        if graph_complete(target, args.dataset, args.rate, seed):
            print(f"[SKIP] Complete LAPT graph: seed {seed}", flush=True)
            continue
        if not args.dry_run and not graph_complete(source, args.dataset, args.rate, seed):
            raise FileNotFoundError(f"Run HyDRO first; seed {seed} is missing")
        command = [
            sys.executable, "-u", str(SOURCE_ROOT / "run" / "select_adaptive_rank_lapt.py"),
            *common_lp_args(args.dataset, args.rate, seed, args.gpu, args.validation_runs),
            "--paper_lapt_source_path", str(source),
            "--paper_lapt_lambdas", args.lambdas,
            "--paper_lapt_rank_blends", args.rank_blends,
            "--paper_lapt_selection_tolerance", "0.001",
            "--paper_lapt_validation_runs", str(args.validation_runs),
            "--paper_lapt_split_seed", str(args.split_seed),
            "--load_path", str(DATA_ROOT), "--save_path", str(target),
        ]
        run_logged(command, SOURCE_ROOT, root / "logs" / f"lapt_seed_{seed}.txt", args.dry_run)
        if not args.dry_run and not graph_complete(target, args.dataset, args.rate, seed):
            raise FileNotFoundError(f"LAPT did not produce a complete seed-{seed} graph")


def evaluate(args: argparse.Namespace, root: Path) -> None:
    seed_text = ",".join(map(str, args.seeds))
    for method in ("hydro", "lapt"):
        source = root / method
        destination = root / "evaluation" / method
        report = destination / "fixed_seed_graph_report.csv"
        if report.is_file():
            print(f"[SKIP] Existing evaluation: {report}", flush=True)
            continue
        if not args.dry_run:
            missing = [seed for seed in args.seeds if not graph_complete(source, args.dataset, args.rate, seed)]
            if missing:
                raise FileNotFoundError(f"Missing {method} graphs for seeds {missing}")
        command = [
            sys.executable, "-u", str(SOURCE_ROOT / "run" / "evaluate_fixed_seed_outputs.py"),
            *common_lp_args(args.dataset, args.rate, args.eval_seed, args.gpu, args.eval_runs),
            "--scltc_source_paths", ",".join(str(source) for _ in args.seeds),
            "--scltc_source_seeds", seed_text,
            "--scltc_eval_seed", str(args.eval_seed),
            "--load_path", str(DATA_ROOT), "--save_path", str(destination),
        ]
        run_logged(command, SOURCE_ROOT, root / "logs" / f"evaluate_{method}.txt", args.dry_run)


def read_report(path: Path, seeds: tuple[int, ...]) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = {int(row["condensation_seed"]): row for row in rows}
    if set(report) != set(seeds):
        raise ValueError(f"{path} must contain exactly seeds {seeds}")
    return report


def summarize(args: argparse.Namespace, root: Path) -> None:
    if args.dry_run:
        print(f"[SUMMARIZE] paired reports under {root / 'evaluation'}", flush=True)
        return
    hydro = read_report(root / "evaluation" / "hydro" / "fixed_seed_graph_report.csv", args.seeds)
    lapt = read_report(root / "evaluation" / "lapt" / "fixed_seed_graph_report.csv", args.seeds)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        baseline = 100.0 * float(hydro[seed]["test_mean_f1"])
        adapted = 100.0 * float(lapt[seed]["test_mean_f1"])
        same_graph = bool(hydro[seed].get("adjacency_sha256")) and (
            hydro[seed].get("adjacency_sha256") == lapt[seed].get("adjacency_sha256")
        )
        if same_graph:
            adapted = baseline
        rows.append({
            "dataset": args.dataset,
            "reduction_rate": args.rate,
            "condensation_seed": seed,
            "hydro_macro_f1": baseline,
            "lapt_macro_f1": adapted,
            "gain_points": adapted - baseline,
            "selected_lambda": lapt[seed].get("selected_lambda", ""),
            "selected_rank_blend": lapt[seed].get("selected_rank_blend", ""),
            "same_adjacency_as_hydro": same_graph,
        })
    baseline = np.asarray([float(row["hydro_macro_f1"]) for row in rows])
    adapted = np.asarray([float(row["lapt_macro_f1"]) for row in rows])
    delta = adapted - baseline
    if len(delta) > 1:
        margin = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        p_value = float(stats.ttest_rel(adapted, baseline).pvalue)
        baseline_std = float(baseline.std(ddof=1))
        adapted_std = float(adapted.std(ddof=1))
    else:
        margin = float("nan")
        p_value = float("nan")
        baseline_std = adapted_std = 0.0
    summary = {
        "dataset": args.dataset,
        "reduction_rate": args.rate,
        "condensation_seeds": len(args.seeds),
        "hydro_mean_f1": float(baseline.mean()),
        "hydro_seed_std_f1": baseline_std,
        "lapt_mean_f1": float(adapted.mean()),
        "lapt_seed_std_f1": adapted_std,
        "gain_points": float(delta.mean()),
        "relative_gain_percent": float(100.0 * delta.mean() / baseline.mean()),
        "gain_ci95_lower": float(delta.mean() - margin),
        "gain_ci95_upper": float(delta.mean() + margin),
        "paired_t_p": p_value,
        "positive_seeds": int(np.sum(delta > 1e-12)),
        "neutral_seeds": int(np.sum(np.abs(delta) <= 1e-12)),
        "negative_seeds": int(np.sum(delta < -1e-12)),
    }
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_per_seed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "paired_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    text = (
        f"dataset={args.dataset}\nreduction_rate={args.rate}\nseeds={','.join(map(str, args.seeds))}\n"
        f"HyDRO={summary['hydro_mean_f1']:.4f} +/- {summary['hydro_seed_std_f1']:.4f}\n"
        f"LAPT={summary['lapt_mean_f1']:.4f} +/- {summary['lapt_seed_std_f1']:.4f}\n"
        f"gain={summary['gain_points']:+.4f} pp; relative={summary['relative_gain_percent']:+.3f}%; "
        f"95% CI=[{summary['gain_ci95_lower']:+.4f}, {summary['gain_ci95_upper']:+.4f}]; "
        f"paired p={summary['paired_t_p']:.6g}; pos/neutral/neg="
        f"{summary['positive_seeds']}/{summary['neutral_seeds']}/{summary['negative_seeds']}\n"
    )
    (output / "paired_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"Reports: {output}", flush=True)


def validate_bundle() -> None:
    required = [
        SOURCE_ROOT / "run" / "train_all.py",
        SOURCE_ROOT / "run" / "select_adaptive_rank_lapt.py",
        SOURCE_ROOT / "run" / "evaluate_fixed_seed_outputs.py",
        SOURCE_ROOT / "graphslim" / "condensation" / "paper_lapt.py",
        DATA_ROOT / "Computers" / "raw" / "amazon_electronics_computers.npz",
        DATA_ROOT / "Photo" / "raw" / "amazon_electronics_photo.npz",
        DATA_ROOT / "cora" / "raw" / "ind.cora.graph",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete portable bundle:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULT_RATES), required=True)
    parser.add_argument("--rate", type=float, default=None, help="Defaults: computers=.005, photo=.01, cora=.5")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1"))
    parser.add_argument("--stage", choices=("all", "hydro", "lapt", "evaluate", "summarize"), default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--validation-runs", type=int, default=3)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--rank-blends", default="0,0.5,1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.rate = DEFAULT_RATES[args.dataset] if args.rate is None else float(args.rate)
    args.output_root = args.output_root.expanduser().resolve()
    validate_bundle()
    root = experiment_root(args.output_root, args.dataset, args.rate)
    print(
        f"Step={STEP_ROOT}\nProject={PROJECT_ROOT}\n"
        f"Dataset={args.dataset}\nRate={args.rate}\n"
        f"Seeds={args.seeds}\nOutput={root}",
        flush=True,
    )
    stages = ("hydro", "lapt", "evaluate", "summarize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        {"hydro": train_hydro, "lapt": select_lapt, "evaluate": evaluate, "summarize": summarize}[stage](args, root)


if __name__ == "__main__":
    main()
