"""Evaluate saved HyDRO or LAPT graphs under one fixed LP protocol."""

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from graphslim.condensation.graph_io import (
    load_seed_graph,
    parse_integer_list,
    parse_source_paths,
    tensor_sha256,
    validate_seed_graphs,
)
from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.evaluation import PaperLinkPredictionEvaluator
from graphslim.utils import seed_everything


def _step23_audit(root, dataset, rate, condensation_seed):
    path = Path(root) / "paper_lapt_audit" / (
        f"{dataset}_{rate}_seed_{condensation_seed}.json"
    )
    if not path.is_file():
        return {"audit_path": None}
    with open(path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    return {
        "audit_path": str(path.resolve()),
        "selected_lambda": audit.get("selected_lambda"),
        "selected_rank_blend": audit.get("selected_rank_blend"),
        "selected_validation_mean_f1": audit.get("selected_validation_mean_f1"),
    }


if __name__ == "__main__":
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        sys.exit(args if isinstance(args, int) else 0)
    if not args.scltc_source_paths:
        raise ValueError(
            "--scltc_source_paths must list the saved HyDRO or LAPT roots"
        )
    source_paths = parse_source_paths(args.scltc_source_paths)
    source_seeds = parse_integer_list(args.scltc_source_seeds)
    if len(source_paths) != len(source_seeds):
        raise ValueError("Source path/seed counts differ")
    if len(source_seeds) < 1:
        raise ValueError("At least one independently selected graph is required")

    eval_seed = int(args.scltc_eval_seed)
    args.seed = eval_seed
    seed_everything(eval_seed)
    data = get_dataset(args.dataset, args, args.load_path)
    graphs = [
        load_seed_graph(
            root=path,
            method=args.method,
            dataset=args.dataset,
            rate=float(args.reduction_rate),
            seed=seed,
        )
        for path, seed in zip(source_paths, source_seeds)
    ]
    # Multi-seed reports additionally verify compatible class budgets.  A
    # single locked graph is also supported for confirmatory evaluation of a
    # validation-selected candidate while keeping the downstream LP seed fixed.
    if len(graphs) >= 2:
        validate_seed_graphs(graphs)
    evaluator = PaperLinkPredictionEvaluator(args)
    records = []
    for graph in graphs:
        # Explicit assignment prevents the evaluator from loading a graph by
        # args.seed, which must remain fixed for a fair downstream comparison.
        args.seed = eval_seed
        seed_everything(eval_seed)
        data.feat_syn = graph.features
        data.labels_syn = graph.labels
        data.adj_syn = graph.adjacency
        result = evaluator.evaluate(
            data,
            reduced=True,
            verbose=True,
            validation_only=False,
        )
        loaded_hash = tensor_sha256(data.adj_syn)
        source_hash = tensor_sha256(graph.adjacency)
        if loaded_hash != source_hash:
            raise RuntimeError(
                f"In-memory graph mismatch for condensation seed {graph.condensation_seed}"
            )
        record = {
            "condensation_seed": int(graph.condensation_seed),
            "source_root": graph.source_root,
            "adjacency_sha256": source_hash,
            "test_mean_f1": float(result["test_mean"]),
            "test_std_f1": float(result["test_std"]),
            "validation_mean_f1": float(result["val_mean"]),
            "validation_std_f1": float(result["val_std"]),
            **_step23_audit(
                graph.source_root,
                args.dataset,
                float(args.reduction_rate),
                graph.condensation_seed,
            ),
        }
        records.append(record)
        print(
            f"Fixed-LP evaluation, condensation seed {graph.condensation_seed}: "
            f"test F1={100 * record['test_mean_f1']:.2f} "
            f"+/- {100 * record['test_std_f1']:.2f}"
        )

    values = np.asarray([item["test_mean_f1"] for item in records])
    summary = {
        "experiment": "fixed-LP-seed evaluation of independently selected graphs",
        "dataset": args.dataset,
        "reduction_rate": float(args.reduction_rate),
        "fixed_lp_eval_seed": eval_seed,
        "run_eval": int(args.run_eval),
        "per_condensation_seed": records,
        "condensation_seed_mean_f1": float(values.mean()),
        "condensation_seed_std_f1": float(values.std(ddof=0)),
        "worst_condensation_seed_f1": float(values.min()),
        "best_condensation_seed_f1": float(values.max()),
    }
    output_root = Path(args.save_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "fixed_seed_graph_report.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    csv_path = output_root / "fixed_seed_graph_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condensation_seed", "selected_lambda", "selected_rank_blend",
                "selected_validation_mean_f1", "test_mean_f1", "test_std_f1",
                "validation_mean_f1", "validation_std_f1", "source_root",
                "audit_path", "adjacency_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(
        "Fixed-LP cross-seed result: "
        f"{100 * summary['condensation_seed_mean_f1']:.2f} +/- "
        f"{100 * summary['condensation_seed_std_f1']:.2f}; "
        f"worst seed={100 * summary['worst_condensation_seed_f1']:.2f}"
    )
    print(f"Fixed-LP report: {json_path}")
