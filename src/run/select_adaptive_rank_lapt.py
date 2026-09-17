"""Select LAPT topology and rank-calibration strengths on validation data."""

import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from graphslim.condensation.link_task_data import (
    build_link_task_training_data,
)
from graphslim.condensation.paper_lapt import (
    build_paper_lapt_target,
    parse_lambdas,
    rank_calibrate_target,
)
from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.dataset.utils import load_reduced, save_reduced
from graphslim.evaluation import PaperLinkPredictionEvaluator
from graphslim.utils import seed_everything


def _dense_adjacency(value):
    if hasattr(value, "to_dense"):
        try:
            return value.to_dense()
        except TypeError:
            pass
    if hasattr(value, "toarray"):
        return torch.as_tensor(value.toarray())
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_rank_blends(values):
    result = tuple(
        sorted(
            {
                float(item.strip())
                for item in str(values).split(",")
                if item.strip()
            }
        )
    )
    if not result or any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("paper_lapt_rank_blends must lie in [0, 1]")
    if 0.0 not in result:
        raise ValueError(
            "paper_lapt_rank_blends must include 0 so the raw LAPT target "
            "is an explicit validation candidate"
        )
    return result


def _adaptive_target(raw_target, rank_target, rank_blend):
    rank_blend = float(rank_blend)
    return (1.0 - rank_blend) * raw_target + rank_blend * rank_target


if __name__ == "__main__":
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        sys.exit(args if isinstance(args, int) else 0)
    if not args.paper_lapt_source_path:
        raise ValueError(
            "--paper_lapt_source_path must point to the saved "
            "paper-protocol HyDRO checkpoint root"
        )

    source_path = os.path.abspath(args.paper_lapt_source_path)
    output_path = os.path.abspath(args.save_path)
    if os.path.normcase(source_path) == os.path.normcase(output_path):
        raise ValueError(
            "--save_path must be a new output directory; it cannot overwrite "
            "--paper_lapt_source_path"
        )

    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    original_output_path = args.save_path
    args.save_path = source_path
    try:
        source_adj, source_feat, source_labels = load_reduced(args, data)
    finally:
        args.save_path = original_output_path

    source_adj = _dense_adjacency(source_adj).detach().float().to(args.device)
    source_adj = (source_adj + source_adj.T) / 2.0
    source_adj.fill_diagonal_(0.0)
    training_data = build_link_task_training_data(
        adjacency=data.adj_full,
        seed=int(args.paper_lapt_split_seed),
        validation_ratio=float(args.paper_lapt_validation_ratio),
        test_ratio=float(args.paper_lapt_test_ratio),
        query_ratio=float(args.paper_lapt_query_ratio),
        negative_ratio=1.0,
        hard_negative_ratio=float(args.paper_lapt_hard_negative_ratio),
    )
    lapt_target, target_summary = build_paper_lapt_target(
        args=args,
        training_data=training_data,
        original_features=data.feat_full,
        original_labels=data.labels_full,
        synthetic_features=source_feat,
        synthetic_labels=source_labels,
        device=args.device,
        synthetic_adjacency=source_adj,
    )
    raw_target = lapt_target.to(source_adj.device)
    rank_target = rank_calibrate_target(source_adj, raw_target)
    if bool(args.paper_lapt_use_rank_calibration):
        rank_blends = _parse_rank_blends(args.paper_lapt_rank_blends)
        calibration_mode = "adaptive_rank"
    else:
        rank_blends = (0.0,)
        calibration_mode = "raw"

    candidates = []
    if args.paper_lapt_selection_mode == "fixed":
        selected_lambda = float(args.paper_lapt_fixed_lambda)
        selected_rank_blend = float(args.paper_lapt_fixed_rank_blend)
        if not 0.0 <= selected_lambda <= 1.0:
            raise ValueError("--paper_lapt_fixed_lambda must lie in [0, 1]")
        if not 0.0 <= selected_rank_blend <= 1.0:
            raise ValueError(
                "--paper_lapt_fixed_rank_blend must lie in [0, 1]"
            )
        candidates.append(
            {
                "lambda": selected_lambda,
                "rank_blend": selected_rank_blend,
                "validation_mean_f1": None,
                "validation_std_f1": None,
                "selected_thresholds": [],
            }
        )
        print(
            "Adaptive-Rank LAPT fixed mode: "
            f"lambda={selected_lambda:.2f}, "
            f"rank_blend={selected_rank_blend:.2f}; "
            "validation selection skipped"
        )
    else:
        old_run_eval = int(args.run_eval)
        args.run_eval = int(args.paper_lapt_validation_runs)
        evaluator = PaperLinkPredictionEvaluator(args)
        try:
            mixture_weights = parse_lambdas(args.paper_lapt_lambdas)
            for rank_blend in rank_blends:
                candidate_target = _adaptive_target(
                    raw_target,
                    rank_target,
                    rank_blend,
                )
                for mixture_weight in mixture_weights:
                    # lambda=0 is the same unchanged HyDRO graph for every rho.
                    if float(mixture_weight) == 0.0 and float(rank_blend) != 0.0:
                        continue
                    seed_everything(args.seed)
                    candidate_adj = (
                        (1.0 - float(mixture_weight)) * source_adj
                        + float(mixture_weight) * candidate_target
                    )
                    data.feat_syn = source_feat
                    data.labels_syn = source_labels
                    data.adj_syn = candidate_adj
                    result = evaluator.evaluate(
                        data,
                        reduced=True,
                        verbose=False,
                        validation_only=True,
                    )
                    record = {
                        "lambda": float(mixture_weight),
                        "rank_blend": float(rank_blend),
                        "validation_mean_f1": float(result["val_mean"]),
                        "validation_std_f1": float(result["val_std"]),
                        "selected_thresholds": [
                            float(value)
                            for value in result["selected_thresholds"]
                        ],
                    }
                    candidates.append(record)
                    print(
                        "Adaptive-Rank LAPT candidate "
                        f"lambda={mixture_weight:.2f}, "
                        f"rank_blend={rank_blend:.2f}: "
                        f"validation F1={100 * result['val_mean']:.2f} "
                        f"+/- {100 * result['val_std']:.2f}"
                    )
        finally:
            args.run_eval = old_run_eval

        tolerance = float(args.paper_lapt_selection_tolerance)
        if tolerance < 0.0:
            raise ValueError(
                "--paper_lapt_selection_tolerance must be non-negative"
            )
        best_validation = max(
            item["validation_mean_f1"] for item in candidates
        )
        eligible = [
            item
            for item in candidates
            if item["validation_mean_f1"] >= best_validation - tolerance
        ]
        selected = min(
            eligible,
            key=lambda item: (
                item["rank_blend"],
                item["lambda"],
                -item["validation_mean_f1"],
            ),
        )
        selected_lambda = float(selected["lambda"])
        selected_rank_blend = float(selected["rank_blend"])

    selected_target = _adaptive_target(
        raw_target,
        rank_target,
        selected_rank_blend,
    )
    selected_adj = (
        (1.0 - selected_lambda) * source_adj
        + selected_lambda * selected_target
    )
    selected = next(
        item for item in candidates
        if (
            float(item["lambda"]) == selected_lambda
            and float(item["rank_blend"]) == selected_rank_blend
        )
    )
    args.save_path = output_path
    save_reduced(
        adj_syn=selected_adj.detach().cpu(),
        feat_syn=source_feat.detach().cpu(),
        labels_syn=source_labels.detach().cpu(),
        args=args,
    )

    audit = {
        "experiment": "Step23 adaptive rank-gated LAPT",
        "source_path": source_path,
        "output_path": output_path,
        "dataset": args.dataset,
        "reduction_rate": float(args.reduction_rate),
        "condensation_seed": int(args.seed),
        "lapt_split_seed": int(args.paper_lapt_split_seed),
        "paper_protocol": {
            "edge_split": "RandomLinkSplit 70/10/20 per evaluation run",
            "f1_average": args.lp_f1_average,
            "synthetic_thresholds": args.lp_syn_thresholds,
            "lp_epochs": int(args.lp_epochs),
            "validation_runs_for_lambda": int(
                args.paper_lapt_validation_runs
            ),
        },
        "training_only_edge_split": training_data.summary,
        "lapt_target": target_summary.to_dict(),
        "teacher_enabled": bool(args.paper_lapt_use_teacher),
        "calibration_mode": calibration_mode,
        "selection_mode": args.paper_lapt_selection_mode,
        "rank_calibration": (
            "Validation selects rho in T_rho=(1-rho)T_raw+rho*T_rank; "
            "T_rank transfers the LAPT ordering while preserving the exact "
            "off-diagonal HyDRO weight multiset."
            if calibration_mode == "adaptive_rank"
            else "Disabled: only the raw transported target is considered."
        ),
        "rank_blends": [float(value) for value in rank_blends],
        "selection_tolerance": float(
            args.paper_lapt_selection_tolerance
        ),
        "candidates": candidates,
        "selected_lambda": selected_lambda,
        "selected_rank_blend": selected_rank_blend,
        "selected_validation_mean_f1": (
            None
            if selected["validation_mean_f1"] is None
            else float(selected["validation_mean_f1"])
        ),
    }
    audit_directory = os.path.join(output_path, "paper_lapt_audit")
    os.makedirs(audit_directory, exist_ok=True)
    audit_path = os.path.join(
        audit_directory,
        f"{args.dataset}_{args.reduction_rate}_seed_{args.seed}.json",
    )
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(audit), handle, ensure_ascii=False, indent=2)

    validation_text = (
        "not used"
        if selected["validation_mean_f1"] is None
        else f"{100 * selected['validation_mean_f1']:.2f}"
    )
    print(
        f"Selected Adaptive-Rank LAPT lambda={selected_lambda:.2f}, "
        f"rank_blend={selected_rank_blend:.2f}; "
        f"validation F1={validation_text}. Saved graph to {output_path}"
    )
    print(f"Audit: {audit_path}")
