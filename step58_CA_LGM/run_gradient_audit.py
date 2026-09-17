"""Run the non-intervening Step58 CA-LGM gradient audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(STEP_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_ROOT))

from calgm_audit import GradientAuditHyDRO
from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.utils import seed_everything


def _points(text: str) -> tuple[int, ...]:
    if str(text).strip().lower() in {"", "none"}:
        return ()
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Audit points must be non-negative")
    return values


def _extract_step_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--step58-output", type=Path, required=True)
    parser.add_argument("--step58-adj-points", type=_points, default=_points("0,50,100"))
    parser.add_argument("--step58-feat-points", type=_points, default=_points("10,60,110"))
    parser.add_argument("--step58-pair-limit", type=int, default=2048)
    parser.add_argument("--step58-syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--step58-distance-alpha", type=float, default=0.50)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


if __name__ == "__main__":
    step = _extract_step_arguments()
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        raise SystemExit(args if isinstance(args, int) else 0)
    if args.method != "hydro":
        raise ValueError("Step58 currently supports only -M hydro")
    requested = set(step.step58_adj_points) | set(step.step58_feat_points)
    if requested and max(requested) >= int(args.epochs):
        raise ValueError(
            f"Audit point {max(requested)} requires at least {max(requested) + 1} epochs"
        )
    args.checkpoints = []
    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    agent = GradientAuditHyDRO(
        setting=args.setting,
        reduction_rate=args.reduction_rate,
        data=data,
        args=args,
        audit_output=step.step58_output,
        adjacency_points=step.step58_adj_points,
        feature_points=step.step58_feat_points,
        pair_limit=step.step58_pair_limit,
        synthetic_pair_fraction=step.step58_syn_pair_fraction,
        distance_alpha=step.step58_distance_alpha,
    )
    agent.reduce(data, verbose=args.verbose)

