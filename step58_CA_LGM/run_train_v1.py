"""Train and evaluate Step58 V1 HyDRO+LGM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (SOURCE_ROOT, STEP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calgm_train import CALGMHyDRO
from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.dataset.utils import load_reduced
from graphslim.evaluation.paper_link_prediction import PaperLinkPredictionEvaluator
from graphslim.utils import seed_everything


def _extract_step_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--step58-output", type=Path, required=True)
    parser.add_argument("--step58-lp-weight", type=float, default=0.05)
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
        raise ValueError("Step58 V1 supports only -M hydro")
    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    agent = CALGMHyDRO(
        setting=args.setting,
        reduction_rate=args.reduction_rate,
        data=data,
        args=args,
        audit_output=step.step58_output,
        adjacency_points=(),
        feature_points=(),
        pair_limit=step.step58_pair_limit,
        synthetic_pair_fraction=step.step58_syn_pair_fraction,
        distance_alpha=step.step58_distance_alpha,
        lp_weight=step.step58_lp_weight,
    )
    agent.reduce(data, verbose=args.verbose)
    data.adj_syn, data.feat_syn, data.labels_syn = load_reduced(args, data)
    PaperLinkPredictionEvaluator(args).evaluate(data, reduced=True, verbose=True)

