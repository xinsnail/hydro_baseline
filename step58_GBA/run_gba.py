"""Run a diagnostic-only replay of fixed-weight Step58 CA-LGM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
CA_LGM_ROOT = PROJECT_ROOT / "step58_CA_LGM"
for path in (SOURCE_ROOT, CA_LGM_ROOT, STEP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gba_train import GradientBehaviorAuditCALGM
from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.dataset.utils import load_reduced
from graphslim.evaluation.paper_link_prediction import PaperLinkPredictionEvaluator
from graphslim.utils import seed_everything


def _extract_gba_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gba-output", type=Path, required=True)
    parser.add_argument("--gba-lp-weight", type=float, default=0.05)
    parser.add_argument("--gba-pair-limit", type=int, default=2048)
    parser.add_argument("--gba-syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--gba-distance-alpha", type=float, default=0.50)
    parser.add_argument("--gba-audit-outer", type=int, default=0)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


if __name__ == "__main__":
    gba = _extract_gba_arguments()
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        raise SystemExit(args if isinstance(args, int) else 0)
    if args.method != "hydro":
        raise ValueError("Step58-GBA supports only -M hydro")
    if args.condensed_selection_metric != "lp":
        raise ValueError("Step58-GBA requires --condensed_selection_metric lp")
    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    agent = GradientBehaviorAuditCALGM(
        setting=args.setting,
        reduction_rate=args.reduction_rate,
        data=data,
        args=args,
        audit_output=gba.gba_output,
        adjacency_points=(), feature_points=(),
        pair_limit=gba.gba_pair_limit,
        synthetic_pair_fraction=gba.gba_syn_pair_fraction,
        distance_alpha=gba.gba_distance_alpha,
        lp_weight=gba.gba_lp_weight,
        audit_outer=gba.gba_audit_outer,
    )
    agent.reduce(data, verbose=args.verbose)
    data.adj_syn, data.feat_syn, data.labels_syn = load_reduced(args, data)
    PaperLinkPredictionEvaluator(args).evaluate(data, reduced=True, verbose=True)

