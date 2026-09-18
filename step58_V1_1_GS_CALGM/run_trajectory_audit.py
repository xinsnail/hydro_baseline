"""Run the Step58-V1.1 full PGE-gradient trajectory audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
PREVIOUS_STEP_ROOT = PROJECT_ROOT / "step58_CA_LGM"
for path in (SOURCE_ROOT, PREVIOUS_STEP_ROOT, STEP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.utils import seed_everything
from trajectory_audit import FullTrajectoryAuditHyDRO


def _extract_step_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--trajectory-rho", type=float, default=0.02)
    parser.add_argument("--trajectory-lambda-max", type=float, default=0.05)
    parser.add_argument("--trajectory-pair-limit", type=int, default=2048)
    parser.add_argument("--trajectory-syn-pair-fraction", type=float, default=0.20)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


if __name__ == "__main__":
    step = _extract_step_arguments()
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        raise SystemExit(args if isinstance(args, int) else 0)
    if args.method != "hydro":
        raise ValueError("Step58-V1.1 supports only -M hydro")
    args.checkpoints = []
    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    agent = FullTrajectoryAuditHyDRO(
        setting=args.setting,
        reduction_rate=args.reduction_rate,
        data=data,
        args=args,
        audit_output=step.trajectory_output,
        adjacency_points=(),
        feature_points=(),
        pair_limit=step.trajectory_pair_limit,
        synthetic_pair_fraction=step.trajectory_syn_pair_fraction,
        distance_alpha=1.0,
        rho=step.trajectory_rho,
        lambda_max=step.trajectory_lambda_max,
    )
    agent.reduce(data, verbose=args.verbose)
