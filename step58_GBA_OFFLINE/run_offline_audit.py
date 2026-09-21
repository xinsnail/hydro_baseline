"""Compute GBA-v2 diagnostics after snapshot-only training has finished."""

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

from graphslim.config import cli
from graphslim.dataset import get_dataset
from graphslim.utils import seed_everything
from offline_audit import run_offline_audit


def _extract_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--offline-trace-root", type=Path, required=True)
    parser.add_argument("--offline-audit-output", type=Path, required=True)
    parser.add_argument("--offline-lp-weight", type=float, default=0.05)
    parser.add_argument("--offline-pair-limit", type=int, default=2048)
    parser.add_argument("--offline-syn-pair-fraction", type=float, default=0.20)
    parser.add_argument("--offline-distance-alpha", type=float, default=0.50)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return known


if __name__ == "__main__":
    offline = _extract_arguments()
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        raise SystemExit(args if isinstance(args, int) else 0)
    if args.method != "hydro":
        raise ValueError("Step58-GBA-v2 supports only -M hydro")
    seed_everything(args.seed)
    data = get_dataset(args.dataset, args, args.load_path)
    run_offline_audit(
        data=data,
        args=args,
        trace_root=offline.offline_trace_root,
        output=offline.offline_audit_output,
        pair_limit=offline.offline_pair_limit,
        synthetic_pair_fraction=offline.offline_syn_pair_fraction,
        distance_alpha=offline.offline_distance_alpha,
        lp_weight=offline.offline_lp_weight,
    )
