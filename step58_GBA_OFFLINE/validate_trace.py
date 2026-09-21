"""Compare a snapshot-only trace with a reference CA-LGM run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import torch


F1_PATTERN = re.compile(r"LP test mean F1:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _relative_error(reference, value) -> float:
    reference = torch.as_tensor(reference).float()
    value = torch.as_tensor(value).float()
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(value - reference) / denominator)


def _loss_differences(reference_path: Path, trace_path: Path):
    with reference_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reference = list(csv.DictReader(stream))
    with trace_path.open("r", encoding="utf-8-sig", newline="") as stream:
        trace = list(csv.DictReader(stream))
    count = min(len(reference), len(trace))
    node = [
        abs(float(reference[i]["node_gm_loss"]) - float(trace[i]["node_gm_loss"]))
        for i in range(count)
    ]
    link = [
        abs(float(reference[i]["link_gm_loss"]) - float(trace[i]["link_gm_loss"]))
        for i in range(count)
    ]
    return {
        "reference_rows": len(reference),
        "trace_rows": len(trace),
        "compared_rows": count,
        "node_loss_max_abs_error": max(node) if node else float("nan"),
        "node_loss_mean_abs_error": sum(node) / len(node) if node else float("nan"),
        "link_loss_max_abs_error": max(link) if link else float("nan"),
        "link_loss_mean_abs_error": sum(link) / len(link) if link else float("nan"),
    }


def _read_f1(root: Path):
    candidates = list(root.glob("*.log")) + list(root.glob("logs/**/*.log"))
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(F1_PATTERN.finditer(text))
        if matches:
            match = matches[-1]
            return float(match.group(1)), float(match.group(2)), path
    return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.trace / "trace_validation.txt"

    loss = _loss_differences(
        args.reference / "v1_training_diagnostics.csv",
        args.trace / "v1_training_diagnostics.csv",
    )
    reference_adj = _find_one(args.reference, "reduced_graph/hydro/adj_*.pt")
    trace_adj = _find_one(args.trace, "reduced_graph/hydro/adj_*.pt")
    reference_feat = _find_one(args.reference, "reduced_graph/hydro/feat_*.pt")
    trace_feat = _find_one(args.trace, "reduced_graph/hydro/feat_*.pt")
    adjacency_error = _relative_error(_load(reference_adj), _load(trace_adj))
    feature_error = _relative_error(_load(reference_feat), _load(trace_feat))
    reference_f1, reference_std, reference_log = _read_f1(args.reference)
    trace_f1, trace_std, trace_log = _read_f1(args.trace)
    f1_difference = (
        float("nan") if reference_f1 is None or trace_f1 is None
        else trace_f1 - reference_f1
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "Step58-GBA-v2 trace validation\n"
        f"reference={args.reference}\n"
        f"trace={args.trace}\n"
        + "".join(f"{key}={value}\n" for key, value in loss.items())
        + f"selected_adjacency_relative_error={adjacency_error:.8e}\n"
        + f"selected_feature_relative_error={feature_error:.8e}\n"
        + f"reference_f1={reference_f1}\n"
        + f"reference_f1_std={reference_std}\n"
        + f"trace_f1={trace_f1}\n"
        + f"trace_f1_std={trace_std}\n"
        + f"trace_minus_reference_f1_pp={f1_difference}\n"
        + f"reference_log={reference_log}\n"
        + f"trace_log={trace_log}\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
