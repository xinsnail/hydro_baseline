"""Create a descriptive four-case summary for Step58-GBA.

This script deliberately does not compute a correlation or significance test:
four representative cases are enough for mechanism diagnosis, but not enough
for a reliable correlation estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = STEP_ROOT.parent / "outputs" / STEP_ROOT.name
CASES = (
    ("computers_seed1_positive", "computers", "0.005", 1),
    ("computers_seed2_negative", "computers", "0.005", 2),
    ("photo_seed1_negative", "photo", "0.05", 1),
    ("cora_seed1_negative", "cora", "1", 1),
)
FIELDS = (
    "case", "dataset", "rate", "seed", "status", "samples_total",
    "samples_finite", "ratio_mean", "ratio_median", "ratio_p90",
    "ratio_p95", "ratio_max", "cosine_mean", "cosine_median",
    "cosine_min", "cosine_max", "conflict_fraction",
    "best_validation_f1", "final_raw_adjacency_change_from_initial",
    "final_normalized_propagation_change_from_initial",
    "final_normalized_propagation_cosine_from_initial",
    "final_feature_change_from_initial",
)


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _read_last_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_ROOT / "gba_cross_case_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    for case, dataset, rate, seed in CASES:
        case_root = args.root / dataset / f"r_{rate}" / f"seed_{seed}"
        summary_path = case_root / "gba_summary.txt"
        state_path = case_root / "gba_state_trajectory.csv"
        row = {
            "case": case,
            "dataset": dataset,
            "rate": rate,
            "seed": seed,
            "status": "complete" if summary_path.exists() else "missing",
        }
        if summary_path.exists():
            summary = _read_key_values(summary_path)
            state = _read_last_state(state_path)
            for key in (
                "samples_total", "samples_finite", "ratio_mean",
                "ratio_median", "ratio_p90", "ratio_p95", "ratio_max",
                "cosine_mean", "cosine_median", "cosine_min", "cosine_max",
                "conflict_fraction", "best_validation_f1",
            ):
                row[key] = summary.get(key, "")
            row.update(
                {
                    "final_raw_adjacency_change_from_initial": state.get(
                        "raw_adjacency_change_from_initial", ""
                    ),
                    "final_normalized_propagation_change_from_initial": state.get(
                        "normalized_propagation_change_from_initial", ""
                    ),
                    "final_normalized_propagation_cosine_from_initial": state.get(
                        "normalized_propagation_cosine_from_initial", ""
                    ),
                    "final_feature_change_from_initial": state.get(
                        "feature_change_from_initial", ""
                    ),
                }
            )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    text_path = args.output.with_suffix(".txt")
    complete = sum(row["status"] == "complete" for row in rows)
    text_path.write_text(
        "Step58-GBA descriptive cross-case summary\n"
        f"complete_cases={complete}/4\n"
        "formal_correlation_test=false\n"
        "interpretation=compare ratio and cosine patterns descriptively; "
        "do not tune a running case from these diagnostics\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")
    print(f"Saved: {text_path}")
    print(f"Complete cases: {complete}/4")


if __name__ == "__main__":
    main()
