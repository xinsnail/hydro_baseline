"""Step58-GBA: diagnostic replay of fixed-weight CA-LGM.

The optimizer update is identical to Step58 CA-LGM. This module only reads
the two PGE gradients before the normal backward pass and writes diagnostic
CSV/text files. It never projects, rescales, clips, or replaces gradients.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import trange

from calgm_train import CALGMHyDRO
from calgm_audit import _teacher_synthetic_pairs
from graphslim.models import *  # noqa: F401,F403 - selected from config
from graphslim.utils import normalize_adj_tensor, regularization, tensor2onehot, to_tensor


TRAIN_FIELDS = (
    "epoch", "outer_loop", "node_gm_loss", "link_gm_loss",
    "positive_gradient_distance", "negative_gradient_distance",
    "lp_weight", "weighted_link_gm_loss",
)
GBA_FIELDS = (
    "epoch", "outer_loop", "phase_start_epoch", "hydro_gradient_norm",
    "link_gradient_norm", "weighted_link_gradient_norm",
    "raw_link_injection_ratio", "gradient_dot", "gradient_cosine",
    "gradient_conflict", "finite",
)
PHASE_FIELDS = (
    "phase_start_epoch", "phase_end_epoch", "samples", "ratio_mean",
    "ratio_median", "ratio_std", "ratio_p90", "ratio_p95", "ratio_max",
    "cosine_mean", "cosine_median", "cosine_min", "cosine_max",
    "conflict_fraction", "hydro_gradient_norm_mean",
    "link_gradient_norm_mean",
)
STATE_FIELDS = (
    "epoch", "best_validation_f1_so_far",
    "raw_adjacency_change_from_initial",
    "normalized_propagation_change_from_initial",
    "normalized_propagation_cosine_from_initial",
    "feature_change_from_initial",
)


def _write_rows(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gradient_dot(first, second, device) -> torch.Tensor:
    value = torch.zeros((), device=device)
    for left, right in zip(first, second):
        if left is not None and right is not None:
            value = value + (left * right).sum()
    return value


def _gradient_norm(gradient, device) -> torch.Tensor:
    return _gradient_dot(gradient, gradient, device).clamp_min(0.0).sqrt()


def _percentile(values, probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values) -> float:
    return statistics.fmean(values) if values else float("nan")


def _std(values) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _relative_change(reference: torch.Tensor, value: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float(
        (torch.linalg.vector_norm(value - reference) / denominator).detach()
    )


def _tensor_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    ).clamp_min(1e-12)
    return float(((first * second).sum() / denominator).detach())


class GradientBehaviorAuditCALGM(CALGMHyDRO):
    """Fixed CA-LGM plus read-only PGE-gradient behavior diagnostics."""

    def __init__(self, *args, audit_outer: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_outer = int(audit_outer)
        if self.audit_outer < 0:
            raise ValueError("audit_outer must be non-negative")

    def _measure_pge_gradients(self, hydro_loss, link_loss):
        parameters = tuple(
            parameter for parameter in self.hyp.parameters()
            if parameter.requires_grad
        )
        hydro_gradient = torch.autograd.grad(
            hydro_loss, parameters, retain_graph=True, allow_unused=True
        )
        link_gradient = torch.autograd.grad(
            link_loss, parameters, retain_graph=True, allow_unused=True
        )
        hydro_norm = _gradient_norm(hydro_gradient, self.device)
        link_norm = _gradient_norm(link_gradient, self.device)
        dot = _gradient_dot(hydro_gradient, link_gradient, self.device)
        hydro_value = float(hydro_norm.detach())
        link_value = float(link_norm.detach())
        dot_value = float(dot.detach())
        denominator = hydro_value * link_value
        cosine = (
            dot_value / denominator if denominator > 1e-12 else float("nan")
        )
        weighted_link_norm = self.lp_weight * link_value
        ratio = weighted_link_norm / max(hydro_value, 1e-12)
        finite = all(
            math.isfinite(value)
            for value in (hydro_value, link_value, dot_value, ratio)
        ) and (math.isfinite(cosine) or denominator <= 1e-12)
        return {
            "hydro_gradient_norm": hydro_value,
            "link_gradient_norm": link_value,
            "weighted_link_gradient_norm": weighted_link_norm,
            "raw_link_injection_ratio": ratio,
            "gradient_dot": dot_value,
            "gradient_cosine": cosine,
            "gradient_conflict": int(math.isfinite(cosine) and cosine < 0.0),
            "finite": int(finite),
        }

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        training_path = self.audit_output / "v1_training_diagnostics.csv"
        gradient_path = self.audit_output / "gba_gradient_diagnostics.csv"
        phase_path = self.audit_output / "gba_phase_summary.csv"
        state_path = self.audit_output / "gba_state_trajectory.csv"
        snapshot_root = self.audit_output / "trajectory_snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)

        feat_syn, labels_syn = to_tensor(
            self.feat_syn, label=data.labels_syn, device=self.device
        )
        if args.setting == "trans":
            features, adjacency, labels = to_tensor(
                data.feat_full, data.adj_full,
                label=data.labels_full, device=self.device,
            )
        else:
            features, adjacency, labels = to_tensor(
                data.feat_train, data.adj_train,
                label=data.labels_train, device=self.device,
            )
        self.feat_syn.data.copy_(self.init())
        adjacency = normalize_adj_tensor(adjacency, sparse=True)
        (
            _training_data, teacher, teacher_metrics, real_positive,
            real_negative, message_adjacency,
        ) = self._prepare_link_audit(data)
        print(
            "Step58-GBA teacher: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}, "
            f"lp_weight={self.lp_weight:.6f}, "
            f"audit_outer={self.audit_outer}"
        )

        initial_features = self.feat_syn.detach().clone()
        with torch.no_grad():
            initial_raw_adjacency = self.hyp.inference(initial_features).detach()
            initial_normalized_adjacency = normalize_adj_tensor(
                initial_raw_adjacency, sparse=False
            ).detach()
        torch.save(initial_raw_adjacency.cpu(), snapshot_root / "adj_epoch_0000.pt")
        torch.save(initial_features.cpu(), snapshot_root / "feat_epoch_0000.pt")

        outer_loop, inner_loop = self.get_loops(args)
        if self.audit_outer >= outer_loop:
            raise ValueError(
                f"audit_outer={self.audit_outer} but outer_loop={outer_loop}"
            )
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        best_val = 0.0
        training_rows = []
        gradient_rows = []
        state_rows = []

        for epoch_index in trange(args.epochs, desc="Step58-GBA fixed CA-LGM"):
            epoch = epoch_index + 1
            model.initialize()
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            adjacency_phase = epoch_index % 50 < 10
            if adjacency_phase:
                synthetic_positive, synthetic_negative = _teacher_synthetic_pairs(
                    teacher, self.feat_syn,
                    self.synthetic_pair_fraction, self.device,
                )
            epoch_node_loss = 0.0
            epoch_link_loss = 0.0

            for outer in range(outer_loop):
                if adjacency.size(0) > 5000:
                    random_nodes = torch.randperm(
                        adjacency.size(0), device=self.device
                    )[:5000]
                    if isinstance(adjacency, SparseTensor):
                        sampled_adjacency = (
                            adjacency.index_select(0, random_nodes)
                            .index_select(1, random_nodes).to_dense()
                        )
                    else:
                        sampled_adjacency = (
                            adjacency.index_select(0, random_nodes)
                            .index_select(1, random_nodes)
                        )
                    raw_synthetic_adjacency, spectral_loss = self.hyp(
                        self.feat_syn, Lx=sampled_adjacency
                    )
                else:
                    raw_synthetic_adjacency, spectral_loss = self.hyp(
                        self.feat_syn, Lx=adjacency
                    )
                self.adj_syn = normalize_adj_tensor(
                    raw_synthetic_adjacency, sparse=False
                )
                model = self.check_bn(model)
                node_gm_loss = self.train_class(
                    model, adjacency, features, labels, labels_syn, args
                )
                total_loss = node_gm_loss
                hydro_loss = node_gm_loss
                link_gm_loss = torch.zeros((), device=self.device)
                positive_distance = torch.zeros((), device=self.device)
                negative_distance = torch.zeros((), device=self.device)

                if adjacency_phase:
                    (
                        link_gm_loss, positive_distance, negative_distance,
                    ) = self._link_gm_loss(
                        model=model,
                        real_features=features,
                        message_adjacency=message_adjacency,
                        real_positive=real_positive,
                        real_negative=real_negative,
                        synthetic_features=self.feat_syn,
                        synthetic_adjacency=self.adj_syn,
                        synthetic_positive=synthetic_positive,
                        synthetic_negative=synthetic_negative,
                    )
                    total_loss = total_loss + self.lp_weight * link_gm_loss
                    training_rows.append(
                        {
                            "epoch": epoch,
                            "outer_loop": outer,
                            "node_gm_loss": float(node_gm_loss.detach()),
                            "link_gm_loss": float(link_gm_loss.detach()),
                            "positive_gradient_distance": float(
                                positive_distance.detach()
                            ),
                            "negative_gradient_distance": float(
                                negative_distance.detach()
                            ),
                            "lp_weight": self.lp_weight,
                            "weighted_link_gm_loss": float(
                                (self.lp_weight * link_gm_loss).detach()
                            ),
                        }
                    )
                epoch_node_loss += float(node_gm_loss.detach())
                epoch_link_loss += float(link_gm_loss.detach())

                if args.beta > 0:
                    regularization_loss = args.beta * regularization(
                        raw_synthetic_adjacency,
                        tensor2onehot(labels_syn.to("cpu")).to(
                            raw_synthetic_adjacency.device
                        ),
                        args,
                    )
                    total_loss = total_loss + regularization_loss
                    hydro_loss = hydro_loss + regularization_loss
                if args.spectral_objective and adjacency_phase:
                    total_loss = total_loss + spectral_loss
                    hydro_loss = hydro_loss + spectral_loss

                if adjacency_phase and outer == self.audit_outer:
                    diagnostics = self._measure_pge_gradients(
                        hydro_loss, link_gm_loss
                    )
                    gradient_rows.append(
                        {
                            "epoch": epoch,
                            "outer_loop": outer,
                            "phase_start_epoch": (epoch_index // 50) * 50 + 1,
                            **diagnostics,
                        }
                    )

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                total_loss.backward()
                if adjacency_phase:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                inner_features = self.feat_syn.detach()
                inner_raw_adjacency = self.hyp.inference(inner_features)
                inner_normalized_adjacency = normalize_adj_tensor(
                    inner_raw_adjacency, sparse=False
                )
                for _ in range(inner_loop):
                    optimizer_model.zero_grad()
                    inner_output = model(
                        inner_features, inner_normalized_adjacency
                    )
                    inner_loss = F.nll_loss(inner_output, labels_syn)
                    inner_loss.backward()
                    optimizer_model.step()

            if adjacency_phase:
                _write_rows(training_path, TRAIN_FIELDS, training_rows)
                _write_rows(gradient_path, GBA_FIELDS, gradient_rows)

            if epoch_index in args.checkpoints:
                self.adj_syn = inner_raw_adjacency
                data.adj_syn = self.adj_syn.detach()
                data.feat_syn = self.feat_syn.detach()
                data.labels_syn = labels_syn.detach()
                best_val = self.intermediate_evaluation_lp(
                    best_val,
                    (epoch_node_loss + self.lp_weight * epoch_link_loss)
                    / max(outer_loop, 1),
                )
                current_raw = inner_raw_adjacency.detach()
                current_normalized = inner_normalized_adjacency.detach()
                current_features = self.feat_syn.detach()
                state_rows.append(
                    {
                        "epoch": epoch,
                        "best_validation_f1_so_far": best_val,
                        "raw_adjacency_change_from_initial": _relative_change(
                            initial_raw_adjacency, current_raw
                        ),
                        "normalized_propagation_change_from_initial": (
                            _relative_change(
                                initial_normalized_adjacency,
                                current_normalized,
                            )
                        ),
                        "normalized_propagation_cosine_from_initial": (
                            _tensor_cosine(
                                initial_normalized_adjacency,
                                current_normalized,
                            )
                        ),
                        "feature_change_from_initial": _relative_change(
                            initial_features, current_features
                        ),
                    }
                )
                _write_rows(state_path, STATE_FIELDS, state_rows)
                torch.save(
                    current_raw.cpu(), snapshot_root / f"adj_epoch_{epoch:04d}.pt"
                )
                torch.save(
                    current_features.cpu(),
                    snapshot_root / f"feat_epoch_{epoch:04d}.pt",
                )

        _write_rows(training_path, TRAIN_FIELDS, training_rows)
        _write_rows(gradient_path, GBA_FIELDS, gradient_rows)
        _write_rows(state_path, STATE_FIELDS, state_rows)

        finite_rows = [
            row for row in gradient_rows
            if int(row["finite"]) == 1
            and math.isfinite(float(row["gradient_cosine"]))
        ]
        phase_rows = []
        for phase_start in sorted(
            {int(row["phase_start_epoch"]) for row in finite_rows}
        ):
            block = [
                row for row in finite_rows
                if int(row["phase_start_epoch"]) == phase_start
            ]
            ratios = [float(row["raw_link_injection_ratio"]) for row in block]
            cosines = [float(row["gradient_cosine"]) for row in block]
            hydro_norms = [float(row["hydro_gradient_norm"]) for row in block]
            link_norms = [float(row["link_gradient_norm"]) for row in block]
            phase_rows.append(
                {
                    "phase_start_epoch": phase_start,
                    "phase_end_epoch": phase_start + 9,
                    "samples": len(block),
                    "ratio_mean": _mean(ratios),
                    "ratio_median": statistics.median(ratios),
                    "ratio_std": _std(ratios),
                    "ratio_p90": _percentile(ratios, 0.90),
                    "ratio_p95": _percentile(ratios, 0.95),
                    "ratio_max": max(ratios),
                    "cosine_mean": _mean(cosines),
                    "cosine_median": statistics.median(cosines),
                    "cosine_min": min(cosines),
                    "cosine_max": max(cosines),
                    "conflict_fraction": _mean(
                        [float(value < 0.0) for value in cosines]
                    ),
                    "hydro_gradient_norm_mean": _mean(hydro_norms),
                    "link_gradient_norm_mean": _mean(link_norms),
                }
            )
        _write_rows(phase_path, PHASE_FIELDS, phase_rows)

        ratios = [float(row["raw_link_injection_ratio"]) for row in finite_rows]
        cosines = [float(row["gradient_cosine"]) for row in finite_rows]
        summary_path = self.audit_output / "gba_summary.txt"
        summary_path.write_text(
            "Step58-GBA fixed CA-LGM gradient behavior audit\n"
            "diagnostic_only=true\n"
            "gradient_parameters=PGE_only\n"
            f"epochs={args.epochs}\n"
            f"lp_weight={self.lp_weight:.8f}\n"
            f"audit_outer={self.audit_outer}\n"
            f"teacher_query_auc={teacher_metrics.query_auc:.8f}\n"
            f"teacher_query_gap={teacher_metrics.query_score_gap:.8f}\n"
            f"best_validation_f1={best_val:.8f}\n"
            f"samples_total={len(gradient_rows)}\n"
            f"samples_finite={len(finite_rows)}\n"
            f"ratio_mean={_mean(ratios):.8f}\n"
            f"ratio_median={statistics.median(ratios) if ratios else float('nan'):.8f}\n"
            f"ratio_std={_std(ratios):.8f}\n"
            f"ratio_p90={_percentile(ratios, 0.90):.8f}\n"
            f"ratio_p95={_percentile(ratios, 0.95):.8f}\n"
            f"ratio_max={max(ratios) if ratios else float('nan'):.8f}\n"
            f"cosine_mean={_mean(cosines):.8f}\n"
            f"cosine_median={statistics.median(cosines) if cosines else float('nan'):.8f}\n"
            f"cosine_min={min(cosines) if cosines else float('nan'):.8f}\n"
            f"cosine_max={max(cosines) if cosines else float('nan'):.8f}\n"
            f"conflict_fraction={_mean([float(v < 0.0) for v in cosines]):.8f}\n",
            encoding="utf-8",
        )
        return data

