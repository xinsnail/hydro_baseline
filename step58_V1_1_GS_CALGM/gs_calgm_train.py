"""Step58-V1.1: conflict-safe direction-only CA-LGM training.

The HyDRO gradient is kept unchanged.  The direction-only link gradient is
projected away from the HyDRO gradient whenever the two conflict, then scaled
to a small fixed ratio of the HyDRO gradient norm.  Only PGE/adjacency updates
receive the link-task intervention.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import trange

from calgm_audit import (
    GradientAuditHyDRO,
    _gradient_distance,
    _hidden_embedding,
    _link_losses,
    _teacher_synthetic_pairs,
)
from graphslim.models import *  # noqa: F401,F403 - selected from config
from graphslim.utils import normalize_adj_tensor, regularization, tensor2onehot, to_tensor


FIELDS = (
    "epoch",
    "outer_loop",
    "node_gm_loss",
    "direction_loss",
    "positive_direction_distance",
    "negative_direction_distance",
    "hydro_gradient_norm",
    "direction_gradient_norm",
    "gradient_cosine_before",
    "gradient_conflict",
    "projected_direction_norm",
    "gradient_cosine_after",
    "rho",
    "scale_unclipped",
    "scale_effective",
    "effective_gradient_ratio",
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
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


def _gradient_cosine(first, second, device, eps: float = 1e-12) -> float:
    left_norm = _gradient_norm(first, device)
    right_norm = _gradient_norm(second, device)
    denominator = left_norm * right_norm
    if float(denominator.detach()) <= eps:
        return float("nan")
    return float((_gradient_dot(first, second, device) / denominator).detach())


def _pcgrad_link_against_hydro(link_gradient, hydro_gradient, device, eps=1e-12):
    """Remove only the link component that opposes the HyDRO gradient."""
    dot = _gradient_dot(link_gradient, hydro_gradient, device)
    hydro_sq = _gradient_dot(hydro_gradient, hydro_gradient, device)
    conflict = bool(float(dot.detach()) < 0.0 and float(hydro_sq.detach()) > eps)
    if not conflict:
        return tuple(link_gradient), False
    coefficient = dot / (hydro_sq + eps)
    projected = []
    for link, hydro in zip(link_gradient, hydro_gradient):
        if link is None:
            projected.append(None)
        elif hydro is None:
            projected.append(link)
        else:
            projected.append(link - coefficient * hydro)
    return tuple(projected), True


class GradientSafeCALGMHyDRO(GradientAuditHyDRO):
    """HyDRO with PCGrad-protected, norm-limited direction CA-LGM."""

    def __init__(self, *args, rho: float, max_scale: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)
        self.max_scale = float(max_scale)
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")
        if self.max_scale <= 0.0:
            raise ValueError("max_scale must be positive")

    def _direction_loss(
        self,
        *,
        model,
        real_features,
        message_adjacency,
        real_positive,
        real_negative,
        synthetic_adjacency,
        synthetic_positive,
        synthetic_negative,
    ):
        model_parameters = list(model.parameters())
        was_training = model.training
        model.eval()
        real_hidden = _hidden_embedding(model, real_features, message_adjacency)
        real_positive_loss, real_negative_loss = _link_losses(
            real_hidden, real_positive, real_negative
        )
        real_positive_gradient = torch.autograd.grad(
            real_positive_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_negative_gradient = torch.autograd.grad(
            real_negative_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_positive_gradient = tuple(
            None if value is None else value.detach()
            for value in real_positive_gradient
        )
        real_negative_gradient = tuple(
            None if value is None else value.detach()
            for value in real_negative_gradient
        )

        synthetic_hidden = _hidden_embedding(
            model, self.feat_syn, synthetic_adjacency
        )
        synthetic_positive_loss, synthetic_negative_loss = _link_losses(
            synthetic_hidden, synthetic_positive, synthetic_negative
        )
        synthetic_positive_gradient = torch.autograd.grad(
            synthetic_positive_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        synthetic_negative_gradient = torch.autograd.grad(
            synthetic_negative_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        positive_direction, _ = _gradient_distance(
            real_positive_gradient,
            synthetic_positive_gradient,
            model_parameters,
            alpha=1.0,
        )
        negative_direction, _ = _gradient_distance(
            real_negative_gradient,
            synthetic_negative_gradient,
            model_parameters,
            alpha=1.0,
        )
        if was_training:
            model.train()
        return (
            0.5 * (positive_direction + negative_direction),
            positive_direction,
            negative_direction,
        )

    def _compose_pge_gradient(self, hydro_loss, direction_loss):
        parameters = [
            parameter for parameter in self.hyp.parameters() if parameter.requires_grad
        ]
        hydro_gradient = torch.autograd.grad(
            hydro_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        direction_gradient = torch.autograd.grad(
            direction_loss,
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        cosine_before = _gradient_cosine(
            hydro_gradient, direction_gradient, self.device
        )
        projected, conflict = _pcgrad_link_against_hydro(
            direction_gradient, hydro_gradient, self.device
        )
        hydro_norm = _gradient_norm(hydro_gradient, self.device)
        direction_norm = _gradient_norm(direction_gradient, self.device)
        projected_norm = _gradient_norm(projected, self.device)
        epsilon = 1e-12
        hydro_value = float(hydro_norm.detach())
        projected_value = float(projected_norm.detach())
        if (
            not math.isfinite(hydro_value)
            or not math.isfinite(projected_value)
            or hydro_value <= epsilon
            or projected_value <= epsilon
        ):
            scale_unclipped = 0.0
            scale_effective = 0.0
        else:
            scale_unclipped = self.rho * hydro_value / projected_value
            scale_effective = min(self.max_scale, scale_unclipped)
        for parameter, hydro, link in zip(parameters, hydro_gradient, projected):
            if hydro is None and link is None:
                parameter.grad = None
                continue
            combined = torch.zeros_like(parameter)
            if hydro is not None:
                combined = combined + hydro
            if link is not None:
                combined = combined + scale_effective * link
            parameter.grad = combined.detach()
        effective_ratio = (
            scale_effective * projected_value / max(hydro_value, epsilon)
            if hydro_value > epsilon
            else 0.0
        )
        return {
            "hydro_gradient_norm": hydro_value,
            "direction_gradient_norm": float(direction_norm.detach()),
            "gradient_cosine_before": cosine_before,
            "gradient_conflict": int(conflict),
            "projected_direction_norm": projected_value,
            "gradient_cosine_after": _gradient_cosine(
                hydro_gradient, projected, self.device
            ),
            "rho": self.rho,
            "scale_unclipped": scale_unclipped,
            "scale_effective": scale_effective,
            "effective_gradient_ratio": effective_ratio,
        }

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        diagnostics_path = self.audit_output / "v1_1_training_diagnostics.csv"
        feat_syn, labels_syn = to_tensor(
            self.feat_syn, label=data.labels_syn, device=self.device
        )
        if args.setting == "trans":
            features, adjacency, labels = to_tensor(
                data.feat_full,
                data.adj_full,
                label=data.labels_full,
                device=self.device,
            )
        else:
            features, adjacency, labels = to_tensor(
                data.feat_train,
                data.adj_train,
                label=data.labels_train,
                device=self.device,
            )
        self.feat_syn.data.copy_(self.init())
        adjacency = normalize_adj_tensor(adjacency, sparse=True)
        (
            _training_data,
            teacher,
            teacher_metrics,
            real_positive,
            real_negative,
            message_adjacency,
        ) = self._prepare_link_audit(data)
        print(
            "Step58-V1.1 teacher: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}, "
            f"rho={self.rho:.6f}, max_scale={self.max_scale:.3f}"
        )

        outer_loop, inner_loop = self.get_loops(args)
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        best_val = 0.0
        rows: list[dict[str, object]] = []
        for epoch in trange(args.epochs, desc="Step58-V1.1 GS-CA-LGM"):
            model.initialize()
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            adjacency_phase = epoch % 50 < 10
            if adjacency_phase:
                synthetic_positive, synthetic_negative = _teacher_synthetic_pairs(
                    teacher,
                    self.feat_syn,
                    self.synthetic_pair_fraction,
                    self.device,
                )
            epoch_node_loss = 0.0
            for outer in range(outer_loop):
                if adjacency.size(0) > 5000:
                    random_nodes = torch.randperm(
                        adjacency.size(0), device=self.device
                    )[:5000]
                    if isinstance(adjacency, SparseTensor):
                        sampled_adjacency = (
                            adjacency.index_select(0, random_nodes)
                            .index_select(1, random_nodes)
                            .to_dense()
                        )
                    else:
                        sampled_adjacency = (
                            adjacency.index_select(0, random_nodes)
                            .index_select(1, random_nodes)
                        )
                    raw_adjacency, spectral_loss = self.hyp(
                        self.feat_syn, Lx=sampled_adjacency
                    )
                else:
                    raw_adjacency, spectral_loss = self.hyp(
                        self.feat_syn, Lx=adjacency
                    )
                normalized_adjacency = normalize_adj_tensor(
                    raw_adjacency, sparse=False
                )
                self.adj_syn = normalized_adjacency
                model = self.check_bn(model)
                node_gm_loss = self.train_class(
                    model,
                    adjacency,
                    features,
                    labels,
                    labels_syn,
                    args,
                )
                hydro_loss = node_gm_loss
                if args.beta > 0:
                    hydro_loss = hydro_loss + args.beta * regularization(
                        raw_adjacency,
                        tensor2onehot(labels_syn.to("cpu")).to(
                            raw_adjacency.device
                        ),
                        args,
                    )
                if args.spectral_objective and adjacency_phase:
                    hydro_loss = hydro_loss + spectral_loss

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                if adjacency_phase:
                    direction_loss, positive_direction, negative_direction = (
                        self._direction_loss(
                            model=model,
                            real_features=features,
                            message_adjacency=message_adjacency,
                            real_positive=real_positive,
                            real_negative=real_negative,
                            synthetic_adjacency=normalized_adjacency,
                            synthetic_positive=synthetic_positive,
                            synthetic_negative=synthetic_negative,
                        )
                    )
                    gradient_row = self._compose_pge_gradient(
                        hydro_loss, direction_loss
                    )
                    rows.append(
                        {
                            "epoch": epoch,
                            "outer_loop": outer,
                            "node_gm_loss": float(node_gm_loss.detach()),
                            "direction_loss": float(direction_loss.detach()),
                            "positive_direction_distance": float(
                                positive_direction.detach()
                            ),
                            "negative_direction_distance": float(
                                negative_direction.detach()
                            ),
                            **gradient_row,
                        }
                    )
                    _write_rows(diagnostics_path, rows)
                    self.optimizer_pge.step()
                else:
                    hydro_loss.backward()
                    self.optimizer_feat.step()
                epoch_node_loss += float(node_gm_loss.detach())

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

            if epoch in args.checkpoints:
                self.adj_syn = inner_raw_adjacency
                data.adj_syn = self.adj_syn.detach()
                data.feat_syn = self.feat_syn.detach()
                data.labels_syn = labels_syn.detach()
                best_val = self.intermediate_evaluation_lp(
                    best_val, epoch_node_loss / max(outer_loop, 1)
                )

        _write_rows(diagnostics_path, rows)
        finite_cosines = [
            float(row["gradient_cosine_before"])
            for row in rows
            if math.isfinite(float(row["gradient_cosine_before"]))
        ]
        conflicts = sum(int(row["gradient_conflict"]) for row in rows)
        effective = [float(row["effective_gradient_ratio"]) for row in rows]
        (self.audit_output / "v1_1_training_summary.txt").write_text(
            "Step58-V1.1 gradient-safe CA-LGM training complete\n"
            f"epochs={args.epochs}\n"
            f"rho={self.rho:.8f}\n"
            f"max_scale={self.max_scale:.8f}\n"
            "update_phase=adjacency_only\n"
            "conflict_control=pcgrad_link_against_hydro\n"
            "link_objective=positive_negative_direction_only\n"
            f"teacher_query_auc={teacher_metrics.query_auc:.8f}\n"
            f"teacher_query_gap={teacher_metrics.query_score_gap:.8f}\n"
            f"gradient_updates={len(rows)}\n"
            f"conflict_updates={conflicts}\n"
            f"conflict_fraction={conflicts / max(len(rows), 1):.8f}\n"
            f"gradient_cosine_before_mean={sum(finite_cosines) / max(len(finite_cosines), 1):.8f}\n"
            f"effective_gradient_ratio_mean={sum(effective) / max(len(effective), 1):.8f}\n"
            f"effective_gradient_ratio_max={max(effective, default=0.0):.8f}\n"
            f"best_validation_f1={best_val:.8f}\n",
            encoding="utf-8",
        )
        return data
