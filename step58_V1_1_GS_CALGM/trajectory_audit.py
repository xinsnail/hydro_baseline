"""Full PGE-gradient trajectory audit for Step58-V1.1.

The audit follows an unmodified HyDRO optimization trajectory.  At every PGE
update it measures the relationship between the complete HyDRO gradient and a
direction-only class-aware link-gradient matching objective.  The link loss is
never added to the optimizer in this stage.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import trange

from calgm_audit import (
    GradientAuditHyDRO,
    _cosine,
    _gradient_distance,
    _gradient_vector,
    _hidden_embedding,
    _link_losses,
    _teacher_synthetic_pairs,
)
from graphslim.models import *  # noqa: F401,F403 - configured model name
from graphslim.utils import normalize_adj_tensor, regularization, tensor2onehot, to_tensor


FIELDS = (
    "epoch",
    "outer_loop",
    "teacher_query_auc",
    "teacher_query_gap",
    "synthetic_positive_pairs",
    "synthetic_negative_pairs",
    "positive_direction_distance",
    "negative_direction_distance",
    "direction_loss",
    "gradient_cosine",
    "hydro_gradient_norm",
    "direction_gradient_norm",
    "raw_gradient_ratio",
    "rho",
    "lambda_max",
    "lambda_unclipped",
    "lambda_effective",
    "effective_gradient_ratio",
    "gradient_conflict",
    "gate_skipped",
)


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


class FullTrajectoryAuditHyDRO(GradientAuditHyDRO):
    """Unmodified HyDRO with a non-intervening PGE-gradient audit."""

    def __init__(self, *args, rho: float, lambda_max: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)
        self.lambda_max = float(lambda_max)
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")
        if self.lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive")

    def _audit_pge_update(
        self,
        *,
        epoch: int,
        outer_loop: int,
        model,
        hydro_loss: torch.Tensor,
        real_features: torch.Tensor,
        message_adjacency,
        real_positive: torch.Tensor,
        real_negative: torch.Tensor,
        synthetic_adjacency: torch.Tensor,
        teacher,
        teacher_metrics,
    ) -> dict[str, object]:
        model_parameters = list(model.parameters())
        pge_parameters = [
            parameter for parameter in self.hyp.parameters() if parameter.requires_grad
        ]
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

        synthetic_positive, synthetic_negative = _teacher_synthetic_pairs(
            teacher,
            self.feat_syn,
            self.synthetic_pair_fraction,
            self.device,
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
        direction_loss = 0.5 * (positive_direction + negative_direction)

        hydro_gradient = torch.autograd.grad(
            hydro_loss,
            pge_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        direction_gradient = torch.autograd.grad(
            direction_loss,
            pge_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        hydro_vector = _gradient_vector(
            hydro_gradient, pge_parameters, detach=True
        )
        direction_vector = _gradient_vector(
            direction_gradient, pge_parameters, detach=True
        )
        hydro_norm = float(hydro_vector.norm())
        direction_norm = float(direction_vector.norm())
        cosine = _cosine(hydro_vector, direction_vector)
        eps = 1e-12
        raw_ratio = direction_norm / max(hydro_norm, eps)
        lambda_unclipped = self.rho * hydro_norm / max(direction_norm, eps)
        conflict = bool(math.isfinite(cosine) and cosine < 0.0)
        invalid = not math.isfinite(cosine)
        lambda_effective = (
            0.0
            if conflict or invalid
            else min(self.lambda_max, lambda_unclipped)
        )
        effective_ratio = (
            lambda_effective * direction_norm / max(hydro_norm, eps)
        )
        if was_training:
            model.train()
        return {
            "epoch": int(epoch),
            "outer_loop": int(outer_loop),
            "teacher_query_auc": float(teacher_metrics.query_auc),
            "teacher_query_gap": float(teacher_metrics.query_score_gap),
            "synthetic_positive_pairs": int(synthetic_positive.size(1)),
            "synthetic_negative_pairs": int(synthetic_negative.size(1)),
            "positive_direction_distance": float(positive_direction.detach()),
            "negative_direction_distance": float(negative_direction.detach()),
            "direction_loss": float(direction_loss.detach()),
            "gradient_cosine": float(cosine),
            "hydro_gradient_norm": hydro_norm,
            "direction_gradient_norm": direction_norm,
            "raw_gradient_ratio": raw_ratio,
            "rho": self.rho,
            "lambda_max": self.lambda_max,
            "lambda_unclipped": lambda_unclipped,
            "lambda_effective": lambda_effective,
            "effective_gradient_ratio": effective_ratio,
            "gradient_conflict": int(conflict),
            "gate_skipped": int(conflict or invalid),
        }

    def _write_summary(self, rows, teacher_metrics) -> None:
        finite = [
            row for row in rows if math.isfinite(float(row["gradient_cosine"]))
        ]
        cosines = [float(row["gradient_cosine"]) for row in finite]
        conflicts = sum(int(row["gradient_conflict"]) for row in finite)
        conflict_fraction = conflicts / max(len(finite), 1)
        recommendation = (
            "skip_negative_cosine"
            if conflict_fraction <= 0.10
            else "pcgrad_candidate"
        )
        lines = [
            "Step58-V1.1 full PGE-gradient trajectory audit",
            f"teacher_query_auc={teacher_metrics.query_auc:.8f}",
            f"teacher_query_gap={teacher_metrics.query_score_gap:.8f}",
            f"audit_rows={len(rows)}",
            f"finite_rows={len(finite)}",
            f"rho={self.rho:.8f}",
            f"lambda_max={self.lambda_max:.8f}",
            f"gradient_cosine_mean={statistics.fmean(cosines):.8f}",
            f"gradient_cosine_median={statistics.median(cosines):.8f}",
            f"gradient_cosine_min={min(cosines):.8f}",
            f"gradient_cosine_max={max(cosines):.8f}",
            f"conflict_updates={conflicts}/{len(finite)}",
            f"conflict_fraction={conflict_fraction:.8f}",
            f"hydro_gradient_norm_mean={_mean(finite, 'hydro_gradient_norm'):.8f}",
            f"direction_gradient_norm_mean={_mean(finite, 'direction_gradient_norm'):.8f}",
            f"raw_gradient_ratio_mean={_mean(finite, 'raw_gradient_ratio'):.8f}",
            f"lambda_effective_mean={_mean(finite, 'lambda_effective'):.8f}",
            f"effective_gradient_ratio_mean={_mean(finite, 'effective_gradient_ratio'):.8f}",
            f"effective_gradient_ratio_max={max(float(row['effective_gradient_ratio']) for row in finite):.8f}",
            f"recommended_conflict_control={recommendation}",
        ]
        for start in (0, 50, 100):
            block = [
                row for row in finite if start <= int(row["epoch"]) < start + 10
            ]
            if not block:
                continue
            block_conflicts = sum(int(row["gradient_conflict"]) for row in block)
            lines.append(
                f"block_{start}_{start + 9}: rows={len(block)}, "
                f"cosine_mean={_mean(block, 'gradient_cosine'):.8f}, "
                f"conflicts={block_conflicts}/{len(block)}, "
                f"raw_ratio_mean={_mean(block, 'raw_gradient_ratio'):.8f}, "
                f"lambda_mean={_mean(block, 'lambda_effective'):.8f}, "
                f"effective_ratio_mean={_mean(block, 'effective_gradient_ratio'):.8f}"
            )
        (self.audit_output / "trajectory_audit_summary.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        csv_path = self.audit_output / "trajectory_audit.csv"
        if csv_path.exists():
            csv_path.unlink()
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
            "Step58-V1.1 trajectory teacher: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}"
        )

        outer_loop, inner_loop = self.get_loops(args)
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        rows: list[dict[str, object]] = []
        for epoch in trange(args.epochs, desc="Step58-V1.1 audit"):
            model.initialize()
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            adjacency_phase = epoch % 50 < 10
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
                self.adj_syn = normalize_adj_tensor(raw_adjacency, sparse=False)
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
                        tensor2onehot(labels_syn.to("cpu")).to(raw_adjacency.device),
                        args,
                    )
                if args.spectral_objective and adjacency_phase:
                    hydro_loss = hydro_loss + spectral_loss

                if adjacency_phase:
                    row = self._audit_pge_update(
                        epoch=epoch,
                        outer_loop=outer,
                        model=model,
                        hydro_loss=hydro_loss,
                        real_features=features,
                        message_adjacency=message_adjacency,
                        real_positive=real_positive,
                        real_negative=real_negative,
                        synthetic_adjacency=self.adj_syn,
                        teacher=teacher,
                        teacher_metrics=teacher_metrics,
                    )
                    rows.append(row)
                    _append_row(csv_path, row)

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                hydro_loss.backward()
                if adjacency_phase:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                inner_features = self.feat_syn.detach()
                inner_adjacency = normalize_adj_tensor(
                    self.hyp.inference(inner_features), sparse=False
                )
                for _ in range(inner_loop):
                    optimizer_model.zero_grad()
                    inner_output = model(inner_features, inner_adjacency)
                    inner_loss = F.nll_loss(inner_output, labels_syn)
                    inner_loss.backward()
                    optimizer_model.step()
            if adjacency_phase:
                block = rows[-outer_loop:]
                print(
                    "\nStep58-V1.1 audit "
                    f"epoch={epoch} "
                    f"C_mean={_mean(block, 'gradient_cosine'):.6f} "
                    f"conflicts={sum(int(row['gradient_conflict']) for row in block)}/{len(block)} "
                    f"ratio_mean={_mean(block, 'raw_gradient_ratio'):.6f}"
                )
        self._write_summary(rows, teacher_metrics)
        return data
