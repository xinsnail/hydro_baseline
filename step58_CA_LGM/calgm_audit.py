"""Gradient compatibility audit for Step58 CA-LGM.

This module instruments the frozen HyDRO training loop without adding the
CA-LGM loss to the optimizer.  It measures whether node-gradient matching and
link-gradient matching request compatible updates of the synthetic features
and the hyperbolic structure generator.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import trange

from graphslim.condensation.hydro import HyDRO
from graphslim.condensation.link_aware_transport import (
    edge_pair_statistics,
    fit_link_teacher,
)
from graphslim.condensation.link_task_data import build_link_task_training_data
from graphslim.models import *  # noqa: F401,F403 - model names come from config
from graphslim.utils import (
    is_sparse_tensor,
    normalize_adj_tensor,
    regularization,
    tensor2onehot,
    to_tensor,
)


CSV_FIELDS = (
    "epoch",
    "phase",
    "outer_loop",
    "teacher_query_auc",
    "teacher_query_gap",
    "synthetic_positive_pairs",
    "synthetic_negative_pairs",
    "task_cosine",
    "condensation_cosine",
    "positive_gradient_distance",
    "negative_gradient_distance",
    "node_condensation_gradient_norm",
    "link_condensation_gradient_norm",
    "link_to_node_gradient_norm_ratio",
    "real_positive_gradient_norm",
    "synthetic_positive_gradient_norm",
    "real_negative_gradient_norm",
    "synthetic_negative_gradient_norm",
)


def _gradient_vector(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
    *,
    detach: bool,
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        if detach:
            value = value.detach()
        pieces.append(value.reshape(-1))
    if not pieces:
        raise ValueError("No parameters were supplied for gradient auditing")
    return torch.cat(pieces)


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator.detach()) <= eps:
        return float("nan")
    return float((left @ right / denominator.clamp_min(eps)).detach())


def _gradient_distance(
    real_gradients: Sequence[torch.Tensor | None],
    synthetic_gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
    alpha: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    real = _gradient_vector(real_gradients, parameters, detach=True)
    synthetic = _gradient_vector(synthetic_gradients, parameters, detach=False)
    real_norm = real.norm()
    synthetic_norm = synthetic.norm()
    direction = 1.0 - (real @ synthetic) / (
        real_norm * synthetic_norm + eps
    )
    magnitude = torch.abs(
        torch.log((real_norm + eps) / (synthetic_norm + eps))
    )
    distance = float(alpha) * direction + (1.0 - float(alpha)) * magnitude
    return distance, {
        "real_norm": float(real_norm.detach()),
        "synthetic_norm": float(synthetic_norm.detach()),
    }


def _subsample_edges(
    edges: torch.Tensor,
    limit: int,
    seed: int,
) -> torch.Tensor:
    edges = edges.detach().cpu().long().contiguous()
    if edges.size(1) <= int(limit):
        return edges
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.randperm(edges.size(1), generator=generator)[: int(limit)]
    return edges.index_select(1, selected).contiguous()


def _edge_logits(embedding: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    left = embedding.index_select(0, edges[0])
    right = embedding.index_select(0, edges[1])
    return (left * right).sum(dim=1)


def _link_losses(
    embedding: torch.Tensor,
    positive_edges: torch.Tensor,
    negative_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_logits = _edge_logits(embedding, positive_edges)
    negative_logits = _edge_logits(embedding, negative_edges)
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits, torch.zeros_like(negative_logits)
    )
    return positive_loss, negative_loss


def _hidden_embedding(model, features, adjacency) -> torch.Tensor:
    hidden, _ = model(features, adjacency, output_layer_features=True)
    # BaseGNN/GCN exposes a list of layer representations, whereas the SGC
    # used by HyDRO exposes one propagated pre-softmax tensor.  In both cases
    # the returned representation is produced by the same encoder parameters
    # used by node-gradient matching.
    if torch.is_tensor(hidden):
        return hidden
    if len(hidden) == 0:
        raise RuntimeError("The configured encoder did not expose hidden features")
    return hidden[0]


def _training_message_adjacency(training_data, node_count: int, device) -> SparseTensor:
    edges = training_data.message_edge_index.to(device)
    sparse = torch.sparse_coo_tensor(
        edges,
        torch.ones(edges.size(1), device=device),
        (int(node_count), int(node_count)),
        device=device,
    ).coalesce()
    return normalize_adj_tensor(sparse, sparse=True)


@torch.no_grad()
def _teacher_synthetic_pairs(
    teacher,
    synthetic_features: torch.Tensor,
    fraction: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = synthetic_features.detach().cpu().float()
    node_count = int(features.size(0))
    pairs = torch.triu_indices(node_count, node_count, offset=1)
    if pairs.size(1) < 2:
        raise ValueError("At least two synthetic node pairs are required")
    statistics = edge_pair_statistics(features, pairs)
    scores = teacher.forward_statistics(statistics)
    count = max(1, int(round(pairs.size(1) * float(fraction))))
    count = min(count, pairs.size(1) // 2)
    order = torch.argsort(scores)
    negative = pairs.index_select(1, order[:count]).to(device)
    positive = pairs.index_select(1, order[-count:]).to(device)
    return positive.contiguous(), negative.contiguous()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class GradientAuditHyDRO(HyDRO):
    """Frozen HyDRO optimizer with non-intervening CA-LGM diagnostics."""

    def __init__(
        self,
        *args,
        audit_output: Path,
        adjacency_points: Iterable[int],
        feature_points: Iterable[int],
        pair_limit: int,
        synthetic_pair_fraction: float,
        distance_alpha: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.audit_output = Path(audit_output)
        self.adjacency_points = frozenset(map(int, adjacency_points))
        self.feature_points = frozenset(map(int, feature_points))
        self.pair_limit = int(pair_limit)
        self.synthetic_pair_fraction = float(synthetic_pair_fraction)
        self.distance_alpha = float(distance_alpha)
        if self.pair_limit <= 0:
            raise ValueError("pair_limit must be positive")
        if not 0.0 < self.synthetic_pair_fraction < 0.5:
            raise ValueError("synthetic_pair_fraction must lie in (0, 0.5)")
        if not 0.0 <= self.distance_alpha <= 1.0:
            raise ValueError("distance_alpha must lie in [0, 1]")

    def _prepare_link_audit(self, data):
        args = self.args
        training_data = build_link_task_training_data(
            adjacency=data.adj_full,
            seed=int(args.paper_lapt_split_seed),
            validation_ratio=float(args.paper_lapt_validation_ratio),
            test_ratio=float(args.paper_lapt_test_ratio),
            query_ratio=float(args.paper_lapt_query_ratio),
            negative_ratio=float(args.lp_neg_ratio),
            hard_negative_ratio=float(args.paper_lapt_hard_negative_ratio),
        )
        teacher, metrics = fit_link_teacher(
            training_data=training_data,
            original_features=data.feat_full,
            hidden_channels=int(args.paper_lapt_teacher_hidden),
            pair_limit=int(args.paper_lapt_teacher_pair_limit),
            training_steps=int(args.paper_lapt_teacher_steps),
            batch_size=int(args.paper_lapt_teacher_batch_size),
            learning_rate=float(args.paper_lapt_teacher_lr),
            seed=int(args.paper_lapt_split_seed),
        )
        if metrics.query_auc < float(args.paper_lapt_min_teacher_auc):
            raise RuntimeError(
                "Link teacher gate failed: "
                f"query_auc={metrics.query_auc:.6f}"
            )
        positive = _subsample_edges(
            training_data.query_pos_edge_index,
            self.pair_limit,
            int(args.seed) + 58001,
        ).to(self.device)
        negative = _subsample_edges(
            training_data.query_neg_edge_index,
            self.pair_limit,
            int(args.seed) + 58003,
        ).to(self.device)
        message_adjacency = _training_message_adjacency(
            training_data,
            int(data.feat_full.shape[0]),
            self.device,
        )
        return training_data, teacher, metrics, positive, negative, message_adjacency

    def _audit(
        self,
        *,
        epoch: int,
        phase: str,
        outer_loop: int,
        model,
        node_gm_loss: torch.Tensor,
        real_features: torch.Tensor,
        real_adjacency,
        real_labels: torch.Tensor,
        real_positive: torch.Tensor,
        real_negative: torch.Tensor,
        message_adjacency,
        synthetic_features: torch.Tensor,
        synthetic_adjacency: torch.Tensor,
        teacher,
        teacher_metrics,
    ) -> dict[str, object]:
        model_parameters = list(model.parameters())
        was_training = model.training
        model.eval()

        real_hidden = _hidden_embedding(model, real_features, message_adjacency)
        real_pos_loss, real_neg_loss = _link_losses(
            real_hidden, real_positive, real_negative
        )
        real_pos_grad = torch.autograd.grad(
            real_pos_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_neg_grad = torch.autograd.grad(
            real_neg_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_pos_grad = tuple(
            None if value is None else value.detach() for value in real_pos_grad
        )
        real_neg_grad = tuple(
            None if value is None else value.detach() for value in real_neg_grad
        )

        synthetic_positive, synthetic_negative = _teacher_synthetic_pairs(
            teacher,
            synthetic_features,
            self.synthetic_pair_fraction,
            self.device,
        )
        synthetic_hidden = _hidden_embedding(
            model, synthetic_features, synthetic_adjacency
        )
        synthetic_pos_loss, synthetic_neg_loss = _link_losses(
            synthetic_hidden, synthetic_positive, synthetic_negative
        )
        synthetic_pos_grad = torch.autograd.grad(
            synthetic_pos_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        synthetic_neg_grad = torch.autograd.grad(
            synthetic_neg_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        positive_distance, positive_norms = _gradient_distance(
            real_pos_grad,
            synthetic_pos_grad,
            model_parameters,
            self.distance_alpha,
        )
        negative_distance, negative_norms = _gradient_distance(
            real_neg_grad,
            synthetic_neg_grad,
            model_parameters,
            self.distance_alpha,
        )
        link_gm_loss = 0.5 * (positive_distance + negative_distance)

        real_output = model(real_features, real_adjacency)
        train_index = torch.as_tensor(
            self.data.idx_train, device=self.device, dtype=torch.long
        )
        if real_output.size(0) == real_labels.size(0):
            node_task_loss = F.nll_loss(
                real_output.index_select(0, train_index),
                real_labels.index_select(0, train_index),
            )
        else:
            node_task_loss = F.nll_loss(real_output, real_labels)
        node_task_gradient = torch.autograd.grad(
            node_task_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_link_loss = 0.5 * (real_pos_loss + real_neg_loss)
        link_task_gradient = torch.autograd.grad(
            real_link_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        node_task_vector = _gradient_vector(
            node_task_gradient, model_parameters, detach=True
        )
        link_task_vector = _gradient_vector(
            link_task_gradient, model_parameters, detach=True
        )

        if phase == "adjacency":
            condensation_parameters = [
                parameter
                for parameter in self.hyp.parameters()
                if parameter.requires_grad
            ]
        elif phase == "feature":
            condensation_parameters = [self.feat_syn]
        else:
            raise ValueError(f"Unknown audit phase: {phase}")
        node_condensation_gradient = torch.autograd.grad(
            node_gm_loss,
            condensation_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        link_condensation_gradient = torch.autograd.grad(
            link_gm_loss,
            condensation_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        node_condensation_vector = _gradient_vector(
            node_condensation_gradient, condensation_parameters, detach=True
        )
        link_condensation_vector = _gradient_vector(
            link_condensation_gradient, condensation_parameters, detach=True
        )
        node_norm = float(node_condensation_vector.norm())
        link_norm = float(link_condensation_vector.norm())
        ratio = link_norm / max(node_norm, 1e-12)
        if was_training:
            model.train()
        return {
            "epoch": int(epoch),
            "phase": phase,
            "outer_loop": int(outer_loop),
            "teacher_query_auc": float(teacher_metrics.query_auc),
            "teacher_query_gap": float(teacher_metrics.query_score_gap),
            "synthetic_positive_pairs": int(synthetic_positive.size(1)),
            "synthetic_negative_pairs": int(synthetic_negative.size(1)),
            "task_cosine": _cosine(node_task_vector, link_task_vector),
            "condensation_cosine": _cosine(
                node_condensation_vector, link_condensation_vector
            ),
            "positive_gradient_distance": float(positive_distance.detach()),
            "negative_gradient_distance": float(negative_distance.detach()),
            "node_condensation_gradient_norm": node_norm,
            "link_condensation_gradient_norm": link_norm,
            "link_to_node_gradient_norm_ratio": ratio,
            "real_positive_gradient_norm": positive_norms["real_norm"],
            "synthetic_positive_gradient_norm": positive_norms["synthetic_norm"],
            "real_negative_gradient_norm": negative_norms["real_norm"],
            "synthetic_negative_gradient_norm": negative_norms["synthetic_norm"],
        }

    def _write_summary(self, rows, teacher_metrics) -> None:
        summary = self.audit_output / "gradient_audit_summary.txt"
        finite_rows = [
            row
            for row in rows
            if math.isfinite(float(row["condensation_cosine"]))
        ]
        lines = [
            "Step58 CA-LGM Gradient Compatibility Audit",
            f"teacher_query_auc={teacher_metrics.query_auc:.8f}",
            f"teacher_query_gap={teacher_metrics.query_score_gap:.8f}",
            f"audit_rows={len(rows)}",
        ]
        for phase in ("adjacency", "feature"):
            selected = [row for row in finite_rows if row["phase"] == phase]
            if not selected:
                lines.append(f"{phase}: no finite audit rows")
                continue
            task_mean = sum(float(row["task_cosine"]) for row in selected) / len(selected)
            cond_mean = sum(float(row["condensation_cosine"]) for row in selected) / len(selected)
            ratio_mean = sum(
                float(row["link_to_node_gradient_norm_ratio"]) for row in selected
            ) / len(selected)
            conflict = sum(float(row["condensation_cosine"]) < 0 for row in selected)
            lines.append(
                f"{phase}: task_cosine_mean={task_mean:.8f}, "
                f"condensation_cosine_mean={cond_mean:.8f}, "
                f"gradient_ratio_mean={ratio_mean:.8f}, "
                f"conflict_rows={conflict}/{len(selected)}"
            )
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        csv_path = self.audit_output / "gradient_audit.csv"
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
            training_data,
            teacher,
            teacher_metrics,
            real_positive,
            real_negative,
            message_adjacency,
        ) = self._prepare_link_audit(data)
        print(
            "Step58 teacher: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}"
        )

        outer_loop, inner_loop = self.get_loops(args)
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        rows: list[dict[str, object]] = []
        for epoch in trange(args.epochs, desc="Step58 audit"):
            model.initialize()
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
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
                    model,
                    adjacency,
                    features,
                    labels,
                    labels_syn,
                    args,
                )
                audit_phase = None
                if epoch in self.adjacency_points:
                    audit_phase = "adjacency"
                elif epoch in self.feature_points:
                    audit_phase = "feature"
                if audit_phase is not None and outer == 0:
                    row = self._audit(
                        epoch=epoch,
                        phase=audit_phase,
                        outer_loop=outer,
                        model=model,
                        node_gm_loss=node_gm_loss,
                        real_features=features,
                        real_adjacency=adjacency,
                        real_labels=labels,
                        real_positive=real_positive,
                        real_negative=real_negative,
                        message_adjacency=message_adjacency,
                        synthetic_features=self.feat_syn,
                        synthetic_adjacency=self.adj_syn,
                        teacher=teacher,
                        teacher_metrics=teacher_metrics,
                    )
                    rows.append(row)
                    _write_rows(csv_path, rows)
                    print(
                        "\nStep58 audit "
                        f"epoch={epoch} phase={audit_phase} "
                        f"C_task={float(row['task_cosine']):.6f} "
                        f"C_cond={float(row['condensation_cosine']):.6f} "
                        f"ratio={float(row['link_to_node_gradient_norm_ratio']):.6f}"
                    )

                total_loss = node_gm_loss
                if args.beta > 0:
                    total_loss = total_loss + args.beta * regularization(
                        raw_synthetic_adjacency,
                        tensor2onehot(labels_syn.to("cpu")).to(
                            raw_synthetic_adjacency.device
                        ),
                        args,
                    )
                if args.spectral_objective and epoch % 50 < 10:
                    total_loss = total_loss + spectral_loss
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                total_loss.backward()
                if epoch % 50 < 10:
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
        _write_rows(csv_path, rows)
        self._write_summary(rows, teacher_metrics)
        return data
