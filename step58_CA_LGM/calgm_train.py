"""Step58 V1: adjacency-phase class-aware link gradient matching."""

from __future__ import annotations

import csv
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


TRAIN_FIELDS = (
    "epoch",
    "outer_loop",
    "node_gm_loss",
    "link_gm_loss",
    "positive_gradient_distance",
    "negative_gradient_distance",
    "lp_weight",
    "weighted_link_gm_loss",
)


def _write_training_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAIN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class CALGMHyDRO(GradientAuditHyDRO):
    """HyDRO plus positive/negative LGM during PGE updates only."""

    def __init__(self, *args, lp_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.lp_weight = float(lp_weight)
        if self.lp_weight <= 0.0:
            raise ValueError("lp_weight must be positive for the V1 intervention")

    def _link_gm_loss(
        self,
        *,
        model,
        real_features,
        message_adjacency,
        real_positive,
        real_negative,
        synthetic_features,
        synthetic_adjacency,
        synthetic_positive,
        synthetic_negative,
    ):
        model_parameters = list(model.parameters())
        real_hidden = _hidden_embedding(model, real_features, message_adjacency)
        real_pos_loss, real_neg_loss = _link_losses(
            real_hidden, real_positive, real_negative
        )
        real_pos_gradient = torch.autograd.grad(
            real_pos_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_neg_gradient = torch.autograd.grad(
            real_neg_loss,
            model_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        real_pos_gradient = tuple(
            None if value is None else value.detach()
            for value in real_pos_gradient
        )
        real_neg_gradient = tuple(
            None if value is None else value.detach()
            for value in real_neg_gradient
        )

        synthetic_hidden = _hidden_embedding(
            model, synthetic_features, synthetic_adjacency
        )
        synthetic_pos_loss, synthetic_neg_loss = _link_losses(
            synthetic_hidden, synthetic_positive, synthetic_negative
        )
        synthetic_pos_gradient = torch.autograd.grad(
            synthetic_pos_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        synthetic_neg_gradient = torch.autograd.grad(
            synthetic_neg_loss,
            model_parameters,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        positive_distance, _ = _gradient_distance(
            real_pos_gradient,
            synthetic_pos_gradient,
            model_parameters,
            self.distance_alpha,
        )
        negative_distance, _ = _gradient_distance(
            real_neg_gradient,
            synthetic_neg_gradient,
            model_parameters,
            self.distance_alpha,
        )
        return (
            0.5 * (positive_distance + negative_distance),
            positive_distance,
            negative_distance,
        )

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        training_csv = self.audit_output / "v1_training_diagnostics.csv"
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
            "Step58 V1 teacher: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}, "
            f"lp_weight={self.lp_weight:.6f}"
        )

        outer_loop, inner_loop = self.get_loops(args)
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        best_val = 0.0
        training_rows: list[dict[str, object]] = []
        for epoch in trange(args.epochs, desc="Step58 V1"):
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
            epoch_link_loss = 0.0
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
                total_loss = node_gm_loss
                link_gm_loss = torch.zeros((), device=self.device)
                positive_distance = torch.zeros((), device=self.device)
                negative_distance = torch.zeros((), device=self.device)
                if adjacency_phase:
                    (
                        link_gm_loss,
                        positive_distance,
                        negative_distance,
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
                    _write_training_rows(training_csv, training_rows)
                epoch_node_loss += float(node_gm_loss.detach())
                epoch_link_loss += float(link_gm_loss.detach())

                if args.beta > 0:
                    total_loss = total_loss + args.beta * regularization(
                        raw_synthetic_adjacency,
                        tensor2onehot(labels_syn.to("cpu")).to(
                            raw_synthetic_adjacency.device
                        ),
                        args,
                    )
                if args.spectral_objective and adjacency_phase:
                    total_loss = total_loss + spectral_loss
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                total_loss.backward()
                if adjacency_phase:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                inner_features = self.feat_syn.detach()
                # Keep the raw PGE output for checkpoint serialization.  The
                # GNN consumes the normalized propagation matrix, but the LP
                # evaluator expects a raw condensed adjacency and performs
                # its own normalization.  Saving the normalized matrix here
                # would therefore change the graph representation and cause
                # it to be normalized a second time during evaluation.
                inner_raw_adjacency = self.hyp.inference(inner_features)
                inner_normalized_adjacency = normalize_adj_tensor(
                    inner_raw_adjacency, sparse=False
                )
                for _ in range(inner_loop):
                    optimizer_model.zero_grad()
                    inner_output = model(inner_features, inner_normalized_adjacency)
                    inner_loss = F.nll_loss(inner_output, labels_syn)
                    inner_loss.backward()
                    optimizer_model.step()

            if epoch in args.checkpoints:
                self.adj_syn = inner_raw_adjacency
                data.adj_syn = self.adj_syn.detach()
                data.feat_syn = self.feat_syn.detach()
                data.labels_syn = labels_syn.detach()
                best_val = self.intermediate_evaluation_lp(
                    best_val,
                    (epoch_node_loss + self.lp_weight * epoch_link_loss)
                    / max(outer_loop, 1),
                )
        _write_training_rows(training_csv, training_rows)
        summary = self.audit_output / "v1_training_summary.txt"
        summary.write_text(
            "Step58 V1 CA-LGM training complete\n"
            f"epochs={args.epochs}\n"
            f"lp_weight={self.lp_weight}\n"
            "update_phase=adjacency_only\n"
            f"teacher_query_auc={teacher_metrics.query_auc:.8f}\n"
            f"teacher_query_gap={teacher_metrics.query_score_gap:.8f}\n"
            f"best_validation_f1={best_val:.8f}\n",
            encoding="utf-8",
        )
        return data
