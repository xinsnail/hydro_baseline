"""Snapshot-only replay of fixed Step58 CA-LGM.

This training loop keeps the original CA-LGM objective and optimizer update.
It never calls ``autograd.grad`` for the GBA diagnostics.  At selected points
it serializes the already available state and sampled class batches so a
separate process can reconstruct the two PGE gradients after training.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import trange

from calgm_audit import _teacher_synthetic_pairs
from calgm_train import CALGMHyDRO, TRAIN_FIELDS, _write_training_rows
from graphslim.models import *  # noqa: F401,F403 - selected by config
from graphslim.utils import normalize_adj_tensor, regularization, tensor2onehot, to_tensor


MANIFEST_FIELDS = (
    "snapshot", "kind", "epoch_index", "epoch_display", "outer_loop",
    "adjacency_phase", "selected_as_best", "reference_node_gm_loss",
    "reference_link_gm_loss", "reference_total_loss",
)


def _state_dict_cpu(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _tensor_cpu(value):
    if value is None:
        return None
    if isinstance(value, SparseTensor):
        return value.cpu()
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _sparse_payload(value: SparseTensor) -> dict[str, object]:
    row, col, edge_value = value.coo()
    return {
        "row": row.cpu(),
        "col": col.cpu(),
        "value": None if edge_value is None else edge_value.cpu(),
        "sparse_sizes": tuple(map(int, value.sparse_sizes())),
    }


def _batch_payload(batch_size, n_id, adjs) -> dict[str, object]:
    return {
        "batch_size": int(batch_size),
        "n_id": n_id.detach().cpu().clone(),
        # GCondBase.train_class only forwards adj[0] from every PyG Adj.
        "adjs": [_tensor_cpu(adj[0]) for adj in adjs],
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class SnapshotOnlyCALGM(CALGMHyDRO):
    """Frozen fixed-weight CA-LGM with serialization but no GBA backward."""

    def __init__(self, *args, trace_points, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_points = frozenset(int(value) for value in trace_points)

    def _pre_forward_state(self, *, epoch_index, model, labels_syn):
        return {
            "format_version": 1,
            "epoch_index": int(epoch_index),
            "epoch_display": int(epoch_index) + 1,
            "hyp_state_dict": _state_dict_cpu(self.hyp),
            "hyp_training": bool(self.hyp.training),
            "hyp_lx_inv": _tensor_cpu(self.hyp.Lx_inv),
            "feat_syn": self.feat_syn.detach().cpu().clone(),
            "labels_syn": labels_syn.detach().cpu().clone(),
            "model_state_dict": _state_dict_cpu(model),
            "model_training": bool(model.training),
            "rng_state": _rng_state(),
        }

    def _save_snapshot(
        self,
        *,
        snapshot_root: Path,
        manifest_rows: list[dict[str, object]],
        filename: str,
        kind: str,
        pre_state: dict[str, object],
        outer: int,
        adjacency_phase: bool,
        sampled_lx,
        class_batches,
        synthetic_positive,
        synthetic_negative,
        raw_adjacency,
        normalized_adjacency,
        node_loss=None,
        link_loss=None,
        total_loss=None,
        selected_as_best: bool = False,
    ) -> None:
        payload = dict(pre_state)
        payload.update(
            {
                "kind": kind,
                "outer_loop": int(outer),
                "adjacency_phase": bool(adjacency_phase),
                "sampled_lx": _tensor_cpu(sampled_lx),
                "class_batches": class_batches,
                "synthetic_positive": _tensor_cpu(synthetic_positive),
                "synthetic_negative": _tensor_cpu(synthetic_negative),
                "reference_raw_adjacency": _tensor_cpu(raw_adjacency),
                "reference_normalized_adjacency": _tensor_cpu(normalized_adjacency),
                "reference_node_gm_loss": (
                    None if node_loss is None else float(node_loss.detach())
                ),
                "reference_link_gm_loss": (
                    None if link_loss is None else float(link_loss.detach())
                ),
                "reference_total_loss": (
                    None if total_loss is None else float(total_loss.detach())
                ),
                "selected_as_best": bool(selected_as_best),
            }
        )
        path = snapshot_root / filename
        torch.save(payload, path)
        manifest_rows.append(
            {
                "snapshot": str(path.name),
                "kind": kind,
                "epoch_index": payload["epoch_index"],
                "epoch_display": payload["epoch_display"],
                "outer_loop": int(outer),
                "adjacency_phase": int(adjacency_phase),
                "selected_as_best": int(selected_as_best),
                "reference_node_gm_loss": payload["reference_node_gm_loss"],
                "reference_link_gm_loss": payload["reference_link_gm_loss"],
                "reference_total_loss": payload["reference_total_loss"],
            }
        )

    def reduce(self, data, verbose=True):
        args = self.args
        self.audit_output.mkdir(parents=True, exist_ok=True)
        snapshot_root = self.audit_output / "offline_snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.audit_output / "offline_snapshot_manifest.csv"
        training_csv = self.audit_output / "v1_training_diagnostics.csv"

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
        torch.save(
            {
                "format_version": 1,
                "real_positive": real_positive.detach().cpu(),
                "real_negative": real_negative.detach().cpu(),
                "message_adjacency": _sparse_payload(message_adjacency),
                "teacher_query_auc": float(teacher_metrics.query_auc),
                "teacher_query_gap": float(teacher_metrics.query_score_gap),
                "lp_weight": float(self.lp_weight),
            },
            self.audit_output / "offline_audit_context.pt",
        )
        print(
            "Step58-GBA-v2 snapshot-only CA-LGM: "
            f"query_auc={teacher_metrics.query_auc:.6f}, "
            f"query_gap={teacher_metrics.query_score_gap:.6f}, "
            f"lp_weight={self.lp_weight:.6f}, "
            f"trace_points={sorted(self.trace_points)}"
        )

        outer_loop, inner_loop = self.get_loops(args)
        model_class = globals()[args.condense_model]
        model = model_class(
            feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(self.device)
        best_val = 0.0
        training_rows: list[dict[str, object]] = []
        manifest_rows: list[dict[str, object]] = []

        for epoch_index in trange(args.epochs, desc="Step58-GBA-v2 trace"):
            model.initialize()
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            adjacency_phase = epoch_index % 50 < 10
            capture_epoch = (
                epoch_index in self.trace_points
                or epoch_index in set(map(int, args.checkpoints))
            )
            pair_positive = pair_negative = None
            if adjacency_phase or capture_epoch:
                pair_positive, pair_negative = _teacher_synthetic_pairs(
                    teacher, self.feat_syn,
                    self.synthetic_pair_fraction, self.device,
                )
            if adjacency_phase:
                synthetic_positive, synthetic_negative = (
                    pair_positive, pair_negative
                )

            epoch_node_loss = 0.0
            epoch_link_loss = 0.0
            checkpoint_capture = None

            for outer in range(outer_loop):
                capture = capture_epoch and outer == 0
                pre_state = (
                    self._pre_forward_state(
                        epoch_index=epoch_index, model=model, labels_syn=labels_syn
                    )
                    if capture else None
                )
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
                    sampled_adjacency = adjacency
                    raw_synthetic_adjacency, spectral_loss = self.hyp(
                        self.feat_syn, Lx=adjacency
                    )
                self.adj_syn = normalize_adj_tensor(
                    raw_synthetic_adjacency, sparse=False
                )
                model = self.check_bn(model)

                captured_batches: list[dict[str, object]] = []
                original_sampler = data.retrieve_class_sampler
                if capture:
                    def traced_sampler(c, adj, sampler_args, num=256):
                        result = original_sampler(c, adj, sampler_args, num)
                        captured_batches.append(_batch_payload(*result))
                        captured_batches[-1]["class_index"] = int(c)
                        return result
                    data.retrieve_class_sampler = traced_sampler
                try:
                    node_gm_loss = self.train_class(
                        model, adjacency, features, labels, labels_syn, args
                    )
                finally:
                    if capture:
                        data.retrieve_class_sampler = original_sampler

                total_loss = node_gm_loss
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
                            "epoch": epoch_index,
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
                    # Keep the same write cadence as the original CA-LGM.
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

                if capture:
                    filename = f"state_epoch_{epoch_index:04d}_outer_00.pt"
                    self._save_snapshot(
                        snapshot_root=snapshot_root,
                        manifest_rows=manifest_rows,
                        filename=filename,
                        kind="pre_update",
                        pre_state=pre_state,
                        outer=outer,
                        adjacency_phase=adjacency_phase,
                        sampled_lx=sampled_adjacency,
                        class_batches=captured_batches,
                        synthetic_positive=pair_positive,
                        synthetic_negative=pair_negative,
                        raw_adjacency=raw_synthetic_adjacency,
                        normalized_adjacency=self.adj_syn,
                        node_loss=node_gm_loss,
                        link_loss=(link_gm_loss if adjacency_phase else None),
                        total_loss=total_loss,
                    )
                    _write_manifest(manifest_path, manifest_rows)
                    checkpoint_capture = {
                        "class_batches": captured_batches,
                        "sampled_lx": sampled_adjacency,
                        "synthetic_positive": pair_positive,
                        "synthetic_negative": pair_negative,
                    }

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

            if epoch_index in args.checkpoints:
                self.adj_syn = inner_raw_adjacency
                data.adj_syn = self.adj_syn.detach()
                data.feat_syn = self.feat_syn.detach()
                data.labels_syn = labels_syn.detach()
                previous_best = best_val
                best_val = self.intermediate_evaluation_lp(
                    best_val,
                    (epoch_node_loss + self.lp_weight * epoch_link_loss)
                    / max(outer_loop, 1),
                )
                if best_val > previous_best and checkpoint_capture is not None:
                    best_state = self._pre_forward_state(
                        epoch_index=epoch_index, model=model, labels_syn=labels_syn
                    )
                    self._save_snapshot(
                        snapshot_root=snapshot_root,
                        manifest_rows=manifest_rows,
                        filename="best_checkpoint_state.pt",
                        kind="best_checkpoint_post",
                        pre_state=best_state,
                        outer=outer_loop - 1,
                        adjacency_phase=adjacency_phase,
                        sampled_lx=checkpoint_capture["sampled_lx"],
                        class_batches=checkpoint_capture["class_batches"],
                        synthetic_positive=checkpoint_capture["synthetic_positive"],
                        synthetic_negative=checkpoint_capture["synthetic_negative"],
                        raw_adjacency=inner_raw_adjacency,
                        normalized_adjacency=inner_normalized_adjacency,
                        selected_as_best=True,
                    )
                    _write_manifest(manifest_path, manifest_rows)

        _write_training_rows(training_csv, training_rows)
        _write_manifest(manifest_path, manifest_rows)
        (self.audit_output / "trace_summary.txt").write_text(
            "Step58-GBA-v2 snapshot-only CA-LGM complete\n"
            "online_gradient_audit=false\n"
            "extra_autograd_grad_calls=0\n"
            f"epochs={args.epochs}\n"
            f"lp_weight={self.lp_weight:.8f}\n"
            f"best_validation_f1={best_val:.8f}\n"
            f"snapshots={len(manifest_rows)}\n",
            encoding="utf-8",
        )
        return data
