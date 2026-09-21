"""Offline PGE-gradient audit for Step58-GBA-v2 snapshots."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

from calgm_train import CALGMHyDRO
from graphslim.condensation.utils import match_loss
from graphslim.models import *  # noqa: F401,F403 - selected by config
from graphslim.utils import normalize_adj_tensor, regularization, tensor2onehot, to_tensor


FIELDS = (
    "snapshot", "kind", "epoch_index", "epoch_display", "adjacency_phase",
    "selected_as_best", "hydro_gradient_norm", "link_gradient_norm",
    "weighted_link_gradient_norm", "raw_link_injection_ratio",
    "gradient_dot", "gradient_cosine", "gradient_conflict", "finite",
    "replay_raw_adjacency_relative_error", "replay_node_loss_absolute_error",
    "replay_link_loss_absolute_error", "node_gm_loss", "link_gm_loss",
    "hydro_loss",
)


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_device(value, device):
    if isinstance(value, SparseTensor):
        return value.to(device)
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _message_adjacency(payload, device) -> SparseTensor:
    value = payload["value"]
    return SparseTensor(
        row=payload["row"].to(device),
        col=payload["col"].to(device),
        value=None if value is None else value.to(device),
        sparse_sizes=tuple(payload["sparse_sizes"]),
    )


def _gradient_dot(first, second, device):
    result = torch.zeros((), device=device)
    for left, right in zip(first, second):
        if left is not None and right is not None:
            result = result + (left * right).sum()
    return result


def _gradient_norm(values, device):
    return _gradient_dot(values, values, device).clamp_min(0.0).sqrt()


def _relative_error(reference, value):
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(value - reference) / denominator).detach())


def _node_gm_from_batches(agent, model, features, labels, labels_syn, batches):
    data = agent.data
    args = agent.args
    feat_syn = agent.feat_syn
    adj_syn = agent.adj_syn
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    hard_labels = labels
    soft_labels = F.one_hot(labels, num_classes=data.nclass).float()
    if labels_syn.dim() == 1:
        hard_labels_syn = labels_syn
        soft_labels_syn = F.one_hot(
            labels_syn, num_classes=data.nclass
        ).float()
    else:
        hard_labels_syn = torch.argmax(labels_syn, dim=-1)
        soft_labels_syn = labels_syn

    result = torch.zeros((), device=agent.device)
    for record in batches:
        class_index = int(record["class_index"])
        batch_size = int(record["batch_size"])
        n_id = record["n_id"].to(agent.device)
        adjs = [_to_device(value, agent.device) for value in record["adjs"]]
        input_real = features[n_id].to(agent.device)
        labels_real = soft_labels[n_id[:batch_size]].to(agent.device)
        output_real = model(input_real, adjs)
        loss_real = loss_fn(output_real, labels_real)
        real_gradient = torch.autograd.grad(
            loss_real, model.parameters(), retain_graph=True
        )
        real_gradient = [value.detach().clone() for value in real_gradient]

        output_syn = model(feat_syn, adj_syn)
        mask = hard_labels_syn == class_index
        loss_syn = loss_fn(output_syn[mask], soft_labels_syn[mask])
        synthetic_gradient = torch.autograd.grad(
            loss_syn, model.parameters(), create_graph=True
        )
        coefficient = agent.num_class_dict[class_index] / agent.nnodes_syn
        result = result + coefficient * match_loss(
            synthetic_gradient, real_gradient, args, device=agent.device
        )
    return result


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_offline_audit(*, data, args, trace_root: Path, output: Path,
                      pair_limit: int, synthetic_pair_fraction: float,
                      distance_alpha: float, lp_weight: float):
    context = _load(trace_root / "offline_audit_context.pt")
    device = args.device
    if args.setting == "trans":
        features, adjacency, labels = to_tensor(
            data.feat_full, data.adj_full,
            label=data.labels_full, device=device,
        )
    else:
        features, adjacency, labels = to_tensor(
            data.feat_train, data.adj_train,
            label=data.labels_train, device=device,
        )
    adjacency = normalize_adj_tensor(adjacency, sparse=True)
    real_positive = context["real_positive"].to(device)
    real_negative = context["real_negative"].to(device)
    message_adjacency = _message_adjacency(
        context["message_adjacency"], device
    )

    snapshot_root = trace_root / "offline_snapshots"
    paths = sorted(snapshot_root.glob("state_epoch_*_outer_00.pt"))
    best_path = snapshot_root / "best_checkpoint_state.pt"
    if best_path.exists():
        paths.append(best_path)
    if not paths:
        raise FileNotFoundError(f"No offline snapshots found in {snapshot_root}")

    rows = []
    for path in paths:
        snapshot = _load(path)
        agent = CALGMHyDRO(
            setting=args.setting,
            reduction_rate=args.reduction_rate,
            data=data,
            args=args,
            audit_output=output,
            adjacency_points=(), feature_points=(),
            pair_limit=pair_limit,
            synthetic_pair_fraction=synthetic_pair_fraction,
            distance_alpha=distance_alpha,
            lp_weight=lp_weight,
        )
        agent.hyp.load_state_dict(snapshot["hyp_state_dict"])
        agent.hyp.Lx_inv = _to_device(snapshot["hyp_lx_inv"], device)
        agent.hyp.train(bool(snapshot["hyp_training"]))
        agent.feat_syn.data.copy_(snapshot["feat_syn"].to(device))
        labels_syn = snapshot["labels_syn"].to(device)

        model_class = globals()[args.condense_model]
        model = model_class(
            agent.feat_syn.shape[1], args.hidden, data.nclass, args
        ).to(device)
        model.load_state_dict(snapshot["model_state_dict"])
        model.train(bool(snapshot["model_training"]))

        sampled_lx = _to_device(snapshot["sampled_lx"], device)
        raw_adjacency, spectral_loss = agent.hyp(
            agent.feat_syn, Lx=sampled_lx
        )
        agent.adj_syn = normalize_adj_tensor(raw_adjacency, sparse=False)
        model = agent.check_bn(model)
        node_loss = _node_gm_from_batches(
            agent, model, features, labels, labels_syn,
            snapshot["class_batches"],
        )
        link_loss, _, _ = agent._link_gm_loss(
            model=model,
            real_features=features,
            message_adjacency=message_adjacency,
            real_positive=real_positive,
            real_negative=real_negative,
            synthetic_features=agent.feat_syn,
            synthetic_adjacency=agent.adj_syn,
            synthetic_positive=snapshot["synthetic_positive"].to(device),
            synthetic_negative=snapshot["synthetic_negative"].to(device),
        )

        hydro_loss = node_loss
        if args.beta > 0:
            hydro_loss = hydro_loss + args.beta * regularization(
                raw_adjacency,
                tensor2onehot(labels_syn.to("cpu")).to(raw_adjacency.device),
                args,
            )
        if args.spectral_objective and bool(snapshot["adjacency_phase"]):
            hydro_loss = hydro_loss + spectral_loss

        parameters = tuple(
            parameter for parameter in agent.hyp.parameters()
            if parameter.requires_grad
        )
        hydro_gradient = torch.autograd.grad(
            hydro_loss, parameters, retain_graph=True, allow_unused=True
        )
        link_gradient = torch.autograd.grad(
            link_loss, parameters, retain_graph=True, allow_unused=True
        )
        hydro_norm = float(_gradient_norm(hydro_gradient, device).detach())
        link_norm = float(_gradient_norm(link_gradient, device).detach())
        dot = float(_gradient_dot(hydro_gradient, link_gradient, device).detach())
        denominator = hydro_norm * link_norm
        cosine = dot / denominator if denominator > 1e-12 else float("nan")
        ratio = lp_weight * link_norm / max(hydro_norm, 1e-12)
        finite = all(
            math.isfinite(value)
            for value in (hydro_norm, link_norm, dot, ratio)
        ) and (math.isfinite(cosine) or denominator <= 1e-12)

        reference_raw = snapshot["reference_raw_adjacency"].to(device)
        reference_node = snapshot["reference_node_gm_loss"]
        reference_link = snapshot["reference_link_gm_loss"]
        rows.append(
            {
                "snapshot": path.name,
                "kind": snapshot["kind"],
                "epoch_index": snapshot["epoch_index"],
                "epoch_display": snapshot["epoch_display"],
                "adjacency_phase": int(snapshot["adjacency_phase"]),
                "selected_as_best": int(snapshot["selected_as_best"]),
                "hydro_gradient_norm": hydro_norm,
                "link_gradient_norm": link_norm,
                "weighted_link_gradient_norm": lp_weight * link_norm,
                "raw_link_injection_ratio": ratio,
                "gradient_dot": dot,
                "gradient_cosine": cosine,
                "gradient_conflict": int(math.isfinite(cosine) and cosine < 0),
                "finite": int(finite),
                "replay_raw_adjacency_relative_error": _relative_error(
                    reference_raw, raw_adjacency
                ),
                "replay_node_loss_absolute_error": (
                    "" if reference_node is None
                    else abs(float(node_loss.detach()) - float(reference_node))
                ),
                "replay_link_loss_absolute_error": (
                    "" if reference_link is None
                    else abs(float(link_loss.detach()) - float(reference_link))
                ),
                "node_gm_loss": float(node_loss.detach()),
                "link_gm_loss": float(link_loss.detach()),
                "hydro_loss": float(hydro_loss.detach()),
            }
        )
        del agent, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "offline_gradient_diagnostics.csv"
    _write_rows(csv_path, rows)
    finite_rows = [
        row for row in rows
        if int(row["finite"]) == 1
        and math.isfinite(float(row["gradient_cosine"]))
    ]
    ratios = [float(row["raw_link_injection_ratio"]) for row in finite_rows]
    cosines = [float(row["gradient_cosine"]) for row in finite_rows]
    replay_errors = [
        float(row["replay_raw_adjacency_relative_error"])
        for row in rows
    ]
    (output / "offline_audit_summary.txt").write_text(
        "Step58-GBA-v2 offline gradient audit complete\n"
        "training_process_modified=false\n"
        f"snapshots={len(rows)}\n"
        f"finite_snapshots={len(finite_rows)}\n"
        f"ratio_mean={statistics.fmean(ratios) if ratios else float('nan'):.8f}\n"
        f"ratio_median={statistics.median(ratios) if ratios else float('nan'):.8f}\n"
        f"cosine_mean={statistics.fmean(cosines) if cosines else float('nan'):.8f}\n"
        f"cosine_median={statistics.median(cosines) if cosines else float('nan'):.8f}\n"
        f"conflict_fraction={statistics.fmean([float(v < 0) for v in cosines]) if cosines else float('nan'):.8f}\n"
        f"max_replay_raw_adjacency_relative_error={max(replay_errors):.8e}\n",
        encoding="utf-8",
    )
    print(f"Saved {csv_path}")
    print(output / "offline_audit_summary.txt")
    return rows
