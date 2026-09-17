"""Paper-protocol bridge for the Step19 LAPT structural target.

The module deliberately leaves HyDRO training and the paper LP evaluator
untouched.  It builds a link-aware target from a deterministic training-only
edge split and rank-calibrates that target to the saved HyDRO adjacency.  Rank
calibration preserves the exact HyDRO edge-weight histogram while allowing
validation-selected changes to pair ordering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from graphslim.condensation.link_aware_transport import (
    binary_auc,
    fit_link_teacher,
    link_aware_balanced_assignment,
    original_prototype_similarities,
    parse_blends,
)
from graphslim.condensation.link_task_data import LinkTaskTrainingData


def _dense_cpu_features(value) -> torch.Tensor:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.is_sparse:
        value = value.to_dense()
    if value.dim() != 2:
        raise ValueError(f"Expected a two-dimensional feature matrix, got {value.shape}")
    return value.detach().cpu().float().contiguous()


def _cpu_labels(value) -> torch.Tensor:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.dim() == 2:
        value = value.argmax(dim=1)
    if value.dim() != 1:
        raise ValueError(f"Expected hard labels, got {value.shape}")
    return value.long().contiguous()


def _dense_cpu_adjacency(value) -> torch.Tensor:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.is_sparse:
        value = value.to_dense()
    value = value.detach().cpu().float().contiguous()
    if value.dim() != 2 or value.size(0) != value.size(1):
        raise ValueError(f"Expected a square synthetic adjacency, got {value.shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Synthetic adjacency contains NaN or infinite values")
    value = ((value + value.T) / 2.0).clamp_min(0.0)
    value.fill_diagonal_(0.0)
    return value


def _support_propagation(
    features: torch.Tensor,
    message_edge_index: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Propagate features using only the support-message graph."""

    features = _dense_cpu_features(features)
    if int(steps) <= 0:
        raise ValueError("Structure-context propagation steps must be positive")
    edge_index = message_edge_index.detach().cpu().long().contiguous()
    node_count = int(features.size(0))
    loops = torch.arange(node_count, dtype=torch.long)
    indices = torch.cat(
        (edge_index, torch.stack((loops, loops), dim=0)), dim=1
    )
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1), dtype=features.dtype),
        (node_count, node_count),
    ).coalesce()
    row, col = adjacency.indices()
    values = adjacency.values()
    degree = torch.zeros(node_count, dtype=features.dtype)
    degree.index_add_(0, row, values)
    inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    normalized = torch.sparse_coo_tensor(
        adjacency.indices(),
        values * inv_sqrt.index_select(0, row) * inv_sqrt.index_select(0, col),
        adjacency.size(),
    ).coalesce()
    propagated = features
    for _ in range(int(steps)):
        propagated = torch.sparse.mm(normalized, propagated)
    return propagated.contiguous()


def _synthetic_propagation(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    features = _dense_cpu_features(features)
    adjacency = _dense_cpu_adjacency(adjacency)
    if adjacency.size(0) != features.size(0):
        raise ValueError("Synthetic features and adjacency have different node counts")
    weighted = adjacency + torch.eye(adjacency.size(0), dtype=adjacency.dtype)
    degree = weighted.sum(dim=1).clamp_min(1e-12)
    inv_sqrt = degree.pow(-0.5)
    normalized = inv_sqrt[:, None] * weighted * inv_sqrt[None, :]
    propagated = features
    for _ in range(int(steps)):
        propagated = normalized @ propagated
    return propagated.contiguous()


def structure_context_similarities(
    training_data: LinkTaskTrainingData,
    original_features,
    synthetic_features,
    synthetic_adjacency,
    steps: int,
    batch_nodes: int,
) -> torch.Tensor:
    """Return support-graph structural-context similarity for N x N'.

    Original representations are propagated only on support edges. Synthetic
    representations are propagated on the saved condensed topology. Both stay
    in the original feature space, which makes cosine transfer well-defined.
    """

    if int(batch_nodes) <= 0:
        raise ValueError("Structure-context batch size must be positive")
    original_context = _support_propagation(
        original_features, training_data.message_edge_index, int(steps)
    )
    synthetic_context = _synthetic_propagation(
        synthetic_features, synthetic_adjacency, int(steps)
    )
    if original_context.size(1) != synthetic_context.size(1):
        raise ValueError("Original and synthetic feature widths differ")
    synthetic_normalized = F.normalize(synthetic_context, dim=1, eps=1e-8)
    blocks = []
    for start in range(0, original_context.size(0), int(batch_nodes)):
        block = original_context[start:start + int(batch_nodes)]
        blocks.append(F.normalize(block, dim=1, eps=1e-8) @ synthetic_normalized.T)
    result = torch.cat(blocks, dim=0)
    return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)


def _subsample_edges(
    edge_index: torch.Tensor,
    limit: int,
    seed: int,
) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long().contiguous()
    if edge_index.size(1) <= int(limit):
        return edge_index
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.randperm(
        edge_index.size(1), generator=generator
    )[: int(limit)]
    return edge_index.index_select(1, selected).contiguous()


def _split_edges(
    edge_index: torch.Tensor,
    calibration_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError(
            "paper_lapt_query_calibration_fraction must be within (0, 1)"
        )
    edge_index = edge_index.detach().cpu().long().contiguous()
    if edge_index.size(1) < 2:
        raise ValueError("Paper-LAPT requires at least two query pairs per class")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(edge_index.size(1), generator=generator)
    calibration_count = min(
        edge_index.size(1) - 1,
        max(1, int(round(edge_index.size(1) * float(calibration_fraction)))),
    )
    return (
        edge_index.index_select(1, order[:calibration_count]).contiguous(),
        edge_index.index_select(1, order[calibration_count:]).contiguous(),
    )


def _transport_pair_mass(
    assignment: torch.Tensor,
    edge_index: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    result = torch.zeros(
        assignment.size(1),
        assignment.size(1),
        device=assignment.device,
        dtype=assignment.dtype,
    )
    edge_index = edge_index.to(assignment.device)
    for start in range(0, edge_index.size(1), int(batch_size)):
        batch = edge_index[:, start:start + int(batch_size)]
        left = assignment.index_select(0, batch[0])
        right = assignment.index_select(0, batch[1])
        result = result + left.T @ right + right.T @ left
    return result


def _build_target(
    assignment: torch.Tensor,
    positive_edges: torch.Tensor,
    negative_edges: torch.Tensor,
    prototype_count: int,
    budget_fraction: float,
    target_temperature: float,
    calibration_positive: torch.Tensor,
    calibration_negative: torch.Tensor,
    calibration_weight: float,
) -> torch.Tensor:
    positive_mass = _transport_pair_mass(assignment, positive_edges)
    negative_mass = _transport_pair_mass(assignment, negative_edges)
    if float(calibration_weight) > 0.0:
        positive_mass = positive_mass + float(
            calibration_weight
        ) * _transport_pair_mass(assignment, calibration_positive)
        negative_mass = negative_mass + float(
            calibration_weight
        ) * _transport_pair_mass(assignment, calibration_negative)

    raw_target = positive_mass / (
        positive_mass + negative_mass
    ).clamp_min(1e-12)
    raw_target = (raw_target + raw_target.T) / 2.0
    raw_target.fill_diagonal_(0.0)
    upper = torch.triu_indices(
        prototype_count,
        prototype_count,
        offset=1,
        device=assignment.device,
    )
    raw_upper = raw_target[upper[0], upper[1]]
    if raw_upper.numel() < 2:
        raise ValueError("Paper-LAPT requires at least two prototype pairs")
    if not 0.0 < float(budget_fraction) < 1.0:
        raise ValueError("paper_lapt_target_budget_fraction must be within (0, 1)")
    if float(target_temperature) <= 0.0:
        raise ValueError("paper_lapt_target_temperature must be positive")

    target_std = raw_upper.std(unbiased=False).clamp_min(1e-4)
    budget_count = min(
        raw_upper.numel() - 1,
        max(1, int(round(raw_upper.numel() * float(budget_fraction)))),
    )
    sorted_values = torch.sort(raw_upper, descending=True).values
    boundary = (
        sorted_values[budget_count - 1] + sorted_values[budget_count]
    ) / 2.0
    target_upper = torch.sigmoid(
        (raw_upper - boundary)
        / (target_std * float(target_temperature)).clamp_min(1e-4)
    )
    target_matrix = torch.zeros(
        prototype_count,
        prototype_count,
        device=assignment.device,
        dtype=assignment.dtype,
    )
    target_matrix[upper[0], upper[1]] = target_upper
    target_matrix[upper[1], upper[0]] = target_upper
    return target_matrix


def _transported_edge_scores(
    assignment: torch.Tensor,
    target_matrix: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    edge_index = edge_index.to(assignment.device)
    left = assignment.index_select(0, edge_index[0])
    right = assignment.index_select(0, edge_index[1])
    return ((left @ target_matrix) * right).sum(dim=1)


def _pair_audit(
    assignment: torch.Tensor,
    target_matrix: torch.Tensor,
    positive_edges: torch.Tensor,
    negative_edges: torch.Tensor,
) -> Tuple[float, float]:
    positive_scores = _transported_edge_scores(
        assignment, target_matrix, positive_edges
    )
    negative_scores = _transported_edge_scores(
        assignment, target_matrix, negative_edges
    )
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat(
        (torch.ones_like(positive_scores), torch.zeros_like(negative_scores))
    )
    gap = positive_scores.mean() - negative_scores.mean()
    return float(gap.detach().cpu()), binary_auc(scores, labels)


def parse_lambdas(values: str) -> Tuple[float, ...]:
    result = tuple(
        sorted(
            {
                float(item.strip())
                for item in str(values).split(",")
                if item.strip()
            }
        )
    )
    if not result or any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("Paper-LAPT mixture weights must lie in [0, 1]")
    if 0.0 not in result:
        raise ValueError(
            "paper_lapt_lambdas must include 0 so the unchanged HyDRO graph "
            "is an explicit validation candidate"
        )
    return result


def rank_calibrate_target(
    source_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
) -> torch.Tensor:
    """Transfer target ordering while preserving the source weight histogram."""

    source = source_adjacency.detach().float()
    target = target_adjacency.detach().to(source.device).float()
    if source.shape != target.shape or source.dim() != 2:
        raise ValueError(
            f"Source and target must be equal square matrices, got "
            f"{source.shape} and {target.shape}"
        )
    source = (source + source.T) / 2.0
    target = (target + target.T) / 2.0
    source.fill_diagonal_(0.0)
    target.fill_diagonal_(0.0)
    upper = torch.triu_indices(
        source.size(0), source.size(1), offset=1, device=source.device
    )
    source_upper = source[upper[0], upper[1]]
    target_upper = target[upper[0], upper[1]]
    source_sorted = torch.sort(source_upper).values
    target_order = torch.argsort(target_upper, stable=True)
    calibrated_upper = torch.empty_like(source_upper)
    calibrated_upper[target_order] = source_sorted
    calibrated = torch.zeros_like(source)
    calibrated[upper[0], upper[1]] = calibrated_upper
    calibrated[upper[1], upper[0]] = calibrated_upper
    return calibrated


@dataclass(frozen=True)
class PaperLAPTSummary:
    teacher_enabled: bool
    teacher_query_auc: Optional[float]
    teacher_query_gap: Optional[float]
    selected_assignment_blend: float
    structure_context_enabled: bool
    selected_structure_blend: float
    structure_context_steps: int
    calibration_auc: float
    calibration_gap: float
    audit_auc: float
    audit_gap: float
    assignment_entropy_normalized: float
    assignment_column_min: float
    assignment_column_max: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_paper_lapt_target(
    args,
    training_data: LinkTaskTrainingData,
    original_features,
    original_labels,
    synthetic_features,
    synthetic_labels,
    device,
    synthetic_adjacency=None,
) -> Tuple[torch.Tensor, PaperLAPTSummary]:
    """Build and audit a training-only LAPT target for a saved HyDRO graph."""

    original_features = _dense_cpu_features(original_features)
    synthetic_features = _dense_cpu_features(synthetic_features)
    original_labels = _cpu_labels(original_labels)
    synthetic_labels = _cpu_labels(synthetic_labels)
    device = torch.device(device)

    positive_edges = _subsample_edges(
        training_data.support_pos_edge_index,
        int(args.paper_lapt_teacher_pair_limit),
        int(args.paper_lapt_split_seed) + 22011,
    )
    negative_edges = _subsample_edges(
        training_data.support_neg_edge_index,
        int(args.paper_lapt_teacher_pair_limit),
        int(args.paper_lapt_split_seed) + 22029,
    )
    calibration_positive, audit_positive = _split_edges(
        training_data.query_pos_edge_index,
        float(args.paper_lapt_query_calibration_fraction),
        int(args.paper_lapt_split_seed) + 22031,
    )
    calibration_negative, audit_negative = _split_edges(
        training_data.query_neg_edge_index,
        float(args.paper_lapt_query_calibration_fraction),
        int(args.paper_lapt_split_seed) + 22041,
    )

    teacher_enabled = bool(args.paper_lapt_use_teacher)
    if teacher_enabled:
        teacher, teacher_metrics = fit_link_teacher(
            training_data=training_data,
            original_features=original_features,
            hidden_channels=int(args.paper_lapt_teacher_hidden),
            pair_limit=int(args.paper_lapt_teacher_pair_limit),
            training_steps=int(args.paper_lapt_teacher_steps),
            batch_size=int(args.paper_lapt_teacher_batch_size),
            learning_rate=float(args.paper_lapt_teacher_lr),
            seed=int(args.paper_lapt_split_seed),
        )
        if (
            float(teacher_metrics.query_auc)
            < float(args.paper_lapt_min_teacher_auc)
            or float(teacher_metrics.query_score_gap) <= 0.0
        ):
            raise RuntimeError(
                "Paper-LAPT teacher gate failed: "
                f"query_auc={teacher_metrics.query_auc:.6f}, "
                f"query_gap={teacher_metrics.query_score_gap:.6f}"
            )
        feature_similarity, teacher_similarity = (
            original_prototype_similarities(
                teacher=teacher,
                original_features=original_features,
                synthetic_features=synthetic_features,
                batch_nodes=int(args.paper_lapt_teacher_batch_nodes),
            )
        )
        assignment_blends = parse_blends(
            args.paper_lapt_assignment_blends
        )
        teacher_query_auc = float(teacher_metrics.query_auc)
        teacher_query_gap = float(teacher_metrics.query_score_gap)
    else:
        feature_similarity = (
            F.normalize(original_features, dim=1, eps=1e-8)
            @ F.normalize(synthetic_features, dim=1, eps=1e-8).T
        )
        teacher_similarity = torch.zeros_like(feature_similarity)
        assignment_blends = (0.0,)
        teacher_query_auc = None
        teacher_query_gap = None

    structure_enabled = bool(args.paper_lapt_use_structure_context)
    if structure_enabled:
        if synthetic_adjacency is None:
            raise ValueError(
                "Structure-aware Paper-LAPT requires synthetic_adjacency"
            )
        structure_similarity = structure_context_similarities(
            training_data=training_data,
            original_features=original_features,
            synthetic_features=synthetic_features,
            synthetic_adjacency=synthetic_adjacency,
            steps=int(args.paper_lapt_structure_steps),
            batch_nodes=int(args.paper_lapt_teacher_batch_nodes),
        )
        structure_blends = parse_blends(args.paper_lapt_structure_blends)
        if 0.0 not in structure_blends:
            raise ValueError(
                "paper_lapt_structure_blends must include 0 as the exact LAPT control"
            )
    else:
        structure_similarity = torch.zeros_like(feature_similarity)
        structure_blends = (0.0,)

    candidates = []
    for structure_blend in structure_blends:
        contextual_similarity = (
            (1.0 - float(structure_blend)) * feature_similarity
            + float(structure_blend) * structure_similarity
        )
        for blend in assignment_blends:
            assignment = link_aware_balanced_assignment(
                feature_similarity=contextual_similarity,
                teacher_similarity=teacher_similarity,
                original_labels=original_labels,
                synthetic_labels=synthetic_labels,
                teacher_blend=blend,
                temperature=float(args.paper_lapt_prototype_temperature),
                iterations=int(args.paper_lapt_sinkhorn_iter),
                class_aware=bool(args.paper_lapt_class_aware),
                device=device,
            )
            target = _build_target(
                assignment=assignment,
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                calibration_positive=calibration_positive,
                calibration_negative=calibration_negative,
                calibration_weight=float(args.paper_lapt_query_target_weight),
                prototype_count=int(synthetic_features.size(0)),
                budget_fraction=float(args.paper_lapt_target_budget_fraction),
                target_temperature=float(args.paper_lapt_target_temperature),
            )
            calibration_gap, calibration_auc = _pair_audit(
                assignment,
                target,
                calibration_positive,
                calibration_negative,
            )
            candidates.append(
                (
                    calibration_auc,
                    calibration_gap,
                    -float(structure_blend),
                    -abs(float(blend) - 0.5),
                    float(blend),
                    float(structure_blend),
                    assignment,
                    target,
                )
            )

    (
        calibration_auc,
        calibration_gap,
        _,
        _,
        selected_blend,
        selected_structure_blend,
        assignment,
        target,
    ) = max(candidates, key=lambda value: value[:4])
    audit_gap, audit_auc = _pair_audit(
        assignment,
        target,
        audit_positive,
        audit_negative,
    )
    if (
        audit_auc < float(args.paper_lapt_min_target_query_auc)
        or audit_gap < float(args.paper_lapt_min_target_query_gap)
    ):
        raise RuntimeError(
            "Paper-LAPT target gate failed: "
            f"selected_blend={selected_blend:.2f}, "
            f"audit_auc={audit_auc:.6f}, audit_gap={audit_gap:.6f}"
        )

    entropy = -(
        assignment.clamp_min(1e-12)
        * assignment.clamp_min(1e-12).log()
    ).sum(dim=1)
    entropy = entropy / torch.log(
        torch.tensor(
            float(assignment.size(1)),
            device=assignment.device,
            dtype=assignment.dtype,
        )
    )
    column_mass = assignment.sum(dim=0)
    summary = PaperLAPTSummary(
        teacher_enabled=teacher_enabled,
        teacher_query_auc=teacher_query_auc,
        teacher_query_gap=teacher_query_gap,
        selected_assignment_blend=float(selected_blend),
        structure_context_enabled=structure_enabled,
        selected_structure_blend=float(selected_structure_blend),
        structure_context_steps=(
            int(args.paper_lapt_structure_steps) if structure_enabled else 0
        ),
        calibration_auc=float(calibration_auc),
        calibration_gap=float(calibration_gap),
        audit_auc=float(audit_auc),
        audit_gap=float(audit_gap),
        assignment_entropy_normalized=float(entropy.mean().detach().cpu()),
        assignment_column_min=float(column_mass.min().detach().cpu()),
        assignment_column_max=float(column_mass.max().detach().cpu()),
    )
    return target.detach(), summary
