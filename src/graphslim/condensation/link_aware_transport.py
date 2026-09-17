"""Leakage-free link-aware prototype assignment for Step18.

The pair teacher is fitted only on support pairs.  It then scores
original-to-synthetic feature pairs, so Sinkhorn assignment is guided by
training-edge utility instead of feature cosine alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphslim.condensation.link_task_data import LinkTaskTrainingData


PAIR_STATISTIC_COUNT = 6


def _cpu_features(value) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.is_sparse:
        value = value.to_dense()
    if value.dim() != 2:
        raise ValueError(f"Expected feature matrix, got {value.shape}")
    return value.detach().cpu().float().contiguous()


def _cpu_labels(value) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().cpu()
    if value.dim() == 2:
        value = value.argmax(dim=1)
    if value.dim() != 1:
        raise ValueError(f"Expected hard labels, got {value.shape}")
    return value.long().contiguous()


def pair_statistics(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Symmetric compact pair statistics for aligned feature rows."""

    if left.shape != right.shape or left.dim() != 2:
        raise ValueError(
            f"Aligned pair features must share a 2D shape, got "
            f"{left.shape} and {right.shape}"
        )
    epsilon = torch.finfo(left.dtype).eps
    left_norm = left.square().sum(dim=1).sqrt()
    right_norm = right.square().sum(dim=1).sqrt()
    difference = left - right
    cosine = (left * right).sum(dim=1) / (
        left_norm * right_norm
    ).clamp_min(epsilon)
    minimum_norm_ratio = torch.minimum(left_norm, right_norm) / torch.maximum(
        left_norm, right_norm
    ).clamp_min(epsilon)
    nonzero_overlap = (
        ((left != 0) & (right != 0)).float().mean(dim=1)
    )
    result = torch.stack(
        (
            cosine,
            (left * right).mean(dim=1),
            difference.abs().mean(dim=1),
            difference.square().mean(dim=1).sqrt(),
            minimum_norm_ratio,
            nonzero_overlap,
        ),
        dim=1,
    )
    return torch.nan_to_num(
        result, nan=0.0, posinf=1e4, neginf=-1e4
    )


def edge_pair_statistics(
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    return pair_statistics(
        features.index_select(0, edge_index[0]),
        features.index_select(0, edge_index[1]),
    )


def _subsample(
    edge_index: torch.Tensor,
    limit: int,
    generator: torch.Generator,
) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long().contiguous()
    if edge_index.size(1) <= int(limit):
        return edge_index
    selected = torch.randperm(
        edge_index.size(1), generator=generator
    )[: int(limit)]
    return edge_index.index_select(1, selected).contiguous()


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().cpu().flatten()
    labels = labels.detach().cpu().bool().flatten()
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_scores = scores.index_select(0, order)
    sorted_labels = labels.index_select(0, order)
    _, tie_counts = torch.unique_consecutive(
        sorted_scores, return_counts=True
    )
    tie_ends = torch.cumsum(tie_counts, dim=0).float()
    tie_starts = tie_ends - tie_counts.float()
    average_rank = (tie_starts + 1.0 + tie_ends) / 2.0
    ranks = torch.repeat_interleave(average_rank, tie_counts)
    positive_rank_sum = ranks[sorted_labels].sum()
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


class LinkPairTeacher(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PAIR_STATISTIC_COUNT, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.register_buffer(
            "statistic_mean", torch.zeros(PAIR_STATISTIC_COUNT)
        )
        self.register_buffer(
            "statistic_std", torch.ones(PAIR_STATISTIC_COUNT)
        )

    @torch.no_grad()
    def set_normalization(self, statistics: torch.Tensor):
        self.statistic_mean.copy_(statistics.mean(dim=0))
        self.statistic_std.copy_(
            statistics.std(dim=0, unbiased=False).clamp_min(1e-6)
        )

    def forward_statistics(self, statistics: torch.Tensor):
        normalized = (
            statistics - self.statistic_mean
        ) / self.statistic_std
        return self.network(normalized).squeeze(1)


@dataclass(frozen=True)
class LinkTeacherMetrics:
    support_bce: float
    support_accuracy: float
    query_bce: float
    query_accuracy: float
    query_auc: float
    query_score_gap: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def fit_link_teacher(
    training_data: LinkTaskTrainingData,
    original_features: torch.Tensor,
    hidden_channels: int,
    pair_limit: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Tuple[LinkPairTeacher, LinkTeacherMetrics]:
    """Fit on support pairs and audit on the held-out training query."""

    if min(
        hidden_channels,
        pair_limit,
        training_steps,
        batch_size,
    ) <= 0 or learning_rate <= 0.0:
        raise ValueError("Link teacher hyperparameters must be positive")
    features = _cpu_features(original_features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 18001)
    support_positive = _subsample(
        training_data.support_pos_edge_index, pair_limit, generator
    )
    support_negative = _subsample(
        training_data.support_neg_edge_index, pair_limit, generator
    )
    support_edges = torch.cat(
        (support_positive, support_negative), dim=1
    )
    support_labels = torch.cat(
        (
            torch.ones(support_positive.size(1)),
            torch.zeros(support_negative.size(1)),
        )
    )
    statistics = edge_pair_statistics(features, support_edges)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 18013)
        teacher = LinkPairTeacher(int(hidden_channels))
        teacher.set_normalization(statistics)
        optimizer = torch.optim.Adam(
            teacher.parameters(), lr=float(learning_rate)
        )
        for _ in range(int(training_steps)):
            selected = torch.randint(
                statistics.size(0),
                (min(int(batch_size), statistics.size(0)),),
                generator=generator,
            )
            logits = teacher.forward_statistics(
                statistics.index_select(0, selected)
            )
            target = support_labels.index_select(0, selected)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    query_positive = _subsample(
        training_data.query_pos_edge_index, pair_limit, generator
    )
    query_negative = _subsample(
        training_data.query_neg_edge_index, pair_limit, generator
    )
    query_edges = torch.cat(
        (query_positive, query_negative), dim=1
    )
    query_labels = torch.cat(
        (
            torch.ones(query_positive.size(1)),
            torch.zeros(query_negative.size(1)),
        )
    )
    with torch.no_grad():
        support_logits = teacher.forward_statistics(statistics)
        query_logits = teacher.forward_statistics(
            edge_pair_statistics(features, query_edges)
        )
        positive_mean = query_logits[: query_positive.size(1)].mean()
        negative_mean = query_logits[query_positive.size(1):].mean()
        metrics = LinkTeacherMetrics(
            support_bce=float(
                F.binary_cross_entropy_with_logits(
                    support_logits, support_labels
                )
            ),
            support_accuracy=float(
                (
                    (support_logits >= 0)
                    == support_labels.bool()
                ).float().mean()
            ),
            query_bce=float(
                F.binary_cross_entropy_with_logits(
                    query_logits, query_labels
                )
            ),
            query_accuracy=float(
                (
                    (query_logits >= 0) == query_labels.bool()
                ).float().mean()
            ),
            query_auc=binary_auc(query_logits, query_labels),
            query_score_gap=float(positive_mean - negative_mean),
        )
    return teacher, metrics


@torch.no_grad()
def original_prototype_similarities(
    teacher: LinkPairTeacher,
    original_features: torch.Tensor,
    synthetic_features: torch.Tensor,
    batch_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return feature and teacher link-profile similarity for each N x N'.

    Direct edge probability is not an assignment probability: two nodes
    belong to the same prototype when they have similar link preferences,
    not necessarily when they are likely to connect to each other.  We
    therefore compare each original node's teacher-predicted affinity
    profile over all prototypes with every prototype's own affinity profile.
    """

    original = _cpu_features(original_features)
    synthetic = _cpu_features(synthetic_features)
    if original.size(1) != synthetic.size(1):
        raise ValueError("Original and synthetic feature widths differ")
    synthetic_normalized = F.normalize(
        synthetic, dim=1, eps=1e-8
    )
    feature_blocks = []
    affinity_blocks = []
    for start in range(0, original.size(0), int(batch_nodes)):
        block = original[start:start + int(batch_nodes)]
        feature_blocks.append(
            F.normalize(block, dim=1, eps=1e-8)
            @ synthetic_normalized.T
        )
        left = block.repeat_interleave(synthetic.size(0), dim=0)
        right = synthetic.repeat(block.size(0), 1)
        utility = torch.sigmoid(
            teacher.forward_statistics(pair_statistics(left, right))
        )
        affinity_blocks.append(
            utility.reshape(block.size(0), synthetic.size(0))
        )
    original_affinity = torch.cat(affinity_blocks, dim=0)
    prototype_left = synthetic.repeat_interleave(
        synthetic.size(0), dim=0
    )
    prototype_right = synthetic.repeat(synthetic.size(0), 1)
    prototype_affinity = torch.sigmoid(
        teacher.forward_statistics(
            pair_statistics(prototype_left, prototype_right)
        )
    ).reshape(synthetic.size(0), synthetic.size(0))

    affinity_mean = original_affinity.mean(dim=0, keepdim=True)
    affinity_std = original_affinity.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-5)
    original_profile = (
        original_affinity - affinity_mean
    ) / affinity_std
    prototype_profile = (
        prototype_affinity - affinity_mean
    ) / affinity_std
    profile_similarity = (
        F.normalize(original_profile, dim=1, eps=1e-8)
        @ F.normalize(prototype_profile, dim=1, eps=1e-8).T
    )
    return torch.cat(feature_blocks, dim=0), profile_similarity


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    logits = logits - logits.max(dim=1, keepdim=True).values
    assignment = logits.exp().clamp_min(1e-20)
    column_mass = float(logits.size(0)) / float(logits.size(1))
    for _ in range(int(iterations)):
        assignment = assignment / assignment.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-20)
        assignment = assignment / assignment.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-20)
        assignment = assignment * column_mass
    return assignment / assignment.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-20)


def link_aware_balanced_assignment(
    feature_similarity: torch.Tensor,
    teacher_similarity: torch.Tensor,
    original_labels: torch.Tensor,
    synthetic_labels: torch.Tensor,
    teacher_blend: float,
    temperature: float,
    iterations: int,
    class_aware: bool,
    device,
) -> torch.Tensor:
    if not 0.0 <= teacher_blend <= 1.0:
        raise ValueError("teacher_blend must be within [0, 1]")
    if temperature <= 0.0 or iterations <= 0:
        raise ValueError("Sinkhorn temperature/iterations must be positive")
    similarity = (
        (1.0 - teacher_blend) * feature_similarity
        + teacher_blend * teacher_similarity
    )
    original_labels = _cpu_labels(original_labels)
    synthetic_labels = _cpu_labels(synthetic_labels)
    if not class_aware:
        assignment = _sinkhorn_from_logits(
            similarity / float(temperature), iterations
        )
        return assignment.to(device)

    assignment = torch.zeros_like(similarity)
    for class_value in torch.unique(
        original_labels, sorted=True
    ).tolist():
        real_index = torch.where(original_labels == class_value)[0]
        synthetic_index = torch.where(
            synthetic_labels == class_value
        )[0]
        if synthetic_index.numel() == 0:
            raise ValueError(
                f"No synthetic prototype for class {class_value}"
            )
        block = similarity[
            real_index[:, None], synthetic_index[None, :]
        ]
        block = _sinkhorn_from_logits(
            block / float(temperature), iterations
        )
        assignment[
            real_index[:, None], synthetic_index[None, :]
        ] = block
    return assignment.to(device)


def parse_blends(values: str) -> Tuple[float, ...]:
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
        raise ValueError("LAPT assignment blends must lie in [0, 1]")
    return result
