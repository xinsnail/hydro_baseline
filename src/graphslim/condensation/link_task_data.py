"""Leakage-safe edge data preparation for link-task-aware condensation.

This module is intentionally independent from the downstream evaluator.  It
creates a research training split once, using only the original graph, and
returns only the support/query portion of the *training* edges to HyDRO.  The
held-out validation and test positives are counted for auditing but are never
returned to the condensation training loop.
"""

from dataclasses import dataclass
import random
from typing import Dict, Tuple

import torch
from torch_sparse import SparseTensor


@dataclass(frozen=True)
class LinkTaskTrainingData:
    """Training-only edge tensors used by the future TAL objective."""

    message_edge_index: torch.Tensor
    support_pos_edge_index: torch.Tensor
    support_neg_edge_index: torch.Tensor
    query_pos_edge_index: torch.Tensor
    query_neg_edge_index: torch.Tensor
    validation_pos_edge_index: torch.Tensor
    validation_neg_edge_index: torch.Tensor
    test_pos_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor
    query_pos_closure: torch.Tensor
    query_neg_closure: torch.Tensor
    summary: Dict[str, int]

    def to(self, device):
        """Return a device-local copy while preserving audit metadata."""

        return LinkTaskTrainingData(
            message_edge_index=self.message_edge_index.to(device),
            support_pos_edge_index=self.support_pos_edge_index.to(device),
            support_neg_edge_index=self.support_neg_edge_index.to(device),
            query_pos_edge_index=self.query_pos_edge_index.to(device),
            query_neg_edge_index=self.query_neg_edge_index.to(device),
            validation_pos_edge_index=self.validation_pos_edge_index.to(device),
            validation_neg_edge_index=self.validation_neg_edge_index.to(device),
            test_pos_edge_index=self.test_pos_edge_index.to(device),
            test_neg_edge_index=self.test_neg_edge_index.to(device),
            query_pos_closure=self.query_pos_closure.to(device),
            query_neg_closure=self.query_neg_closure.to(device),
            summary=dict(self.summary),
        )


def _edge_rows(adjacency) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Extract row/column indices and node count from supported adjacencies."""

    if isinstance(adjacency, SparseTensor):
        row, col, _ = adjacency.coo()
        return row.cpu().long(), col.cpu().long(), int(adjacency.size(0))

    if torch.is_tensor(adjacency):
        node_count = int(adjacency.size(0))
        if adjacency.is_sparse:
            indices = adjacency.coalesce().indices().cpu().long()
        else:
            indices = adjacency.detach().cpu().nonzero(as_tuple=False).T.long()
        return indices[0], indices[1], node_count

    if hasattr(adjacency, "tocoo"):
        coo = adjacency.tocoo()
        row = torch.as_tensor(coo.row, dtype=torch.long)
        col = torch.as_tensor(coo.col, dtype=torch.long)
        return row, col, int(adjacency.shape[0])

    raise TypeError(f"Unsupported adjacency type: {type(adjacency)!r}")


def _unique_undirected_edges(adjacency) -> Tuple[torch.Tensor, int]:
    """Return canonical ``u < v`` edges with duplicates and loops removed."""

    row, col, node_count = _edge_rows(adjacency)
    low = torch.minimum(row, col)
    high = torch.maximum(row, col)
    keep = low < high
    keys = torch.unique(low[keep] * node_count + high[keep], sorted=True)
    edges = torch.stack((keys // node_count, keys % node_count), dim=0)
    return edges.long(), node_count


def _take_columns(edge_index: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return edge_index.index_select(1, indices.long()).contiguous()


def _sample_negative_edges(
    all_positive_edges: torch.Tensor,
    node_count: int,
    sample_count: int,
    hard_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    """Sample non-edges, prioritising open wedges as hard negatives.

    An open wedge ``u-w-v`` is a two-hop pair ``(u, v)`` that is not an edge.
    Sampling such pairs is linear in the stored graph size and avoids creating
    an ``N x N`` dense matrix for the original graph.
    """

    if sample_count <= 0:
        return torch.empty((2, 0), dtype=torch.long), 0

    rng = random.Random(int(seed))
    positive_keys = {
        int(u) * node_count + int(v)
        for u, v in all_positive_edges.T.tolist()
    }
    neighbours = [[] for _ in range(node_count)]
    for u, v in all_positive_edges.T.tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)

    wedge_centres = [node for node, values in enumerate(neighbours) if len(values) >= 2]
    hard_target = min(sample_count, int(round(sample_count * hard_ratio)))
    negative_keys = set()
    hard_keys = set()

    attempts = 0
    max_hard_attempts = max(2000, hard_target * 80)
    while len(hard_keys) < hard_target and wedge_centres and attempts < max_hard_attempts:
        centre = rng.choice(wedge_centres)
        u, v = rng.sample(neighbours[centre], 2)
        if u > v:
            u, v = v, u
        key = u * node_count + v
        attempts += 1
        if u != v and key not in positive_keys and key not in negative_keys:
            hard_keys.add(key)
            negative_keys.add(key)

    attempts = 0
    max_random_attempts = max(5000, sample_count * 100)
    while len(negative_keys) < sample_count and attempts < max_random_attempts:
        u = rng.randrange(node_count)
        v = rng.randrange(node_count)
        attempts += 1
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = u * node_count + v
        if key not in positive_keys and key not in negative_keys:
            negative_keys.add(key)

    if len(negative_keys) < sample_count:
        raise RuntimeError(
            f"Could sample only {len(negative_keys)} of {sample_count} requested "
            "negative edges; the graph may be too dense."
        )

    negative_list = list(negative_keys)
    rng.shuffle(negative_list)
    negative_list = negative_list[:sample_count]
    keys = torch.as_tensor(negative_list, dtype=torch.long)
    edge_index = torch.stack((keys // node_count, keys % node_count), dim=0)
    actual_hard = sum(key in hard_keys for key in negative_list)
    return edge_index.contiguous(), int(actual_hard)


def _normalized_common_neighbor_scores(
    support_edges: torch.Tensor,
    query_edges: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """Measure training-only structural closure for a set of node pairs.

    The score is the number of common support neighbours divided by the
    geometric mean of the two support degrees.  It lies in ``[0, 1]`` and is
    comparable across original graphs and much smaller condensed graphs.
    Query, validation, and test edges are never inserted into the neighbour
    sets, so this statistic cannot expose a held-out positive to condensation.
    """

    neighbours = [set() for _ in range(node_count)]
    for u, v in support_edges.T.tolist():
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))

    scores = []
    for u, v in query_edges.T.tolist():
        u = int(u)
        v = int(v)
        degree_product = len(neighbours[u]) * len(neighbours[v])
        if degree_product == 0:
            scores.append(0.0)
            continue
        common = len(neighbours[u].intersection(neighbours[v]))
        scores.append(common / (degree_product ** 0.5))
    return torch.tensor(scores, dtype=torch.float32)


def build_link_task_training_data(
    adjacency,
    seed: int,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.20,
    query_ratio: float = 0.10,
    negative_ratio: float = 1.0,
    hard_negative_ratio: float = 0.50,
) -> LinkTaskTrainingData:
    """Build deterministic support/query LP data without validation/test leakage."""

    for name, value in (
        ("validation_ratio", validation_ratio),
        ("test_ratio", test_ratio),
        ("query_ratio", query_ratio),
        ("hard_negative_ratio", hard_negative_ratio),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")
    if negative_ratio < 0.0:
        raise ValueError("negative_ratio must be non-negative")

    all_edges, node_count = _unique_undirected_edges(adjacency)
    edge_count = int(all_edges.size(1))
    if edge_count < 4:
        raise ValueError("At least four undirected edges are required")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(edge_count, generator=generator)

    test_count = int(round(edge_count * test_ratio))
    validation_count = int(round(edge_count * validation_ratio))
    train_count = edge_count - test_count - validation_count
    if train_count < 2:
        raise ValueError("The requested split leaves fewer than two training edges")

    train_indices = permutation[:train_count]
    validation_indices = permutation[train_count:train_count + validation_count]
    test_indices = permutation[train_count + validation_count:]

    # These assertions are the leakage guard. Only train_indices are used below.
    assert set(train_indices.tolist()).isdisjoint(validation_indices.tolist())
    assert set(train_indices.tolist()).isdisjoint(test_indices.tolist())
    assert set(validation_indices.tolist()).isdisjoint(test_indices.tolist())

    train_edges = _take_columns(all_edges, train_indices)
    query_count = int(round(train_count * query_ratio))
    query_count = min(max(query_count, 1), train_count - 1)
    train_permutation = torch.randperm(train_count, generator=generator)
    query_edges = _take_columns(train_edges, train_permutation[:query_count])
    support_edges = _take_columns(train_edges, train_permutation[query_count:])

    support_negative_count = int(round(support_edges.size(1) * negative_ratio))
    query_negative_count = int(round(query_edges.size(1) * negative_ratio))
    validation_negative_count = int(round(validation_count * negative_ratio))
    test_negative_count = int(round(test_count * negative_ratio))
    negatives, hard_count = _sample_negative_edges(
        all_positive_edges=all_edges,
        node_count=node_count,
        sample_count=(
            support_negative_count
            + query_negative_count
            + validation_negative_count
            + test_negative_count
        ),
        hard_ratio=hard_negative_ratio,
        seed=seed + 104729,
    )
    negative_permutation = torch.randperm(negatives.size(1), generator=generator)
    support_negatives = _take_columns(
        negatives, negative_permutation[:support_negative_count]
    )
    query_negatives = _take_columns(
        negatives,
        negative_permutation[
            support_negative_count:support_negative_count + query_negative_count
        ],
    )
    validation_negative_start = support_negative_count + query_negative_count
    validation_negatives = _take_columns(
        negatives,
        negative_permutation[
            validation_negative_start:
            validation_negative_start + validation_negative_count
        ],
    )
    test_negative_start = validation_negative_start + validation_negative_count
    test_negatives = _take_columns(
        negatives,
        negative_permutation[
            test_negative_start:test_negative_start + test_negative_count
        ],
    )
    validation_edges = _take_columns(all_edges, validation_indices)
    test_edges = _take_columns(all_edges, test_indices)

    query_positive_closure = _normalized_common_neighbor_scores(
        support_edges=support_edges,
        query_edges=query_edges,
        node_count=node_count,
    )
    query_negative_closure = _normalized_common_neighbor_scores(
        support_edges=support_edges,
        query_edges=query_negatives,
        node_count=node_count,
    )

    message_edge_index = torch.cat((support_edges, support_edges.flip(0)), dim=1)
    summary = {
        "nodes": node_count,
        "all_positive": edge_count,
        "train_positive": train_count,
        "validation_positive_held_out": validation_count,
        "test_positive_held_out": test_count,
        "support_positive": int(support_edges.size(1)),
        "query_positive": int(query_edges.size(1)),
        "support_negative": int(support_negatives.size(1)),
        "query_negative": int(query_negatives.size(1)),
        "validation_negative_held_out": int(validation_negatives.size(1)),
        "test_negative_held_out": int(test_negatives.size(1)),
        "hard_negative_total": hard_count,
    }

    return LinkTaskTrainingData(
        message_edge_index=message_edge_index.contiguous(),
        support_pos_edge_index=support_edges,
        support_neg_edge_index=support_negatives,
        query_pos_edge_index=query_edges,
        query_neg_edge_index=query_negatives,
        validation_pos_edge_index=validation_edges,
        validation_neg_edge_index=validation_negatives,
        test_pos_edge_index=test_edges,
        test_neg_edge_index=test_negatives,
        query_pos_closure=query_positive_closure,
        query_neg_closure=query_negative_closure,
        summary=summary,
    )


def resample_training_query_data(
    training_data: LinkTaskTrainingData,
    seed: int,
    query_ratio: float = 0.10,
) -> LinkTaskTrainingData:
    """Resample support/query pairs while preserving the outer edge split.

    The union of support and query positives is the original 70% training
    partition. Validation and test tensors are copied unchanged. This permits
    multiple proxy views without ever promoting an outer held-out edge into
    model selection.
    """

    if not 0.0 < query_ratio < 1.0:
        raise ValueError("query_ratio must lie strictly within (0, 1)")

    train_positive = torch.cat(
        (training_data.support_pos_edge_index,
         training_data.query_pos_edge_index),
        dim=1,
    ).cpu()
    train_negative = torch.cat(
        (training_data.support_neg_edge_index,
         training_data.query_neg_edge_index),
        dim=1,
    ).cpu()
    positive_count = int(train_positive.size(1))
    negative_count = int(train_negative.size(1))
    if positive_count < 2 or negative_count < 2:
        raise ValueError("At least two training positive/negative pairs are required")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    positive_permutation = torch.randperm(positive_count, generator=generator)
    negative_permutation = torch.randperm(negative_count, generator=generator)
    query_positive_count = min(
        max(int(round(positive_count * query_ratio)), 1), positive_count - 1
    )
    query_negative_count = min(
        max(int(round(negative_count * query_ratio)), 1), negative_count - 1
    )

    query_positive = _take_columns(
        train_positive, positive_permutation[:query_positive_count]
    )
    support_positive = _take_columns(
        train_positive, positive_permutation[query_positive_count:]
    )
    query_negative = _take_columns(
        train_negative, negative_permutation[:query_negative_count]
    )
    support_negative = _take_columns(
        train_negative, negative_permutation[query_negative_count:]
    )
    node_count = int(training_data.summary["nodes"])
    query_positive_closure = _normalized_common_neighbor_scores(
        support_edges=support_positive,
        query_edges=query_positive,
        node_count=node_count,
    )
    query_negative_closure = _normalized_common_neighbor_scores(
        support_edges=support_positive,
        query_edges=query_negative,
        node_count=node_count,
    )
    message_edge_index = torch.cat(
        (support_positive, support_positive.flip(0)), dim=1
    )
    summary = dict(training_data.summary)
    summary.update({
        "support_positive": int(support_positive.size(1)),
        "query_positive": int(query_positive.size(1)),
        "support_negative": int(support_negative.size(1)),
        "query_negative": int(query_negative.size(1)),
        "proxy_query_seed": int(seed),
    })

    return LinkTaskTrainingData(
        message_edge_index=message_edge_index.contiguous(),
        support_pos_edge_index=support_positive.contiguous(),
        support_neg_edge_index=support_negative.contiguous(),
        query_pos_edge_index=query_positive.contiguous(),
        query_neg_edge_index=query_negative.contiguous(),
        validation_pos_edge_index=training_data.validation_pos_edge_index,
        validation_neg_edge_index=training_data.validation_neg_edge_index,
        test_pos_edge_index=training_data.test_pos_edge_index,
        test_neg_edge_index=training_data.test_neg_edge_index,
        query_pos_closure=query_positive_closure,
        query_neg_closure=query_negative_closure,
        summary=summary,
    )


def evaluator_split_from_training_data(
    training_data: LinkTaskTrainingData,
    features: torch.Tensor,
    device,
) -> Dict[str, torch.Tensor]:
    """Build the downstream evaluator split from the exact BOLT split.

    The message-passing graph contains every positive edge assigned to the
    70% training partition.  BOLT still keeps its internal support/query
    partition for condensation, while the downstream evaluator trains on
    their union.  Validation and test pairs remain held out from both.
    """

    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    if features.is_sparse:
        features = features.to_dense()
    features = features.detach().float().to(device)

    train_positive = torch.cat(
        (
            training_data.support_pos_edge_index,
            training_data.query_pos_edge_index,
        ),
        dim=1,
    )
    train_negative = torch.cat(
        (
            training_data.support_neg_edge_index,
            training_data.query_neg_edge_index,
        ),
        dim=1,
    )

    def labelled_pairs(positive, negative):
        edge_index = torch.cat((positive, negative), dim=1).long().to(device)
        labels = torch.cat(
            (
                torch.ones(positive.size(1), dtype=torch.float32),
                torch.zeros(negative.size(1), dtype=torch.float32),
            )
        ).to(device)
        return edge_index, labels

    train_edge_index, train_labels = labelled_pairs(
        train_positive, train_negative
    )
    validation_edge_index, validation_labels = labelled_pairs(
        training_data.validation_pos_edge_index,
        training_data.validation_neg_edge_index,
    )
    test_edge_index, test_labels = labelled_pairs(
        training_data.test_pos_edge_index,
        training_data.test_neg_edge_index,
    )
    message_edge_index = torch.cat(
        (train_positive, train_positive.flip(0)), dim=1
    ).long().to(device)

    return {
        "x": features,
        "message_edge_index": message_edge_index,
        "train_edge_label_index": train_edge_index,
        "train_edge_label": train_labels,
        "val_edge_label_index": validation_edge_index,
        "val_edge_label": validation_labels,
        "test_edge_label_index": test_edge_index,
        "test_edge_label": test_labels,
    }


def audit_against_fixed_evaluator_split(
    training_data: LinkTaskTrainingData,
    evaluator_split: Dict[str, torch.Tensor],
    node_count: int,
) -> Dict[str, int | bool]:
    """Prove that task-aware condensation uses the evaluator's train edges.

    Two independently implemented 70/10/20 splits can disagree even when they
    use the same integer seed. This audit compares canonical undirected edge
    identifiers at runtime. LMRC aborts before optimization unless its
    support/query union exactly equals the positive training set and has zero
    overlap with the fixed evaluator's validation and test positives.
    """

    def canonical_keys(edge_index: torch.Tensor) -> set[int]:
        edge_index = edge_index.detach().cpu().long()
        low = torch.minimum(edge_index[0], edge_index[1])
        high = torch.maximum(edge_index[0], edge_index[1])
        keep = low < high
        return set((low[keep] * int(node_count) + high[keep]).tolist())

    def positive_keys(prefix: str) -> set[int]:
        labels = evaluator_split[f"{prefix}_edge_label"].detach().cpu()
        indices = evaluator_split[
            f"{prefix}_edge_label_index"
        ].detach().cpu()
        return canonical_keys(indices[:, labels > 0.5])

    condensation_train = canonical_keys(
        torch.cat(
            (
                training_data.support_pos_edge_index,
                training_data.query_pos_edge_index,
            ),
            dim=1,
        )
    )
    evaluator_train = positive_keys("train")
    evaluator_validation = positive_keys("val")
    evaluator_test = positive_keys("test")

    training_difference = condensation_train.symmetric_difference(
        evaluator_train
    )
    validation_overlap = condensation_train.intersection(
        evaluator_validation
    )
    test_overlap = condensation_train.intersection(evaluator_test)
    passed = (
        not training_difference
        and not validation_overlap
        and not test_overlap
    )
    return {
        "condensation_train_positive": len(condensation_train),
        "evaluator_train_positive": len(evaluator_train),
        "train_positive_symmetric_difference": len(training_difference),
        "validation_positive_overlap": len(validation_overlap),
        "test_positive_overlap": len(test_overlap),
        "fixed_split_match_pass": passed,
    }
