"""Load and validate saved condensed graphs for fixed-protocol evaluation."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch

def _dense(value):
    if hasattr(value, "to_dense"):
        try:
            value = value.to_dense()
        except TypeError:
            pass
    if hasattr(value, "toarray"):
        value = value.toarray()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.is_sparse:
        value = value.to_dense()
    return value.detach().cpu().contiguous()

def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def parse_integer_list(values: str) -> Tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(values).split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("Seed list must contain unique integers")
    return result

def parse_source_paths(values: str) -> Tuple[str, ...]:
    result = tuple(item.strip().strip('"') for item in str(values).split(",") if item.strip())
    if not result:
        raise ValueError("Source paths cannot be empty")
    return tuple(str(Path(item).expanduser().resolve()) for item in result)

@dataclass(frozen=True)
class SeedGraph:
    condensation_seed: int
    source_root: str
    adjacency: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor

def load_seed_graph(root: str, method: str, dataset: str, rate: float, seed: int) -> SeedGraph:
    directory = Path(root) / "reduced_graph" / method
    stem = f"{dataset}_{float(rate)}_{int(seed)}.pt"
    paths = {name: directory / f"{prefix}_{stem}" for name, prefix in (
        ("adjacency", "adj"), ("features", "feat"), ("labels", "label")
    )}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing complete condensed graph: " + "; ".join(missing))
    adjacency = _dense(_load(paths["adjacency"])).float()
    adjacency = (adjacency + adjacency.T) / 2.0
    adjacency.fill_diagonal_(0.0)
    features = _dense(_load(paths["features"])).float()
    labels = _dense(_load(paths["labels"]))
    if labels.dim() == 2:
        labels = labels.argmax(dim=1)
    labels = labels.long().reshape(-1)
    if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
        raise ValueError("Saved adjacency must be square")
    if features.size(0) != adjacency.size(0) or labels.numel() != adjacency.size(0):
        raise ValueError("Saved adjacency, features and labels have incompatible sizes")
    return SeedGraph(int(seed), str(Path(root).resolve()), adjacency, features, labels)

def _label_counts(labels) -> Dict[int, int]:
    return {int(value): int((labels == value).sum()) for value in torch.unique(labels).tolist()}

def validate_seed_graphs(graphs: Sequence[SeedGraph]):
    reference_shape = graphs[0].features.shape
    reference_counts = _label_counts(graphs[0].labels)
    for graph in graphs[1:]:
        if graph.features.shape != reference_shape or _label_counts(graph.labels) != reference_counts:
            raise ValueError("Condensation seeds have incompatible feature shapes or class budgets")
    return reference_counts

def tensor_sha256(value: torch.Tensor) -> str:
    value = _dense(value)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
