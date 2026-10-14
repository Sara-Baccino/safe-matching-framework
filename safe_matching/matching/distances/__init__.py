from __future__ import annotations

from typing import Any

from safe_matching.exceptions import UnsupportedOptionError
from safe_matching.matching.distances.base import DistanceMetric
from safe_matching.matching.distances.hybrid import GowerDistance, HybridDistance
from safe_matching.matching.distances.mahalanobis import MahalanobisDistance
from safe_matching.matching.distances.minkowski import EuclideanDistance, ManhattanDistance

__all__ = [
    "DistanceMetric", "EuclideanDistance", "ManhattanDistance", "MahalanobisDistance",
    "HybridDistance", "GowerDistance", "DISTANCES", "build_distance",
]

DISTANCES: dict[str, type[DistanceMetric]] = {
    "euclidean": EuclideanDistance,
    "manhattan": ManhattanDistance,
    "mahalanobis": MahalanobisDistance,
    "hybrid": HybridDistance,
    "gower": GowerDistance,
}


def build_distance(name: str, params: dict[str, Any] | None = None) -> DistanceMetric:
    try:
        cls = DISTANCES[name]
    except KeyError as exc:
        raise UnsupportedOptionError(f"Metrica '{name}' non supportata. Valide: {sorted(DISTANCES)}") from exc
    return cls(**(params or {}))
