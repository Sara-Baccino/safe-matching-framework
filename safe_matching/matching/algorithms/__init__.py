from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from safe_matching.exceptions import UnsupportedOptionError
from safe_matching.matching.algorithms.base import MatchingAlgorithm, RawMatch
from safe_matching.matching.algorithms.greedy_nn import GreedyNN
from safe_matching.matching.algorithms.hungarian import Hungarian
from safe_matching.matching.algorithms.partial_ot import PartialOT
from safe_matching.matching.algorithms.sorted_dp import SortedDP

__all__ = [
    "MatchingAlgorithm", "RawMatch", "GreedyNN", "Hungarian", "PartialOT", "SortedDP",
    "ALGORITHMS", "register_algorithm", "build_algorithm",
]

ALGORITHMS: dict[str, type[MatchingAlgorithm]] = {
    "greedy_nn": GreedyNN,
    "hungarian": Hungarian,
    "sorted": SortedDP,
    "partial_ot": PartialOT,
}


def register_algorithm(name: str):
    """Decoratore per aggiungere un algoritmo custom (anche da codice esterno)."""

    def deco(cls: type[MatchingAlgorithm]):
        ALGORITHMS[name] = cls
        cls.name = name
        return cls

    return deco


def build_algorithm(name: str, params: BaseModel, rng: np.random.Generator) -> MatchingAlgorithm:
    try:
        return ALGORITHMS[name](params, rng)
    except KeyError as exc:
        raise UnsupportedOptionError(f"Algoritmo '{name}' non registrato. Disponibili: {sorted(ALGORITHMS)}") from exc
