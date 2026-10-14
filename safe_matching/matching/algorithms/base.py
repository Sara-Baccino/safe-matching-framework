"""Contratto comune degli algoritmi di matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

__all__ = ["RawMatch", "MatchingAlgorithm"]


@dataclass
class RawMatch:
    """Risultato di un algoritmo su UNA matrice di costo (indici relativi alla matrice).

    output_type == "pairs": ``q_idx[i]`` e' abbinato a ``p_idx[i]``; le righe con lo stesso
    ``pair_id`` appartengono alla stessa query (1:K).
    output_type == "selection": sottoinsieme del pool con pesi ``weights`` (massa di trasporto);
    non esiste una corrispondenza uno-a-uno e ``pair_id`` vale -1.
    """

    q_idx: np.ndarray
    p_idx: np.ndarray
    pair_id: np.ndarray
    dist: np.ndarray
    output_type: str = "pairs"
    weights: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def empty() -> "RawMatch":
        z = np.array([], dtype=int)
        return RawMatch(z, z.copy(), z.copy(), np.array([], dtype=float))


class MatchingAlgorithm(ABC):
    name: str = ""

    def __init__(self, params: BaseModel, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng

    @abstractmethod
    def match(self, cost: np.ndarray, aux: dict[str, Any]) -> RawMatch:
        """``cost``: (n_query, n_pool); le celle >= PENALTY_THRESHOLD sono coppie vietate.
        ``aux`` puo' contenere ``sort_q``/``sort_p`` (valori 1-D) e ``target_size``."""
