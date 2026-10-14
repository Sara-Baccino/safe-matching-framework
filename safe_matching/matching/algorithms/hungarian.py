"""Assegnazione ottima 1:1 (algoritmo ungherese)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from safe_matching.matching.algorithms.base import MatchingAlgorithm, RawMatch
from safe_matching.matching.constraints.caliper import PENALTY_THRESHOLD

__all__ = ["Hungarian"]


class Hungarian(MatchingAlgorithm):
    """Minimizza il costo totale. Le coppie vietate (caliper) ricevono un costo molto alto
    ma finito, calcolato in modo da massimizzare PRIMA il numero di coppie ammissibili e poi
    minimizzare la distanza; le assegnazioni vietate scelte comunque dal solver sono scartate."""

    name = "hungarian"

    def match(self, cost: np.ndarray, aux: dict[str, Any]) -> RawMatch:
        n_q, n_p = cost.shape
        feasible = cost < PENALTY_THRESHOLD
        if not feasible.any():
            return RawMatch.empty()
        if feasible.all():
            C = cost
        else:
            big = (float(cost[feasible].max()) + 1.0) * (min(n_q, n_p) + 1)
            C = np.where(feasible, cost, big)
        rows, cols = linear_sum_assignment(C)
        keep = feasible[rows, cols]
        rows, cols = rows[keep], cols[keep]
        return RawMatch(
            rows.astype(int), cols.astype(int), np.arange(rows.size, dtype=int),
            cost[rows, cols].astype(float), meta={"algorithm": "hungarian"},
        )
