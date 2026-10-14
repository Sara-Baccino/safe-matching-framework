"""Optimal transport parziale: selezione di un sottoinsieme del pool.

Dato un costo (n_query x n_pool), trasporta (quasi) tutta la massa uniforme dei query verso
un pool in cui ogni unita' puo' ricevere al piu' 1/target_size; le unita' del pool con piu'
massa ricevuta sono selezionate. Non produce coppie: il risultato e' una *selezione di
popolazione* pesata dalla massa trasportata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from safe_matching.exceptions import MatchingError
from safe_matching.matching.algorithms.base import MatchingAlgorithm, RawMatch
from safe_matching.matching.constraints.caliper import PENALTY_THRESHOLD

__all__ = ["PartialOT"]


class PartialOT(MatchingAlgorithm):
    name = "partial_ot"

    def match(self, cost: np.ndarray, aux: dict[str, Any]) -> RawMatch:
        import ot

        n_q, n_p = cost.shape
        target = int(aux["target_size"])
        if target > n_p:
            raise MatchingError(f"partial_ot: target_size ({target}) maggiore del pool ({n_p}).")
        feasible = cost < PENALTY_THRESHOLD
        if not feasible.any():
            return RawMatch.empty()
        c = cost
        if not feasible.all():
            # celle vietate -> costo alto (non infinito) cosi' il solver resta ben condizionato
            c = np.where(feasible, cost, 10.0 * float(cost[feasible].max()))
        max_cost = float(c.max())
        M = c / max_cost if max_cost > 0 else c

        a = np.full(n_q, 1.0 / n_q)
        b = np.full(n_p, 1.0 / target)
        m = max(0.0, min(1.0, a.sum(), b.sum()) - 1e-10)
        plan = ot.partial.partial_wasserstein(
            a=a, b=b, M=M, m=m, nb_dummies=self.params.nb_dummies, numItermax=self.params.max_iterations
        )
        mass = plan.sum(axis=0)
        min_d = np.where(feasible, cost, np.inf).min(axis=0)
        selected = np.lexsort((min_d, -mass))[:target]
        selected = selected[np.isfinite(min_d[selected])]
        return RawMatch(
            q_idx=np.arange(n_q, dtype=int),
            p_idx=selected.astype(int),
            pair_id=np.full(selected.size, -1, dtype=int),
            dist=min_d[selected].astype(float),
            output_type="selection",
            weights=mass[selected].astype(float),
            meta={"algorithm": "partial_ot", "target_size": target, "transported_mass": m},
        )
