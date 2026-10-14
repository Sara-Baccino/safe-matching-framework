"""Nearest neighbour greedy 1:K."""

from __future__ import annotations

from typing import Any

import numpy as np

from safe_matching.matching.algorithms.base import MatchingAlgorithm, RawMatch
from safe_matching.matching.constraints.caliper import PENALTY_THRESHOLD

__all__ = ["GreedyNN"]


class GreedyNN(MatchingAlgorithm):
    name = "greedy_nn"

    def match(self, cost: np.ndarray, aux: dict[str, Any]) -> RawMatch:
        n_q, n_p = cost.shape
        p = self.params
        used = np.zeros(n_p, dtype=bool)

        if p.order == "random":
            order = self.rng.permutation(n_q)
        elif p.order == "hardest_first":
            nearest = np.where(cost < PENALTY_THRESHOLD, cost, np.inf).min(axis=1)
            order = np.argsort(-np.minimum(nearest, 1e12), kind="stable")
        else:
            order = np.arange(n_q)

        qs, ps, ds, pids = [], [], [], []
        next_pid = 0
        for i in order:
            row = cost[i]
            ok = row < PENALTY_THRESHOLD
            if not p.replacement:
                ok &= ~used
            cand = np.flatnonzero(ok)
            if cand.size == 0:
                continue
            k = min(p.ratio_k, cand.size)
            if k < cand.size:
                part = np.argpartition(row[cand], k - 1)[:k]
                chosen = cand[part[np.argsort(row[cand][part], kind="stable")]]
            else:
                chosen = cand[np.argsort(row[cand], kind="stable")]
            for j in chosen:
                qs.append(i)
                ps.append(j)
                ds.append(row[j])
                pids.append(next_pid)
            if not p.replacement:
                used[chosen] = True
            next_pid += 1

        return RawMatch(
            np.array(qs, dtype=int), np.array(ps, dtype=int), np.array(pids, dtype=int),
            np.array(ds, dtype=float), meta={"algorithm": "greedy_nn", "ratio_k": p.ratio_k},
        )
