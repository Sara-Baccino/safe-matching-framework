"""Matching ottimo monotono su una variabile ordinabile (programmazione dinamica).

Per costi |x - y| esiste sempre un abbinamento ottimo che preserva l'ordine; la DP
sulle sequenze ordinate lo trova in O(n*m) (n <= m) senza costruire un problema di
assegnazione generale. Sono ammessi "salti" (query non abbinate) con penalita' tale da
massimizzare prima il numero di coppie e poi minimizzare la distanza totale, cosi' i
caliper (celle vietate) vengono rispettati.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from safe_matching.exceptions import MatchingError
from safe_matching.matching.algorithms.base import MatchingAlgorithm, RawMatch
from safe_matching.matching.constraints.caliper import PENALTY_THRESHOLD

__all__ = ["SortedDP", "monotone_dp"]


def monotone_dp(a: np.ndarray, b: np.ndarray, feasible: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Abbinamento ottimo monotono tra ``a`` (n) e ``b`` (m), entrambi ORDINATI, con n <= m.

    Ritorna le coppie (i, j) in indici delle sequenze ordinate."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    span = float(max(np.ptp(a), np.ptp(b), abs(a.max() - b.min()), abs(b.max() - a.min()), 1e-12))
    skip_pen = (span + 1.0) * (n + 1)
    prev = np.zeros(m + 1)            # dp[0][j] = 0
    kind = np.zeros((n, m + 1), dtype=np.int8)  # 0 = da sinistra, 1 = abbina, 2 = salta la riga
    inf = np.inf
    for i in range(n):
        c = np.abs(a[i] - b)
        if feasible is not None:
            c = np.where(feasible[i], c, inf)
        mt = prev[:-1] + c            # abbina a_i con b_j     (j = 0..m-1)
        sk = prev[1:] + skip_pen      # a_i resta non abbinata (stato con j+1 elementi di pool)
        e = np.minimum(mt, sk)
        cur0 = prev[0] + skip_pen
        cur = np.empty(m + 1)
        cur[0] = cur0
        cur[1:] = np.minimum.accumulate(np.concatenate(([cur0], e)))[1:]
        left = cur[:-1]               # cur[j-1] per j=1..m
        use_e = e < left
        kind[i, 1:] = np.where(use_e, np.where(mt <= sk, 1, 2), 0)
        prev = cur

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        k = kind[i - 1, j]
        if k == 0:
            j -= 1
        elif k == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        else:
            i -= 1
    pairs.reverse()
    return pairs


class SortedDP(MatchingAlgorithm):
    name = "sorted"

    def match(self, cost: np.ndarray, aux: dict[str, Any]) -> RawMatch:
        if "sort_q" not in aux or "sort_p" not in aux:
            raise MatchingError("Il metodo 'sorted' richiede i valori di ordinamento (sort_q/sort_p).")
        sq, sp = np.asarray(aux["sort_q"], float), np.asarray(aux["sort_p"], float)
        n_q, n_p = cost.shape
        feas = cost < PENALTY_THRESHOLD
        if not feas.any():
            return RawMatch.empty()

        oq, op = np.argsort(sq, kind="stable"), np.argsort(sp, kind="stable")
        sq_s, sp_s = sq[oq], sp[op]
        feas_s = feas[np.ix_(oq, op)]

        if n_q <= n_p:
            pairs = monotone_dp(sq_s, sp_s, feas_s)
            qi = np.array([oq[i] for i, _ in pairs], dtype=int)
            pj = np.array([op[j] for _, j in pairs], dtype=int)
        else:  # il lato piu' piccolo (il pool) fa da "righe"; poi si scambiano i ruoli
            pairs = monotone_dp(sp_s, sq_s, feas_s.T)
            qi = np.array([oq[j] for _, j in pairs], dtype=int)
            pj = np.array([op[i] for i, _ in pairs], dtype=int)

        order = np.argsort(qi, kind="stable")
        qi, pj = qi[order], pj[order]
        return RawMatch(
            qi, pj, np.arange(qi.size, dtype=int), cost[qi, pj].astype(float) if qi.size else np.array([]),
            meta={"algorithm": "sorted"},
        )
