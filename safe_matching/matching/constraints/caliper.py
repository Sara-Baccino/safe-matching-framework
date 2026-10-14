"""Finestre di tolleranza hard per variabile (caliper) applicate a coppie."""

from __future__ import annotations

import numpy as np
import polars as pl

from safe_matching.schemas.matching_config import CaliperWindow

__all__ = ["PENALTY", "PENALTY_THRESHOLD", "window_mask", "apply_mask"]

PENALTY = 1e10
"""Costo assegnato alle coppie vietate; ogni valore >= PENALTY_THRESHOLD e' 'infattibile'."""
PENALTY_THRESHOLD = 1e9


def window_mask(
    q_df: pl.DataFrame,
    p_df: pl.DataFrame,
    windows: list[CaliperWindow],
    kinds: dict[str, str],
    sd: dict[str, float],
) -> np.ndarray:
    """Matrice booleana (n_query, n_pool): True se la coppia rispetta TUTTE le finestre.

    Per variabili categoriche la finestra equivale all'uguaglianza esatta."""
    mask = np.ones((q_df.height, p_df.height), dtype=bool)
    for w in windows:
        qv = q_df[w.variable].to_numpy()
        pv = p_df[w.variable].to_numpy()
        if kinds.get(w.variable) == "categorical":
            ok = qv[:, None] == pv[None, :]
        else:
            thr = w.value * (sd.get(w.variable, 1.0) if w.scale == "standard_deviation" else 1.0)
            ok = np.abs(qv[:, None].astype(float) - pv[None, :].astype(float)) <= thr
        mask &= ok
    return mask


def apply_mask(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cost = cost.copy()
    cost[~mask] = PENALTY
    return cost
