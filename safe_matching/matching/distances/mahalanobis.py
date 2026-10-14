"""Distanza di Mahalanobis con pseudo-inversa regolarizzata."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import cdist

from safe_matching.exceptions import ConfigError, MatchingError
from safe_matching.matching.distances.base import DistanceMetric

__all__ = ["MahalanobisDistance"]


class MahalanobisDistance(DistanceMetric):
    """``params.covariance``: 'total' (covarianza dell'intera popolazione, default)
    oppure 'within' (covarianza pooled entro-gruppo, come in Rubin)."""

    name = "mahalanobis"
    allowed_params = frozenset({"covariance", "regularization"})

    def _fit(self, X, is_query) -> None:
        mode = self.params.get("covariance", "total")
        if mode not in ("total", "within"):
            raise ConfigError("params.covariance deve essere 'total' o 'within'.")
        n, p = X.shape
        if n < p + 2:
            raise MatchingError(
                f"Mahalanobis richiede almeno {p + 2} osservazioni per {p} variabili (trovate {n}). "
                "Riduci le variabili di matching o usa 'euclidean'."
            )
        if mode == "within" and is_query is not None and 1 < is_query.sum() < n - 1:
            Xq, Xp = X[is_query], X[~is_query]
            nq, np_ = len(Xq), len(Xp)
            cov = ((nq - 1) * np.atleast_2d(np.cov(Xq.T)) + (np_ - 1) * np.atleast_2d(np.cov(Xp.T))) / (nq + np_ - 2)
        else:
            cov = np.atleast_2d(np.cov(X.T))
        eps = float(self.params.get("regularization", 1e-8))
        cov = cov + np.eye(p) * eps * max(1.0, float(np.trace(cov)) / p)
        if np.linalg.matrix_rank(cov) < p:
            warnings.warn("Covarianza singolare (variabili collineari): uso la pseudo-inversa.", RuntimeWarning)
        self._vi = np.linalg.pinv(cov)

    def pairwise(self, Xq, Xp) -> np.ndarray:
        return cdist(np.asarray(Xq, float), np.asarray(Xp, float), metric="mahalanobis", VI=self._vi)
