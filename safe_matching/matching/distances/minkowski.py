"""Distanze euclidea (L2) e Manhattan (L1) su variabili scalate."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from safe_matching.exceptions import ConfigError
from safe_matching.matching.distances.base import DistanceMetric

__all__ = ["EuclideanDistance", "ManhattanDistance"]


class _Lp(DistanceMetric):
    scipy_metric = ""
    allowed_params = frozenset({"scale", "weights"})

    def _fit(self, X, is_query) -> None:
        mode = self.params.get("scale", "standard")
        if mode not in ("standard", "minmax", "none"):
            raise ConfigError("params.scale deve essere 'standard', 'minmax' o 'none'.")
        p = X.shape[1]
        center, scale = np.zeros(p), np.ones(p)
        if mode != "none":
            for j, c in enumerate(self.columns):
                if self.kinds.get(c) != "numerical":
                    continue  # le categoriche (0/1) restano nella scala originale
                col = X[:, j]
                if mode == "standard":
                    center[j], scale[j] = col.mean(), col.std(ddof=1) if col.size > 1 else 1.0
                else:
                    center[j], scale[j] = col.min(), np.ptp(col)
        scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
        self._center, self._scale, self._w = center, scale, self._weight_vector()

    def _tf(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._center) / self._scale * self._w

    def pairwise(self, Xq, Xp) -> np.ndarray:
        return cdist(self._tf(Xq), self._tf(Xp), metric=self.scipy_metric)


class EuclideanDistance(_Lp):
    name = "euclidean"
    scipy_metric = "euclidean"


class ManhattanDistance(_Lp):
    name = "manhattan"
    scipy_metric = "cityblock"
