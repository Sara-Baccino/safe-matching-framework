"""Distanze ibride per covariate miste (numeriche + categoriche)."""

from __future__ import annotations

import numpy as np

from safe_matching.exceptions import ConfigError
from safe_matching.matching.distances.base import DistanceMetric

__all__ = ["HybridDistance", "GowerDistance"]


class HybridDistance(DistanceMetric):
    """D = (w_num * D_num + w_cat * D_cat) / (w_num + w_cat).

    * D_num: media (pesata per variabile) di |dx| / range sulle variabili numeriche (L1 normalizzata);
    * D_cat: media (pesata per variabile) del mismatch 0/1 sulle categoriche (Hamming).

    Se manca uno dei due blocchi, il peso dell'altro e' 1. D e' in [0, 1].
    ``weight_numerical``/``weight_categorical`` (default 0.6/0.4) pesano i blocchi;
    ``weights`` pesa le singole variabili all'interno del blocco.
    """

    name = "hybrid"
    allowed_params = frozenset({"weight_numerical", "weight_categorical", "weights"})
    default_weights: tuple[float | None, float | None] = (0.6, 0.4)

    def _fit(self, X, is_query) -> None:
        self._num = [j for j, c in enumerate(self.columns) if self.kinds.get(c) == "numerical"]
        self._cat = [j for j, c in enumerate(self.columns) if self.kinds.get(c) == "categorical"]
        if not self._num and not self._cat:
            raise ConfigError("hybrid: nessuna variabile tipizzata come numerical/categorical.")
        wn = self.params.get("weight_numerical", self.default_weights[0])
        wc = self.params.get("weight_categorical", self.default_weights[1])
        if wn is None or wc is None:  # Gower: pesi proporzionali al numero di variabili
            wn, wc = float(len(self._num)), float(len(self._cat))
        if wn < 0 or wc < 0:
            raise ConfigError("weight_numerical/weight_categorical devono essere >= 0.")
        if not self._cat:
            wn, wc = 1.0, 0.0
        elif not self._num:
            wn, wc = 0.0, 1.0
        if wn + wc == 0:
            raise ConfigError("hybrid: la somma dei pesi dei blocchi e' 0.")
        self._wn, self._wc = wn / (wn + wc), wc / (wn + wc)
        w = self._weight_vector()
        self._w_num = w[self._num] if self._num else np.array([])
        self._w_cat = w[self._cat] if self._cat else np.array([])
        if self._num and self._w_num.sum() == 0 or self._cat and self._w_cat.sum() == 0:
            raise ConfigError("hybrid: tutti i pesi di un blocco sono 0.")
        self._range = np.ptp(X[:, self._num], axis=0) if self._num else np.array([])
        self._range[self._range == 0] = 1.0

    def pairwise(self, Xq, Xp) -> np.ndarray:
        Xq, Xp = np.asarray(Xq, float), np.asarray(Xp, float)
        D = np.zeros((Xq.shape[0], Xp.shape[0]))
        if self._num:
            acc = np.zeros_like(D)
            for k, j in enumerate(self._num):
                acc += self._w_num[k] * np.abs(Xq[:, j][:, None] - Xp[:, j][None, :]) / self._range[k]
            D += self._wn * acc / self._w_num.sum()
        if self._cat:
            acc = np.zeros_like(D)
            for k, j in enumerate(self._cat):
                acc += self._w_cat[k] * (Xq[:, j][:, None] != Xp[:, j][None, :])
            D += self._wc * acc / self._w_cat.sum()
        return D


class GowerDistance(HybridDistance):
    """Gower (1971): media non pesata sulle variabili (pesi blocco proporzionali al numero di variabili)."""

    name = "gower"
    default_weights = (None, None)
