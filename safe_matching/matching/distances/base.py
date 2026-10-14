"""Interfaccia comune delle metriche di distanza."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from safe_matching.exceptions import ConfigError

__all__ = ["DistanceMetric"]


class DistanceMetric(ABC):
    """Metrica con stato: ``fit`` stima scale/covarianze su TUTTA la popolazione
    eleggibile (non per stratum, cosi' le distanze sono comparabili tra strati),
    ``pairwise`` restituisce la matrice (n_query, n_pool)."""

    name: str = ""
    allowed_params: frozenset[str] = frozenset()

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.allowed_params)
        if unknown:
            raise ConfigError(
                f"Parametri sconosciuti per la metrica '{self.name}': {sorted(unknown)}. "
                f"Ammessi: {sorted(self.allowed_params)}"
            )
        self.params = params
        self.columns: list[str] = []
        self.kinds: dict[str, str] = {}

    def fit(
        self,
        X: np.ndarray,
        columns: list[str],
        kinds: dict[str, str],
        is_query: np.ndarray | None = None,
    ) -> "DistanceMetric":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(columns):
            raise ConfigError("Dimensioni di X incoerenti con le colonne dichiarate.")
        self.columns = list(columns)
        self.kinds = dict(kinds)
        self._fit(X, is_query)
        return self

    def _fit(self, X: np.ndarray, is_query: np.ndarray | None) -> None:  # noqa: D401
        """Stima parametri; le sottoclassi sovrascrivono se serve."""

    @abstractmethod
    def pairwise(self, Xq: np.ndarray, Xp: np.ndarray) -> np.ndarray:
        """Matrice (n_query, n_pool) di distanze."""

    # ------------------------------------------------------------------ #
    def _weight_vector(self) -> np.ndarray:
        weights: dict[str, float] = self.params.get("weights") or {}
        unknown = set(weights) - set(self.columns)
        if unknown:
            raise ConfigError(f"weights per variabili non usate nel matching: {sorted(unknown)}")
        if any(v < 0 for v in weights.values()):
            raise ConfigError("weights deve contenere valori >= 0.")
        return np.array([float(weights.get(c, 1.0)) for c in self.columns])
