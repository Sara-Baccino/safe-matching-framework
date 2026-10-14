"""Stima del propensity score P(gruppo=1 | covariate)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from safe_matching.exceptions import MatchingError
from safe_matching.schemas.method_params import PropensityScoreParams

__all__ = ["design_matrix", "fit_propensity"]


def design_matrix(df: pl.DataFrame, columns: list[str], kinds: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Numeriche e binarie cosi' come sono; categoriche con >2 livelli -> one-hot (drop first)."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    for c in columns:
        s = df[c]
        if kinds.get(c) == "categorical" and s.n_unique() > 2:
            levels = sorted(s.drop_nulls().unique().to_list(), key=str)[1:]
            for lv in levels:
                parts.append((s == lv).cast(pl.Float64).to_numpy())
                names.append(f"{c}={lv}")
        else:
            parts.append(s.cast(pl.Float64).to_numpy())
            names.append(c)
    return np.column_stack(parts), names


def fit_propensity(
    X: np.ndarray, y: np.ndarray, params: PropensityScoreParams, random_state: int | None
) -> tuple[np.ndarray, np.ndarray, Any]:
    if len(np.unique(y)) < 2:
        raise MatchingError("Propensity score: serve almeno un'unita' per gruppo.")
    if params.model == "logistic":
        steps = []
        if params.poly_degree > 1:
            steps.append(("poly", PolynomialFeatures(degree=params.poly_degree, include_bias=False)))
        steps += [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=params.C, max_iter=params.max_iter, random_state=random_state)),
        ]
        model: Any = Pipeline(steps)
    elif params.model == "random_forest":
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state)
    else:
        model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X, y)
    ps = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return ps, logit(ps), model
