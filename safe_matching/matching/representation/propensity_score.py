"""Stima del propensity score P(gruppo=1 | covariate): logistica, random forest, XGBoost, gradient boosting."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from safe_matching.exceptions import ConfigError, MatchingError, MissingDependencyError
from safe_matching.schemas.method_params import PropensityScoreParams

__all__ = ["design_matrix", "fit_propensity", "build_ps_model"]

_TREE_MODELS = ("random_forest", "xgboost", "gradient_boosting")


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


def build_ps_model(params: PropensityScoreParams, random_state: int | None) -> Any:
    """Costruisce lo stimatore. I default sono prudenti (alberi poco profondi / foglie grandi) perche' un PS
    troppo flessibile degrada la sovrapposizione; ``model_params`` li sovrascrive."""
    mp = dict(params.model_params)
    try:
        if params.model == "logistic":
            clf_kw = {"C": params.C, "max_iter": params.max_iter, "random_state": random_state, **mp}
            steps = []
            if params.poly_degree > 1:
                steps.append(("poly", PolynomialFeatures(degree=params.poly_degree, include_bias=False)))
            steps += [("scaler", StandardScaler()), ("clf", LogisticRegression(**clf_kw))]
            return Pipeline(steps)
        if params.model == "random_forest":
            kw = {"n_estimators": 500, "min_samples_leaf": 10, "max_features": "sqrt", "n_jobs": -1,
                  "random_state": random_state, **mp}
            return RandomForestClassifier(**kw)
        if params.model == "gradient_boosting":
            return GradientBoostingClassifier(**{"random_state": random_state, **mp})
        if params.model == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError("model='xgboost' richiede il pacchetto xgboost (uv add xgboost).") from exc
            kw = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_lambda": 1.0, "min_child_weight": 5, "tree_method": "hist", "eval_metric": "logloss",
                  "n_jobs": -1, "random_state": random_state, **mp}
            return XGBClassifier(**kw)
    except TypeError as exc:
        raise ConfigError(f"model_params non valido per '{params.model}': {exc}") from exc
    raise ConfigError(f"Modello PS '{params.model}' non supportato.")


def fit_propensity(
    X: np.ndarray, y: np.ndarray, params: PropensityScoreParams, random_state: int | None
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Ritorna (ps, logit(ps), modello finale addestrato su tutti i dati).

    Con ``cross_fit`` il PS di ogni unita' e' predetto da un modello addestrato SENZA quella unita' (K-fold
    stratificato): evita che alberi/boosting memorizzino il gruppo e producano PS ~0/1 in-sample."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        raise MatchingError("Propensity score: serve almeno un'unita' per gruppo.")
    model = build_ps_model(params, random_state)
    use_cf = params.cross_fit if params.cross_fit is not None else params.model in _TREE_MODELS
    if use_cf:
        k = int(min(params.cross_fit_folds, np.bincount(y).min()))
        if k < 2:
            raise MatchingError("Cross-fitting impossibile: meno di 2 unita' in un gruppo.")
        ps = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(X, y):
            ps[te] = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        model.fit(X, y)
    else:
        model.fit(X, y)
        ps = model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    return ps, logit(ps), model
