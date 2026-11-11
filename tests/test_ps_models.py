import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from safe_matching.exceptions import ConfigError
from safe_matching.matching import MatchingEngine
from safe_matching.schemas import MatchingConfig


def run(population, kinds, model, **extra):
    params = {"model": model, **extra}
    c = MatchingConfig.from_dict({"variables": {"matching": ["Age", "BMI", "Male"]},
                                  "method": {"name": "propensity_score", "params": params}})
    return MatchingEngine(c, kinds).run(population)


@pytest.mark.parametrize("model,mp", [("random_forest", {"n_estimators": 100}), ("xgboost", {"n_estimators": 60, "max_depth": 2}),
                                      ("gradient_boosting", {"n_estimators": 40})])
def test_tree_ps_models_are_valid_and_cross_fitted(population, kinds, model, mp):
    if model == "xgboost":
        pytest.importorskip("xgboost")
    out = run(population, kinds, model, model_params=mp)
    ps = out.population["__ps__"].to_numpy()
    assert ((ps > 0) & (ps < 1)).all() and out.metrics["n_query_matched"] > 0
    # fuori-campione: l'AUC non puo' essere vicina a 1 su dati dove il gruppo dipende debolmente dalle covariate
    assert roc_auc_score(out.population["__group__"], ps) < 0.85


def test_cross_fit_can_be_disabled_and_overfits(population, kinds):
    kw = {"model_params": {"n_estimators": 100, "min_samples_leaf": 1}, "caliper_sd": None}
    on = run(population, kinds, "random_forest", **kw)
    off = run(population, kinds, "random_forest", cross_fit=False, **kw)
    auc = lambda o: roc_auc_score(o.population["__group__"], o.population["__ps__"].to_numpy())
    assert auc(off) > auc(on)


def test_invalid_model_params_are_reported(population, kinds):
    with pytest.raises(ConfigError):
        run(population, kinds, "random_forest", model_params={"not_a_param": 1})
