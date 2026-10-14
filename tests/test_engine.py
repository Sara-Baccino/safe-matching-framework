import numpy as np
import pytest

from safe_matching.matching import MatchingEngine
from safe_matching.schemas import MatchingConfig


def cfg(method, params=None, metric="euclidean", matching=("Age", "BMI"), strat=(), cal=(), gc=None):
    return MatchingConfig.from_dict({
        "variables": {"matching": list(matching), "stratification": list(strat)},
        "constraints": {"calipers": list(cal), "global_caliper": gc},
        "metrics": {"name": metric}, "method": {"name": method, "params": params or {}},
    })


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "mahalanobis", "hybrid", "gower"])
@pytest.mark.parametrize("method", ["greedy_nn", "hungarian"])
def test_pairs_are_unique_and_balanced_in_size(population, kinds, method, metric):
    out = MatchingEngine(cfg(method, metric=metric, matching=("Age", "BMI", "Male")), kinds).run(population)
    q = out.matched.filter(out.matched["match_role"] == "query")
    p = out.matched.filter(out.matched["match_role"] == "match")
    assert q.height == p.height == out.metrics["n_query_matched"]
    assert p["__uid__"].n_unique() == p.height  # senza reinserimento


def test_stratification_is_exact(population, kinds):
    out = MatchingEngine(cfg("hungarian", strat=["Male", "Stage"]), kinds).run(population)
    pairs = out.matched.group_by("pair_id").agg(__import__("polars").col("Male").n_unique().alias("nm"),
                                                __import__("polars").col("Stage").n_unique().alias("ns"))
    assert pairs["nm"].max() == 1 and pairs["ns"].max() == 1


def test_variable_caliper_is_hard(population, kinds):
    import polars as pl
    out = MatchingEngine(cfg("hungarian", cal=[{"variable": "Age", "value": 2}]), kinds).run(population)
    g = out.matched.filter(pl.col("match_role") != "unused").group_by("pair_id").agg(
        (pl.col("Age").max() - pl.col("Age").min()).alias("d"))
    assert g["d"].max() <= 2 + 1e-9


def test_global_caliper_limits_distance(population, kinds):
    out = MatchingEngine(cfg("greedy_nn", gc={"value": 0.3}), kinds).run(population)
    assert out.metrics["distance_max"] <= 0.3 + 1e-9


def test_propensity_score_and_ps_sorted(population, kinds):
    for alg in ("greedy_nn", "hungarian", "sorted"):
        out = MatchingEngine(cfg("propensity_score", {"algorithm": alg}), kinds).run(population)
        assert out.has_ps and out.metrics["n_query_matched"] > 0


def test_partial_ot_selects_target(population, kinds):
    out = MatchingEngine(cfg("partial_ot", {"target_size": 150}), kinds).run(population)
    sel = out.matched.filter(out.matched["match_role"] == "selected")
    assert sel.height == 150 and out.output_type == "selection"
    assert np.isclose(sel["weight"].sum(), (population["__group__"] == 1).sum())


def test_sorted_requires_single_variable(population, kinds):
    from safe_matching.exceptions import ConfigError
    with pytest.raises(ConfigError):
        MatchingEngine(cfg("sorted"), kinds).run(population)
    assert MatchingEngine(cfg("sorted", {"sort_variable": "Age"}), kinds).run(population).metrics["n_query_matched"] > 0


def test_missing_rows_are_excluded(population, kinds):
    import polars as pl
    df = population.with_columns(pl.when(pl.col("__uid__") % 10 == 0).then(None).otherwise(pl.col("BMI")).alias("BMI"))
    out = MatchingEngine(cfg("hungarian"), kinds).run(df)
    assert out.metrics["n_excluded_missing"] == 70
