import numpy as np
import polars as pl
import pytest

from safe_matching.exceptions import ConfigError
from safe_matching.estimation import estimate_effects
from safe_matching.matching import MatchingEngine
from safe_matching.schemas import MatchingConfig

KINDS = {"Age": "numerical", "BMI": "numerical", "Male": "categorical"}
TRUE_EFFECT = 2.0


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n = 900
    age, bmi, male = rng.normal(55, 10, n), rng.normal(26, 4, n), rng.integers(0, 2, n)
    g = (rng.random(n) < 1 / (1 + np.exp(-(-3.2 + 0.05 * age + 0.3 * male)))).astype(np.int8)
    y = TRUE_EFFECT * g + 0.1 * age + 0.2 * bmi + rng.normal(0, 2, n)
    ev = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.8 * g)))).astype(int)
    return pl.DataFrame({"__uid__": np.arange(n), "__group__": g, "Age": age, "BMI": bmi, "Male": male, "Y": y, "Ev": ev})


def cfg(method="greedy_nn", params=None, resample="full_pipeline", estimands=("ATT",), n_boot=40):
    return MatchingConfig.from_dict({
        "variables": {"matching": ["Age", "BMI", "Male"]}, "method": {"name": method, "params": params or {}},
        "estimation": {"enabled": True, "estimands": list(estimands),
                       "outcomes": [{"variable": "Y"}, {"variable": "Ev", "type": "binary"}],
                       "bootstrap": {"n_boot": n_boot, "resample": resample}}})


@pytest.mark.parametrize("resample", ["full_pipeline", "matched_pairs"])
def test_att_recovers_true_effect_and_beats_naive(data, resample):
    c = cfg(params={"replacement": True}, resample=resample, estimands=("ATT", "ATC", "ATE"))
    out = MatchingEngine(c, KINDS).run(data)
    r = estimate_effects(out.population, c, KINDS, {"treated_to_control": out.matched})
    t = r.table.filter((pl.col("outcome") == "Y") & (pl.col("measure") == "difference"))
    att = t.filter(pl.col("estimand") == "ATT").row(0, named=True)
    naive = t.filter(pl.col("estimator") == "naive").row(0, named=True)
    assert att["ci_low_pct"] <= TRUE_EFFECT <= att["ci_high_pct"]
    assert abs(att["estimate"] - TRUE_EFFECT) < abs(naive["estimate"] - TRUE_EFFECT)
    assert att["se_boot"] > 0 and att["coverage"] == 1.0
    assert {"ATT", "ATC", "ATE"} <= set(t["estimand"].to_list())


def test_binary_outcome_has_risk_ratio(data):
    c = cfg(params={"replacement": True})
    out = MatchingEngine(c, KINDS).run(data)
    r = estimate_effects(out.population, c, KINDS, {"treated_to_control": out.matched})
    assert r.table.filter((pl.col("outcome") == "Ev") & (pl.col("measure") == "risk_ratio")).height >= 1


def test_low_coverage_is_flagged_for_atc_with_hungarian(data):
    c = cfg("hungarian", estimands=("ATC",), resample="matched_pairs")
    out = MatchingEngine(c, KINDS).run(data)
    r = estimate_effects(out.population, c, KINDS, {"treated_to_control": out.matched})
    assert any("Coverage" in n for n in r.notes)
    assert r.table.filter(pl.col("estimand") == "ATC")["coverage"].min() < 0.9


def test_matched_pairs_not_applicable_to_ot(data):
    c = cfg("partial_ot", {"target_ratio": 1.0}, resample="matched_pairs", n_boot=20)
    out = MatchingEngine(c, KINDS).run(data)
    r = estimate_effects(out.population, c, KINDS, {"treated_to_control": out.matched})
    assert r.metrics["resample"] == "full_pipeline"


def test_outcome_cannot_be_matching_variable():
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["Age"]}, "method": {"name": "hungarian"},
                                  "estimation": {"enabled": True, "outcomes": [{"variable": "Age"}]}})


def test_enabled_requires_outcomes():
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["Age"]}, "method": {"name": "hungarian"}, "estimation": {"enabled": True}})
