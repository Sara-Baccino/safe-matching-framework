import json

import numpy as np
import polars as pl
import pytest

KINDS = {"Age": "numerical", "BMI": "numerical", "Male": "categorical", "Stage": "categorical"}

GROUP_LABELS = ["<35", "35-37", "38-39", "40-41", "42-45"]


@pytest.fixture(scope="session")
def population():
    rng = np.random.default_rng(3)
    n = 700
    age, bmi = rng.normal(55, 10, n), rng.normal(26, 4, n)
    male, stage = rng.integers(0, 2, n), rng.integers(0, 3, n)
    g = (rng.random(n) < 1 / (1 + np.exp(-(-4 + 0.06 * age + 0.4 * male)))).astype(np.int8)
    return pl.DataFrame({"__uid__": np.arange(n), "__group__": g, "Age": age, "BMI": bmi, "Male": male, "Stage": stage})


@pytest.fixture(scope="session")
def kinds():
    return dict(KINDS)


def _col(new, *, cat=False, num=False, typ="int", active=True, imp="knn", maps=None):
    return {"new_name": new, "active": active, "categorical": cat, "numerical": num, "mappings": maps or {},
            "missing_data_management": {"strategy": "impute", "imputer": imp, "missing_values": [], "missing_replace_value": None},
            "type": typ}


def make_cohort(rng, n, shift, name, treated):
    age = np.clip(rng.normal(38 + shift, 4, n).round(), 25, 45).astype(int)
    grp = np.select([age < 35, age < 38, age < 40, age < 42], GROUP_LABELS[:4], GROUP_LABELS[4])
    male = (rng.random(n) < 0.55 + 0.05 * (shift > 0))
    bmi = rng.normal(26 + 0.1 * shift, 4, n)
    outcome = 2.0 * treated + 0.08 * age + 0.15 * bmi + 1.0 * male + rng.normal(0, 2, n)
    bmi = np.where(np.arange(n) % 23 == 0, np.nan, bmi)   # mancanti nel BMI (variabile di matching)
    return pl.DataFrame({
        "ID": [f"{name}{i}" for i in range(n)], "Eta": age, "Fascia": grp, "Sesso": np.where(male, "M", "F"),
        "BMI": [None if np.isnan(x) else round(float(x), 1) for x in bmi], "Diabete": (rng.random(n) < 0.15).astype(int),
        "ALT": np.exp(rng.normal(3.6, 0.5, n)).round(1),
        "Outcome": outcome.round(2), "Evento": (rng.random(n) < 0.3).astype(int), "Note": [f"nota {i}" for i in range(n)],
    })


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """File di esempio + data config nel formato attuale (mappings, ordinale = categorical+numerical)."""
    d = tmp_path_factory.mktemp("demo")
    rng = np.random.default_rng(7)
    treated, controls = make_cohort(rng, 250, 2, "T", 1), make_cohort(rng, 900, 0, "C", 0)
    treated.write_csv(d / "treated.csv")
    controls.write_csv(d / "controls.csv")
    pl.concat([treated.with_columns(pl.lit(1).alias("Gruppo")), controls.with_columns(pl.lit(0).alias("Gruppo"))]).write_csv(d / "all.csv")
    cfg = {
        "ID": _col("ID", typ="str", active=False),
        "Eta": _col("Age", num=True, typ="int"),
        "Fascia": _col("Age_Group", cat=True, num=True, typ="str", maps={lab: i + 1 for i, lab in enumerate(GROUP_LABELS)}),
        "Sesso": _col("Male", cat=True, typ="str", maps={"M": 1, "F": 0}),
        "BMI": _col("BMI", num=True, typ="float"),
        "Diabete": _col("Diabetes", cat=True, typ="int"),
        "ALT": _col("ALT", num=True, typ="float"),
        "Outcome": _col("Outcome", num=True, typ="float"),
        "Evento": _col("Event", cat=True, typ="int"),
    }
    (d / "data_config.json").write_text(json.dumps(cfg, indent=2))
    return d


def matching_cfg(method="hungarian", params=None, metric="euclidean", matching=("Age", "BMI", "ALT"), estimation=False):
    c = {"variables": {"matching": list(matching), "stratification": ["Male"], "evaluation": ["Diabetes", "Age_Group"]},
         "constraints": {"calipers": []}, "metrics": {"name": metric}, "method": {"name": method, "params": params or {}},
         "validation": {"max_distribution_plots": 4}}
    if estimation:
        c["estimation"] = {"enabled": True, "outcomes": [{"variable": "Outcome"}, {"variable": "Event", "type": "binary"}],
                           "bootstrap": {"n_boot": 20}}
    return c