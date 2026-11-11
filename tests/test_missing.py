import numpy as np
import polars as pl
import pytest

from safe_matching import run_pipeline
from safe_matching.exceptions import ConfigError
from safe_matching.preprocessing.missing import ColumnRule, impute_frame
from safe_matching.schemas.matching_config import MissingPolicy

from conftest import matching_cfg


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 60, n)
    bmi = 18 + 0.2 * age + rng.normal(0, 1, n)
    sex = (rng.random(n) < 0.3).astype(int)
    stage = rng.choice([1, 2, 3, 4, 5], n, p=[.1, .2, .4, .2, .1])
    return pl.DataFrame({"Age": age, "BMI": bmi, "Sex": sex, "Stage": stage, "__group__": (rng.random(n) < 0.4).astype(np.int8)})


def run(df, policy=None, rules=None, targets=None):
    kinds = {"Age": "numerical", "BMI": "numerical", "Sex": "categorical", "Stage": "categorical"}
    targets = targets or {"Age": "matching", "BMI": "matching", "Sex": "stratification", "Stage": "evaluation"}
    return impute_frame(df, source="s", targets=targets, kinds=kinds, ordinal={"Stage"}, policy=policy or MissingPolicy(),
                        rules=rules or {}, group_col="__group__")


def blank(df, col, idx):
    v = df[col].cast(pl.Float64).to_numpy().copy()
    v[idx] = np.nan
    return df.with_columns(pl.Series(col, v, nan_to_null=True).cast(df.schema[col]))


def test_continuous_knn_is_close_to_truth_integers_stay_integers_and_cells_are_flagged():
    df = frame()
    idx = np.arange(0, 400, 25)                                  # 4% di mancanti: fascia 'low'
    miss = blank(blank(df, "BMI", idx), "Age", idx[::2] + 1)
    out, rep, dropped = run(miss)
    assert out["BMI"].null_count() == 0 and out["Age"].null_count() == 0 and not dropped
    assert out["Age"].dtype == pl.Int64                          # l'eta' imputata resta intera
    err = np.abs(out["BMI"].to_numpy()[idx] - df["BMI"].to_numpy()[idx]).mean()
    assert err < 1.5                                             # il knn usa l'eta': molto meglio dell'errore della media (~2)
    assert out["__imp_BMI"].sum() == len(idx) and out["__imp_Age"].sum() == len(idx[::2])
    assert {r["variable"]: r["tier"] for r in rep}["BMI"] == "low"


def test_nominal_gets_mode_and_ordinal_gets_an_observed_median_level():
    df = frame()
    miss = blank(blank(df, "Sex", np.arange(0, 400, 20)), "Stage", np.arange(1, 400, 20))
    out, rep, _ = run(miss)
    assert set(out["Sex"].unique()) <= {0, 1} and out["Sex"].null_count() == 0
    assert out.filter(pl.col("__imp_Sex"))["Sex"].unique().to_list() == [int(df["Sex"].mode()[0])]   # moda
    assert out.filter(pl.col("__imp_Stage"))["Stage"].unique().to_list() == [3]                        # mediana dei livelli 1..5
    assert {r["variable"]: r["method"] for r in rep}["Stage"] == "median"


def test_tiers_and_over_threshold_actions():
    df = blank(frame(), "BMI", np.arange(0, 400, 2)[:160])        # 40% mancanti: 'high'
    out, rep, dropped = run(df)
    assert dropped == ["BMI"] and out["BMI"].null_count() == 160
    assert [r["action"] for r in rep if r["variable"] == "BMI"] == ["dropped_variable"]
    with pytest.raises(ConfigError):
        run(df, MissingPolicy(over_threshold_action="error"))
    out2, rep2, dropped2 = run(df, MissingPolicy(over_threshold_action="impute"))
    assert not dropped2 and out2["BMI"].null_count() == 0
    mid = blank(frame(), "BMI", np.arange(0, 400, 5)[:60])        # 15%: 'moderate'
    assert [r["tier"] for r in run(mid)[1] if r["variable"] == "BMI"] == ["moderate"]


def test_data_config_maintain_and_impute_false_leave_values_missing():
    miss = blank(frame(), "BMI", np.arange(0, 400, 25))
    out, rep, _ = run(miss, rules={"BMI": ColumnRule("maintain")})
    assert out["BMI"].null_count() == 16 and [r["action"] for r in rep if r["variable"] == "BMI"] == ["left_missing"]
    out, _, _ = run(miss, MissingPolicy(impute=False))
    assert out["BMI"].null_count() == 16 and "__imp_BMI" not in out.columns


def test_replace_strategy_and_column_imputer_override():
    miss = blank(frame(), "BMI", np.arange(0, 400, 25))
    out, _, _ = run(miss, rules={"BMI": ColumnRule("replace", replace_value=99)})
    assert set(out.filter(pl.col("__imp_BMI"))["BMI"].unique().to_list()) == {99.0}
    med = float(miss["BMI"].drop_nulls().median())
    out, _, _ = run(miss, MissingPolicy(use_column_imputer=True), rules={"BMI": ColumnRule("impute", "median")})
    assert out.filter(pl.col("__imp_BMI"))["BMI"].unique().to_list() == [pytest.approx(med)]


def test_rows_with_too_many_missing_are_not_imputed():
    df = frame()
    bad = np.array([3, 7])
    for c in ("Age", "BMI", "Sex"):
        df = blank(df, c, bad)
    out, rep, _ = run(df)
    assert out["BMI"].is_null().to_numpy()[bad].all() and out["BMI"].null_count() == 2
    assert [r["n_left_missing"] for r in rep if r["variable"] == "BMI"] == [2]


def test_group_is_used_by_knn_only_if_requested():
    rng = np.random.default_rng(1)
    n = 600
    g = (np.arange(n) % 2).astype(np.int8)
    df = pl.DataFrame({"Age": rng.normal(50, 1, n), "BMI": 20.0 + 10 * g + rng.normal(0, .5, n), "__group__": g})
    idx = np.arange(0, n, 10)
    miss = blank(df, "BMI", idx)
    kw = dict(source="s", targets={"Age": "matching", "BMI": "matching"}, kinds={"Age": "numerical", "BMI": "numerical"},
              ordinal=set(), rules={}, group_col="__group__")
    with_g = impute_frame(miss, policy=MissingPolicy(include_group_in_imputation=True), **kw)[0]["BMI"].to_numpy()[idx]
    without = impute_frame(miss, policy=MissingPolicy(include_group_in_imputation=False), **kw)[0]["BMI"].to_numpy()[idx]
    truth = df["BMI"].to_numpy()[idx]
    assert np.abs(with_g - truth).mean() < 1.0 < np.abs(without - truth).mean()   # senza gruppo l'imputazione cancella lo sbilanciamento


# ----------------------------------------------------------------------------------------------- pipeline
def test_pipeline_imputes_flags_and_reports(demo, tmp_path):
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], matching_cfg(),
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.outcome.metrics["n_excluded_missing"] == 0
    mr = r.prepared.missing_report.filter(pl.col("variable") == "BMI")
    assert mr["action"].to_list() == ["imputed", "imputed"] and mr["n_imputed"].sum() > 0
    out = pl.read_csv(tmp_path / "matching_dataset.csv")
    assert out["BMI"].null_count() == 0 and out["imputed_BMI"].sum() > 0 and "imputed_Eta" not in out.columns
    assert (tmp_path / "matching_missing_report.csv").exists()


def test_pipeline_without_imputation_excludes_rows(demo, tmp_path):
    cfg = matching_cfg()
    cfg["options"] = {"missing": {"impute": False}}
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], cfg,
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.outcome.metrics["n_excluded_missing"] > 0


def test_pipeline_never_imputes_outcomes(demo, tmp_path):
    t = pl.read_csv(demo / "treated.csv").with_columns(
        pl.when(pl.int_range(0, 250) % 10 == 0).then(None).otherwise(pl.col("Outcome")).alias("Outcome"))
    r = run_pipeline([t, demo / "controls.csv"], [demo / "data_config.json"], matching_cfg("greedy_nn", {"replacement": True}, estimation=True),
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.prepared.data.filter(pl.col("__group__") == 1)["Outcome"].null_count() == 25
    assert "Outcome" not in set(r.prepared.missing_report["variable"])


def test_pipeline_drops_variable_with_too_many_missing(demo, tmp_path):
    t = pl.read_csv(demo / "treated.csv").with_columns(
        pl.when(pl.int_range(0, 250) % 2 == 0).then(None).otherwise(pl.col("ALT")).alias("ALT"))
    r = run_pipeline([t, demo / "controls.csv"], [demo / "data_config.json"], matching_cfg(), output_dir=tmp_path, output_formats=("csv",))
    assert r.prepared.dropped_variables == ["ALT"]
    assert any("ALT" in n and "missing" in n for n in r.outcome.notes)
    assert "ALT" not in r.balance.table["base_variable"].to_list() and r.outcome.metrics["n_query_matched"] > 0
    with pytest.raises(ConfigError):
        cfg = matching_cfg(matching=("ALT",))
        cfg["variables"]["stratification"] = []
        run_pipeline([t, demo / "controls.csv"], [demo / "data_config.json"], cfg, output_dir=tmp_path / "x")