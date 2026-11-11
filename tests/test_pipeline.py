"""Test end-to-end: usano solo file e config generati dai fixture (non dipendono dai file in configs/)."""
import json

import polars as pl
import pytest

from safe_matching import run_pipeline
from safe_matching.exceptions import ConfigError
from safe_matching.preprocessing.transform import UnmappedValuesError

from conftest import GROUP_LABELS, matching_cfg

CASES = {
    "hungarian": dict(method="hungarian", metric="mahalanobis"),
    "ps_logistic": dict(method="propensity_score", params={"algorithm": "greedy_nn"}),
    "ps_random_forest": dict(method="propensity_score", params={"model": "random_forest", "model_params": {"n_estimators": 60}, "algorithm": "hungarian"}),
    "ps_xgboost": dict(method="propensity_score", params={"model": "xgboost", "model_params": {"n_estimators": 40, "max_depth": 2}}),
    "partial_ot": dict(method="partial_ot", params={"target_ratio": 1.0}, metric="manhattan"),
    "sorted": dict(method="sorted", params={"sort_variable": "Age"}, matching=("Age",)),
    "greedy_hybrid": dict(method="greedy_nn", params={"ratio_k": 2}, metric="hybrid"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_two_datasets_all_methods(demo, tmp_path, case):
    if case == "ps_xgboost":
        pytest.importorskip("xgboost")
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], matching_cfg(**CASES[case]),
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.report_path.stat().st_size > 10_000 and "ps_auc" in r.analysis.metrics
    s = r.balance.summary
    assert s["mean_abs_smd_after"] <= s["mean_abs_smd_before"] + 0.03
    assert r.outcome.metrics["n_excluded_missing"] == 0              # i mancanti del BMI sono imputati (options.missing)


def test_output_has_original_names_labels_types_and_metadata(demo, tmp_path):
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], matching_cfg(),
                     output_dir=tmp_path, output_formats=("csv", "xlsx"))
    out = pl.read_csv(tmp_path / "matching_dataset.csv")
    assert {"ID", "Eta", "Fascia", "Sesso", "BMI", "Diabete", "ALT", "Outcome", "Note"} <= set(out.columns)
    assert set(out["Sesso"].unique()) == {"M", "F"}                       # mapping inverso
    assert set(out["Fascia"].unique()) <= set(GROUP_LABELS)               # etichette ordinali originali
    assert out["Eta"].dtype == pl.Int64 and out["Eta"].min() >= 25        # nessun valore z-scaled
    assert {"match_role", "match_pair_id", "match_weight", "match_distance", "match_group", "match_source", "source_row"} <= set(out.columns)
    assert (tmp_path / "matching_dataset.xlsx").exists() and r.matched_dataset.height == out.height
    # il valore in output coincide con quello del file di partenza
    raw = pl.read_csv(demo / "treated.csv").select("ID", "Eta", "Fascia")
    chk = out.filter(pl.col("match_source") == "treated").select("ID", "Eta", "Fascia").unique().join(raw, on="ID", suffix="_r")
    assert (chk["Eta"] == chk["Eta_r"]).all() and (chk["Fascia"] == chk["Fascia_r"]).all()


def test_raw_output_mode_keeps_original_rows(demo, tmp_path):
    run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], matching_cfg(),
                 output_dir=tmp_path, output_values="raw", output_formats=("csv",))
    out = pl.read_csv(tmp_path / "matching_dataset.csv")
    assert {"Sesso", "Fascia"} <= set(out.columns) and set(out["Sesso"].unique()) == {"M", "F"}


def test_ordinal_column_is_numeric_for_the_engine(demo, tmp_path):
    cfg = matching_cfg(matching=("Age", "Age_Group", "BMI"))
    cfg["variables"]["evaluation"] = ["Diabetes"]
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], cfg,
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.prepared.ordinal == {"Age_Group"} and r.prepared.kinds["Age_Group"] == "numerical"
    assert r.prepared.data["Age_Group"].dtype == pl.Int64


def test_single_dataset_with_group_column(demo, tmp_path):
    r = run_pipeline([demo / "all.csv"], [demo / "data_config.json"], matching_cfg(), group_column="Gruppo",
                     output_dir=tmp_path, output_formats=("csv",))
    assert set(r.matched_dataset["match_group"].unique()) == {0, 1}


def test_single_dataset_requires_group_column(demo, tmp_path):
    with pytest.raises(ConfigError):
        run_pipeline([demo / "all.csv"], [demo / "data_config.json"], matching_cfg(), output_dir=tmp_path)


def test_two_datasets_need_common_matching_variables(demo, tmp_path):
    c = json.loads((demo / "data_config.json").read_text())
    c["ALT"]["new_name"] = "ALT_other"
    (tmp_path / "other.json").write_text(json.dumps(c))
    with pytest.raises(ConfigError):
        run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json", tmp_path / "other.json"],
                     matching_cfg(), output_dir=tmp_path / "o")


def test_unmapped_label_stops_with_a_clear_message_and_can_be_tolerated(demo, tmp_path):
    c = json.loads((demo / "data_config.json").read_text())
    del c["Fascia"]["mappings"]["40-41"]
    (tmp_path / "bad.json").write_text(json.dumps(c))
    args = dict(output_dir=tmp_path / "o", output_formats=("csv",))
    with pytest.raises(UnmappedValuesError) as e:
        run_pipeline([demo / "treated.csv", demo / "controls.csv"], [tmp_path / "bad.json"], matching_cfg(), **args)
    assert "Age_Group" in str(e.value) and "'40-41'" in str(e.value)
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [tmp_path / "bad.json"], matching_cfg(), on_unmapped="missing", **args)
    assert r.outcome.metrics["n_query_matched"] > 0


def test_estimation_and_analysis_outputs(demo, tmp_path):
    cfg = matching_cfg("greedy_nn", {"replacement": True}, estimation=True)
    cfg["validation"]["analysis"] = {"correlation_method": "spearman"}
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json"], cfg,
                     output_dir=tmp_path, output_formats=("csv",))
    assert r.estimation is not None and r.estimation.table.height >= 4
    assert {"pca_variance", "pca_loadings", "correlation_pairs", "correlation_with_group"} <= set(r.analysis.tables)
    assert (tmp_path / "matching_effects.csv").exists()