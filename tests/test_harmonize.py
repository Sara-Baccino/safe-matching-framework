import json

import numpy as np
import polars as pl
import pytest

from safe_matching import run_pipeline
from safe_matching.preprocessing.config import DataConfig
from safe_matching.preprocessing.harmonize import HarmonizationError, harmonize_sources
from safe_matching.preprocessing.source import load_source
from safe_matching.preprocessing.transform import UnmappedValuesError
from safe_matching.schemas.matching_config import HarmonizationPolicy

from conftest import matching_cfg


def col(new, *, cat=False, num=False, typ=None, maps=None):
    return {"new_name": new, "active": True, "categorical": cat, "numerical": num, "mappings": maps or {}, "type": typ,
            "missing_data_management": {"strategy": "maintain", "imputer": "knn", "missing_values": [], "missing_replace_value": None}}


def src(df, name, **cols):
    return load_source(df, DataConfig.from_dict({k: v for k, v in cols.items()}), name)


def run(a, b, variables, **pol):
    return harmonize_sources([a, b], variables=variables, policy=HarmonizationPolicy(**pol))


def test_conflicting_codes_are_unified_and_labels_stay_those_of_each_file():
    a = src(pl.DataFrame({"s": ["M", "F", "M"]}), "A", s=col("Sex", cat=True, maps={"M": 1, "F": 0}))
    b = src(pl.DataFrame({"s": ["M", "F", "F"]}), "B", s=col("Sex", cat=True, maps={"M": 0, "F": 1}))
    assert b.pre["Sex"].to_list() == [0, 1, 1]
    _, rep = run(a, b, ["Sex"])
    assert b.pre["Sex"].to_list() == [1, 0, 0]                                  # M=1 come in A
    assert b.to_original(b.pre)["s"].to_list() == ["M", "F", "F"]               # etichette originali
    assert b.config.columns["s"].mappings == {"M": 1, "F": 0} and any("ricodificati" in x for x in rep.actions)


def test_code_collision_with_a_label_only_in_b_gets_a_new_code():
    a = src(pl.DataFrame({"g": ["a", "b"]}), "A", g=col("G", cat=True, maps={"a": 1, "b": 2}))
    b = src(pl.DataFrame({"g": ["c", "b", "c"]}), "B", g=col("G", cat=True, maps={"c": 1, "b": 2}))
    run(a, b, ["G"])
    assert b.pre["G"].to_list() == [3, 2, 3] and b.config.columns["g"].mappings == {"c": 3, "b": 2}
    assert b.to_original(b.pre)["g"].to_list() == ["c", "b", "c"]


def test_unify_can_be_disabled_and_ordinals_are_never_unified():
    a = src(pl.DataFrame({"s": ["M"]}), "A", s=col("Sex", cat=True, maps={"M": 1, "F": 0}))
    b = src(pl.DataFrame({"s": ["M"]}), "B", s=col("Sex", cat=True, maps={"M": 0, "F": 1}))
    with pytest.raises(HarmonizationError):
        run(a, b, ["Sex"], unify_codes=False)
    oa = src(pl.DataFrame({"g": ["lo", "hi"]}), "A", g=col("G", cat=True, num=True, maps={"lo": 1, "hi": 2}))
    ob = src(pl.DataFrame({"g": ["lo", "hi"]}), "B", g=col("G", cat=True, num=True, maps={"lo": 2, "hi": 1}))
    with pytest.raises(HarmonizationError) as e:
        run(oa, ob, ["G"])
    assert "ordinale" in str(e.value)
    oc = src(pl.DataFrame({"g": ["lo", "hi", "mid"]}), "B", g=col("G", cat=True, num=True, maps={"lo": 1, "hi": 2, "mid": 3}))
    run(oa, oc, ["G"])                                                          # codici uguali per le etichette comuni: ok
    bad = src(pl.DataFrame({"g": ["lo", "hi", "mid"]}), "B", g=col("G", cat=True, num=True, maps={"lo": 1, "hi": 3, "mid": 2}))
    with pytest.raises(HarmonizationError):
        run(oa, bad, ["G"])                                                     # hi=3 vs hi=2 (e mid collide con hi): errore


def test_one_side_without_mappings_is_accepted_only_if_values_are_codes():
    a = src(pl.DataFrame({"s": ["M", "F"]}), "A", s=col("Sex", cat=True, maps={"M": 1, "F": 0}))
    ok = src(pl.DataFrame({"s": ["1", "0", "1"]}), "B", s=col("Sex", cat=True, typ="int"))
    _, rep = run(a, ok, ["Sex"])
    assert any("gia' i valori espressi come codici" in x for x in rep.actions)
    bad = src(pl.DataFrame({"s": ["1", "7"]}), "B", s=col("Sex", cat=True, typ="int"))
    with pytest.raises(HarmonizationError):
        run(a, bad, ["Sex"])


def test_union_adds_untracked_nominal_labels_and_refuses_ordinals():
    cfg = DataConfig.from_dict({"s": col("Sex", cat=True, maps={"M": 1})})
    df = pl.DataFrame({"s": ["M", "F", "X", "F"]})
    with pytest.raises(UnmappedValuesError):
        load_source(df, cfg, "B")
    b = load_source(df, cfg, "B", on_unmapped="union")
    assert b.mapping_additions == {"Sex": {"F": 2, "X": 3}} and b.pre["Sex"].to_list() == [1, 2, 3, 2]
    assert "F" not in cfg.columns["s"].mappings                                  # la config dell'utente non e' modificata
    assert b.to_original(b.pre)["s"].to_list() == ["M", "F", "X", "F"]
    ocfg = DataConfig.from_dict({"g": col("G", cat=True, num=True, maps={"lo": 1})})
    with pytest.raises(UnmappedValuesError) as e:
        load_source(pl.DataFrame({"g": ["lo", "hi"]}), ocfg, "B", on_unmapped="union")
    assert "ordinale" in str(e.value)


def test_categorical_table_shows_levels_present_in_one_dataset_only():
    a = src(pl.DataFrame({"s": ["x", "y", "x", "y"]}), "A", s=col("S", cat=True, maps={"x": 1, "y": 2}))
    b = src(pl.DataFrame({"s": ["x", "z", "x"]}), "B", s=col("S", cat=True, maps={"x": 1, "z": 3}))
    _, rep = run(a, b, ["S"])
    t = rep.categorical.sort("code")
    assert t["only_in"].to_list() == ["", "A", "B"] and t["label"].to_list() == ["x", "y", "z"]


def numeric_pair(a_vals, b_vals):
    return (src(pl.DataFrame({"v": [str(x) for x in a_vals]}), "A", v=col("V", num=True, typ="float")),
            src(pl.DataFrame({"v": [str(x) for x in b_vals]}), "B", v=col("V", num=True, typ="float")))


def test_numeric_check_flags_units_sentinels_and_low_overlap_but_not_similar_data():
    rng = np.random.default_rng(0)
    mgdl = rng.normal(100, 15, 400).round(1)
    mmol = (mgdl / 18.016 + rng.normal(0, 0.3, 400)).round(2)
    _, rep = run(*numeric_pair(mgdl, mmol), ["V"])
    r = rep.numeric.row(0, named=True)
    assert "unit mismatch" in r["flags"] and "glucose" in r["flags"] and r["severity"] == "warn"
    with pytest.raises(HarmonizationError):
        run(*numeric_pair(mgdl, mmol), ["V"], error_on_unit_mismatch=True)

    base = rng.normal(50, 5, 300).round(1)
    withs = np.concatenate([base[:290], np.full(10, -99.0)])
    _, rep = run(*numeric_pair(base, withs), ["V"])
    assert "sentinel-like value -99 in B (n=10)" in rep.numeric["flags"][0]

    _, rep = run(*numeric_pair(rng.normal(20, 1, 300), rng.normal(60, 1, 300)), ["V"])
    assert "low range overlap" in rep.numeric["flags"][0]

    _, rep = run(*numeric_pair(rng.normal(50, 5, 300), rng.normal(50.5, 5, 300)), ["V"])
    assert rep.numeric["severity"][0] == "ok" and rep.numeric["flags"][0] == ""
    assert run(*numeric_pair(mgdl, mmol), ["V"], check_numeric=False)[1].numeric.height == 0


# ------------------------------------------------------------------------------------------ pipeline (richiedono C2)
def test_pipeline_unifies_codes_between_two_configs_and_writes_traceability_files(demo, tmp_path):
    cb = json.loads((demo / "data_config.json").read_text())
    cb["Sesso"]["mappings"] = {"M": 0, "F": 1}                                   # B codifica il sesso al contrario
    (tmp_path / "cfg_b.json").write_text(json.dumps(cb))
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [demo / "data_config.json", tmp_path / "cfg_b.json"], matching_cfg(),
                     output_dir=tmp_path / "o", output_formats=("csv",))
    assert any("ricodificati" in a for a in r.harmonization.actions)
    out = pl.read_csv(tmp_path / "o" / "matching_dataset.csv")
    raw_c = pl.read_csv(demo / "controls.csv").select("ID", "Sesso")
    chk = out.filter(pl.col("match_source") == "controls").select("ID", "Sesso").unique().join(raw_c, on="ID", suffix="_r")
    assert (chk["Sesso"] == chk["Sesso_r"]).all()                                # etichette originali di B intatte
    # lo strato "Male" confronta davvero lo stesso sesso: tutte le coppie hanno la stessa etichetta
    pairs = out.filter(pl.col("match_pair_id").is_not_null()).group_by("match_pair_id").agg(pl.col("Sesso").n_unique().alias("n"))
    assert pairs["n"].max() == 1
    for f in ("matching_harmonization_numeric.csv", "matching_harmonization_categorical.csv", "matching_data_config_treated.json",
              "matching_data_config_controls.json"):
        assert (tmp_path / "o" / f).exists()
    eff = json.loads((tmp_path / "o" / "matching_data_config_controls.json").read_text())
    assert eff["columns"]["Sesso"]["mappings"] == {"M": 1, "F": 0}


def test_pipeline_union_extends_mappings_for_a_shared_config(demo, tmp_path):
    c = json.loads((demo / "data_config.json").read_text())
    c["Sesso"]["mappings"] = {"M": 1}                                            # 'F' non tracciata
    (tmp_path / "cfg.json").write_text(json.dumps(c))
    with pytest.raises(UnmappedValuesError):
        run_pipeline([demo / "treated.csv", demo / "controls.csv"], [tmp_path / "cfg.json"], matching_cfg(), output_dir=tmp_path / "x")
    r = run_pipeline([demo / "treated.csv", demo / "controls.csv"], [tmp_path / "cfg.json"], matching_cfg(), on_unmapped="union",
                     output_dir=tmp_path / "o", output_formats=("csv",))
    assert r.outcome.metrics["n_query_matched"] > 0
    assert any("union" in a for a in r.harmonization.actions)
    assert set(pl.read_csv(tmp_path / "o" / "matching_dataset.csv")["Sesso"].unique()) == {"M", "F"}