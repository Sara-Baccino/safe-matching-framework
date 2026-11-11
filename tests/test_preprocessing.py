import json

import polars as pl
import pytest

from safe_matching.exceptions import PreprocessingError
from safe_matching.preprocessing.config import DataConfig
from safe_matching.preprocessing.reader import read_table
from safe_matching.preprocessing.transform import CastError, UnmappedValuesError, apply_config, infer_types, inverse_transform

LABELS = {"<35": 1, "35-37": 2, "38-39": 3, "40-41": 4, "42-45": 5}


def cfg(**cols):
    base = {"new_name": "X", "active": True, "categorical": False, "numerical": False, "mappings": {}, "type": None,
            "missing_data_management": {"strategy": "impute", "imputer": "knn", "missing_values": [], "missing_replace_value": None}}
    return DataConfig.from_dict({k: {**base, "new_name": k, **v} for k, v in cols.items()})


def test_legacy_and_current_formats_both_parse():
    c = DataConfig.from_dict({
        "A": {"new_name": "A", "categorical": True, "categorical_mappings": {"x": 1}, "missing_data_management": {"condition": ["?"], "value": 0}},
        "B": {"new_name": "B", "categorical": True, "mappings": {"y": 2}, "gene": False, "clinical": True, "id": False, "unknown_field": 1}})
    assert c.columns["A"].mappings == {"x": 1} and c.columns["A"].missing_data_management.missing_values == ["?"]
    assert c.columns["B"].mappings == {"y": 2}


def test_kind_ordinal_is_categorical_plus_numerical():
    c = cfg(G={"categorical": True, "numerical": True}, S={"categorical": True}, N={"numerical": True})
    assert [c.columns[k].kind for k in "GSN"] == ["ordinal", "categorical", "numerical"]


def test_mapping_ordinal_codes_and_inverse_roundtrip():
    raw = pl.DataFrame({"g": ["40-41", "<35", " 35-37 ", None, "42-45"]})
    c = cfg(g={"new_name": "Age_Group", "categorical": True, "numerical": True, "type": "str", "mappings": LABELS})
    r = apply_config(raw, c)
    assert r.data["Age_Group"].to_list() == [4, 1, 2, None, 5] and r.data["Age_Group"].dtype == pl.Int64
    back = inverse_transform(r.data, c, r)
    assert back["g"].to_list() == ["40-41", "<35", "35-37", None, "42-45"]
    assert r.order["Age_Group"][0] == (1.0, "<35")


def test_unmapped_values_are_reported_together_with_counts():
    raw = pl.DataFrame({"g": ["40-41", "50+", "50+", "??"], "s": ["M", "F", "X", "M"]})
    c = cfg(g={"categorical": True, "type": "str", "mappings": LABELS}, s={"categorical": True, "mappings": {"M": 1, "F": 0}})
    with pytest.raises(UnmappedValuesError) as e:
        apply_config(raw, c)
    msg = str(e.value)
    assert "'50+': 2" in msg and "'??': 1" in msg and "'X': 1" in msg      # entrambe le colonne, con conteggi
    r = apply_config(raw, c, on_unmapped="missing")
    assert r.data["g"].null_count() == 3 and r.warnings


def test_mapping_is_case_and_space_insensitive_and_accepts_codes():
    raw = pl.DataFrame({"s": ["m", " F", "M", "1", "0"]})
    c = cfg(s={"categorical": True, "mappings": {"M": 1, "F": 0}})
    r = apply_config(raw, c)
    assert r.data["s"].to_list() == [1, 0, 1, 1, 0]
    assert r.qc[0].n_code_passthrough == 2
    with pytest.raises(UnmappedValuesError):
        apply_config(raw, c, accept_codes=False)


def test_numeric_cast_decimal_comma_tokens_and_multiplier():
    raw = pl.DataFrame({"a": ["12,5", "3.0", "NA", "", "n/a", "7"], "i": ["1", "2.0", " 3 ", None, "NaN", "5"]})
    c = cfg(a={"numerical": True, "type": "float", "multiplier": 2}, i={"numerical": True, "type": "int"})
    r = apply_config(raw, c)
    assert r.data["a"].to_list() == [25.0, 6.0, None, None, None, 14.0]
    assert r.data["i"].to_list() == [1, 2, 3, None, None, 5] and r.data["i"].dtype == pl.Int64


def test_cast_errors_are_reported_and_int_rejects_fractions():
    raw = pl.DataFrame({"i": ["1", "abc", "2.5"]})
    c = cfg(i={"numerical": True, "type": "int"})
    with pytest.raises(CastError) as e:
        apply_config(raw, c)
    assert "abc" in str(e.value) and "2.5" in str(e.value)
    assert apply_config(raw, c, on_cast_error="missing").data["i"].null_count() == 2


def test_extra_missing_tokens_from_config():
    raw = pl.DataFrame({"a": ["1", "-99", "3"]})
    c = cfg(a={"numerical": True, "type": "int", "missing_data_management": {"strategy": "impute", "imputer": "knn", "missing_values": ["-99"]}})
    assert apply_config(raw, c).data["a"].to_list() == [1, None, 3]


def test_scale_flag_is_ignored_and_data_not_scaled():
    raw = pl.DataFrame({"a": ["10", "20", "30"]})
    r = apply_config(raw, cfg(a={"numerical": True, "type": "int", "scale": True}))
    assert r.data["a"].to_list() == [10, 20, 30] and any("scale" in w for w in r.warnings)


def test_missing_active_column_is_explained():
    with pytest.raises(Exception) as e:
        apply_config(pl.DataFrame({"a": ["1"]}), cfg(a={"numerical": True}, zzz={"numerical": True}))
    assert "zzz" in str(e.value)
    assert apply_config(pl.DataFrame({"a": ["1"]}), cfg(a={"numerical": True}, zzz={"numerical": True}), strict_columns=False).warnings


def test_ordinal_without_mapping_requires_numeric_values():
    c = cfg(g={"categorical": True, "numerical": True})
    assert apply_config(pl.DataFrame({"g": ["5", "6", "7"]}), c).data["g"].to_list() == [5, 6, 7]
    with pytest.raises(PreprocessingError):
        apply_config(pl.DataFrame({"g": ["a", "b"]}), c)


def test_reader_csv_semicolon_cp1252_and_excel_mixed_types(tmp_path):
    (tmp_path / "a.csv").write_bytes("Età;Nota\n35;è ok\n41;\n".encode("cp1252"))
    df = read_table(tmp_path / "a.csv")
    assert df.columns == ["Età", "Nota"] and df["Età"].to_list() == ["35", "41"] and df.schema["Età"] == pl.Utf8
    import xlsxwriter
    wb = xlsxwriter.Workbook(str(tmp_path / "m.xlsx")); ws = wb.add_worksheet()
    ws.write_row(0, 0, [" Age ", "Mixed"]); ws.write_row(1, 0, [41.0, 1]); ws.write_row(2, 0, [35, "x"]); wb.close()
    x = read_table(tmp_path / "m.xlsx")
    assert x.columns == ["Age", "Mixed"] and x["Age"].to_list() == ["41", "35"] and x["Mixed"].to_list() == ["1", "x"]


def test_infer_types_only_converts_fully_numeric_text():
    out = infer_types(pl.DataFrame({"n": ["1", "2", ""], "f": ["1.5", "2", None], "t": ["a", "1", "2"]}))
    assert out["n"].dtype == pl.Int64 and out["f"].dtype == pl.Float64 and out["t"].dtype == pl.Utf8