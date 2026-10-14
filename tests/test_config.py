from pathlib import Path

import pytest

from safe_matching.exceptions import ConfigError
from safe_matching.schemas import DataConfig, MatchingConfig

CFG = Path(__file__).parent.parent / "configs"


def test_sara_style_json_is_read():
    c = DataConfig.from_file(CFG / "example_data_config_lampertico.json")
    assert c.kinds()["Age"] == "numerical" and "clinical" in c.columns["Eta"].semantic_roles
    assert c.columns["Eta"].new_name == "Age"  # chiave = nome originale


def test_duplicate_active_new_name_rejected():
    col = {"new_name": "X", "active": True}
    with pytest.raises(ConfigError):
        DataConfig.from_dict({"a": col, "b": col})


def test_inactive_duplicate_new_name_allowed():
    DataConfig.from_dict({"a": {"new_name": "X"}, "b": {"new_name": "X", "active": False}})


def test_method_aliases_and_validation():
    c = MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "OT"}, "distance": {"name": "ibrida"}})
    assert c.method.name == "partial_ot" and c.metrics.name == "hybrid"
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "greedy_nn", "params": {"bogus": 1}}})
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "x"}})
