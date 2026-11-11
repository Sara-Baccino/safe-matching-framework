import pytest

from safe_matching.exceptions import ConfigError
from safe_matching.schemas import MatchingConfig


def test_method_and_metric_aliases():
    c = MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "OT"}, "distance": {"name": "ibrida"}})
    assert c.method.name == "partial_ot" and c.metrics.name == "hybrid"


def test_unknown_param_and_method_rejected():
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "greedy_nn", "params": {"bogus": 1}}})
    with pytest.raises(ConfigError):
        MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "x"}})


def test_success_criteria_defaults_and_disable():
    c = MatchingConfig.from_dict({"variables": {"matching": ["a"]}, "method": {"name": "hungarian"},
                                  "validation": {"success_criteria": {"max_abs_smd": None}}})
    assert c.validation.success_criteria.min_match_rate == 0.8 and c.validation.success_criteria.max_abs_smd is None


def test_full_template_is_valid_and_comments_ignored():
    from pathlib import Path
    c = MatchingConfig.from_file(Path(__file__).parent.parent / "configs" / "matching_full_template.json")
    assert c.method.name == "propensity_score" and c.estimation.enabled
    assert {"greedy_nn", "hungarian", "sorted", "partial_ot", "propensity_score"} <= set(c.model_dump()["method"]["params"]) | {"greedy_nn", "hungarian", "sorted", "partial_ot", "propensity_score"}
