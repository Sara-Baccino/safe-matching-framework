from pathlib import Path

import pytest

from safe_matching import run_pipeline
from safe_matching.exceptions import ConfigError

ROOT = Path(__file__).parent.parent
DATA, CFG = ROOT / "examples" / "data", ROOT / "configs"


@pytest.fixture(scope="module", autouse=True)
def demo_data():
    if not (DATA / "all.csv").exists():
        import runpy
        runpy.run_path(str(ROOT / "examples" / "make_demo_data.py"))


@pytest.mark.parametrize("m", ["hungarian", "propensity_score", "partial_ot", "sorted", "greedy_hybrid"])
def test_two_datasets_all_methods(tmp_path, m):
    r = run_pipeline([DATA / "treated.csv", DATA / "controls.csv"], [CFG / "demo_data_config.json"],
                     CFG / f"matching_{m}.json", output_dir=tmp_path)
    assert r.report_path.stat().st_size > 10_000
    assert r.balance.summary["mean_abs_smd_after"] <= r.balance.summary["mean_abs_smd_before"]


def test_single_dataset_with_group_column(tmp_path):
    r = run_pipeline([DATA / "all.csv"], [CFG / "demo_data_config.json"], CFG / "matching_hungarian.json",
                     group_col="Gruppo", output_dir=tmp_path)
    assert (tmp_path / "matched_dataset.csv").exists() and r.outcome.metrics["match_rate"] == 1.0


def test_single_dataset_requires_group_col(tmp_path):
    with pytest.raises(ConfigError):
        run_pipeline([DATA / "all.csv"], [CFG / "demo_data_config.json"], CFG / "matching_hungarian.json", output_dir=tmp_path)


def test_two_datasets_need_common_matching_variables(tmp_path):
    import json
    c = json.loads((CFG / "demo_data_config.json").read_text())
    c["ALT"]["new_name"] = "ALT_other"
    (tmp_path / "other.json").write_text(json.dumps(c))
    with pytest.raises(ConfigError):
        run_pipeline([DATA / "treated.csv", DATA / "controls.csv"], [CFG / "demo_data_config.json", tmp_path / "other.json"],
                     CFG / "matching_hungarian.json", output_dir=tmp_path / "o")
