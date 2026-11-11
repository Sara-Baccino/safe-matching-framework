"""Rigenera config di esempio, template completo e JSON Schema (usare dopo modifiche allo schema)."""
import json
from pathlib import Path
from safe_matching.schemas.template import write_config_template, write_json_schema

root = Path(__file__).parent.parent / "configs"
write_json_schema(root / "matching_config.schema.json")
write_config_template(root / "matching_full_template.json")

def base(method, params=None, metric="euclidean", matching=("Age", "BMI", "ALT"), est=True, estimands=("ATT",)):
    d = {
        "population": {"direction": "treated_to_control"},
        "variables": {"matching": list(matching), "stratification": ["Male"], "evaluation": ["Diabetes"]},
        "constraints": {"calipers": [{"variable": "Age", "value": 5, "scale": "absolute"}], "global_caliper": None},
        "metrics": {"name": metric, "params": {}},
        "method": {"name": method, "params": params or {}},
        "validation": {"smd_threshold": 0.1, "balance_metrics": ["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon"],
                       "success_criteria": {"min_match_rate": 0.8, "max_mean_abs_smd": 0.1, "max_abs_smd": 0.25, "min_pct_balanced": 90.0},
                       "analysis": {"pca": True, "distributions": True, "correlations": True, "ps_support": True},
                       "max_distribution_plots": 6, "plots": True, "hooks": []},
        "options": {"drop_rows_with_missing": True, "random_state": 42},
    }
    if est:
        d["estimation"] = {"enabled": True, "estimands": list(estimands),
                           "outcomes": [{"variable": "Outcome", "type": "continuous"}, {"variable": "Event", "type": "binary"}],
                           "bootstrap": {"n_boot": 200, "ci_level": 0.95, "resample": "full_pipeline", "n_jobs": 1}}
    return d

cfgs = {
    "matching_hungarian": base("hungarian", metric="mahalanobis"),
    "matching_propensity_score": base("propensity_score", {"algorithm": "greedy_nn", "caliper_sd": 0.2, "algorithm_params": {"ratio_k": 1, "order": "hardest_first"}}),
    "matching_ps_random_forest": base("propensity_score", {"model": "random_forest", "model_params": {"n_estimators": 500, "min_samples_leaf": 10}, "algorithm": "hungarian"}),
    "matching_ps_xgboost": base("propensity_score", {"model": "xgboost", "model_params": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}, "cross_fit": True, "algorithm": "greedy_nn", "algorithm_params": {"replacement": True}}, estimands=("ATT", "ATC", "ATE")),
    "matching_partial_ot": base("partial_ot", {"target_ratio": 2.0}, metric="manhattan", est=False),
    "matching_greedy_hybrid": base("greedy_nn", {"ratio_k": 2, "order": "random"}, metric="hybrid"),
}
s = base("sorted", {"sort_variable": "Age"}, matching=("Age",)); s["constraints"]["calipers"] = []
cfgs["matching_sorted"] = s
for name, d in cfgs.items():
    (root / f"{name}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
print("config scritti in", root)
