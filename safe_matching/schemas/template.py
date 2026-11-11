"""Generazione del config di matching COMPLETO (tutti i parametri con i default) e del JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safe_matching.schemas.matching_config import MatchingConfig
from safe_matching.schemas.method_params import METHOD_PARAMS

__all__ = ["full_template", "write_config_template", "write_json_schema"]

_METRIC_OPTIONS = {
    "euclidean": {"scale": "standard | minmax | none", "weights": {"<variabile>": 1.0}},
    "manhattan": {"scale": "standard | minmax | none", "weights": {"<variabile>": 1.0}},
    "mahalanobis": {"covariance": "total | within", "regularization": 1e-8},
    "hybrid": {"weight_numerical": 0.6, "weight_categorical": 0.4, "weights": {"<variabile>": 1.0}},
    "gower": {"weights": {"<variabile>": 1.0}},
}


def full_template(method: str = "propensity_score") -> dict[str, Any]:
    """Config con tutte le sezioni e tutti i parametri del metodo scelto; le opzioni degli altri metodi
    e delle metriche sono elencate nelle chiavi di commento ``_method_options`` / ``_metric_options``."""
    base = {
        "variables": {"matching": ["Age", "BMI", "ALT"], "stratification": ["Male"], "evaluation": ["Diabetes"]},
        "constraints": {"calipers": [{"variable": "Age", "value": 5, "scale": "absolute"}], "global_caliper": None},
        "method": {"name": method, "params": {}},
        "estimation": {"enabled": True, "outcomes": [{"variable": "Outcome", "type": "continuous", "label": None},
                                                      {"variable": "Event", "type": "binary", "label": None}]},
    }
    cfg = MatchingConfig.from_dict(base)
    d = cfg.to_dict()
    d["estimation"]["estimands"] = ["ATT"]
    out: dict[str, Any] = {
        "$schema": "./matching_config.schema.json",
        "_note": "Config completo: ogni parametro e' mostrato col suo default. Le chiavi che iniziano con '_' sono commenti e vengono ignorate.",
        **d,
        "_method_options": {name: model().model_dump() for name, model in METHOD_PARAMS.items()},
        "_metric_options": _METRIC_OPTIONS,
        "_enum_values": {
            "population.direction": ["treated_to_control", "control_to_treated"],
            "constraints.calipers[].scale": ["absolute", "standard_deviation"],
            "metrics.name": ["euclidean", "manhattan", "mahalanobis", "hybrid", "gower"],
            "method.name": sorted(METHOD_PARAMS),
            "method.params.model (propensity_score)": ["logistic", "random_forest", "xgboost", "gradient_boosting"],
            "method.params.algorithm (propensity_score)": ["greedy_nn", "hungarian", "sorted", "partial_ot"],
            "method.params.trim (propensity_score)": ["none", "minmax", "quantile"],
            "validation.balance_metrics": ["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon", "wasserstein"],
            "validation.analysis.correlation_method": ["pearson", "spearman"],
            "estimation.estimands": ["ATT", "ATC", "ATE"],
            "estimation.outcomes[].type": ["continuous", "binary"],
            "estimation.bootstrap.resample": ["full_pipeline", "matched_pairs"],
        },
    }
    return out


def write_config_template(path: str | Path, method: str = "propensity_score") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(full_template(method), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_json_schema(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(MatchingConfig.model_json_schema(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
