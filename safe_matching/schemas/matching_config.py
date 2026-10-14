"""Configurazione del matching (JSON): variabili, vincoli, metrica, metodo, validazione."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from safe_matching.exceptions import ConfigError
from safe_matching.schemas.method_params import (
    METHOD_PARAMS,
    PropensityScoreParams,
    normalize_method_name,
)

__all__ = [
    "MatchingConfig",
    "PopulationConfig",
    "VariablesConfig",
    "CaliperWindow",
    "GlobalCaliper",
    "ConstraintsConfig",
    "MetricConfig",
    "MethodConfig",
    "ValidationConfig",
    "HookSpec",
    "METRIC_NAMES",
]

METRIC_NAMES = ("euclidean", "manhattan", "mahalanobis", "hybrid", "gower")
_METRIC_ALIASES = {
    "euclidea": "euclidean",
    "l2": "euclidean",
    "cityblock": "manhattan",
    "l1": "manhattan",
    "ibrida": "hybrid",
    "weighted_hybrid": "hybrid",
    "mahalanobis_distance": "mahalanobis",
}


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PopulationConfig(_Cfg):
    direction: Literal["treated_to_control", "control_to_treated"] = "treated_to_control"
    """treated_to_control: per ogni trattato si cercano controlli (ATT); l'inverso per control_to_treated."""


class VariablesConfig(_Cfg):
    matching: list[str] = Field(min_length=1)
    stratification: list[str] = Field(default_factory=list)
    """Match esatto: il matching avviene solo dentro ciascuno strato."""
    evaluation: list[str] = Field(default_factory=list)
    """Variabili non usate per il matching ma incluse nelle diagnostiche di bilanciamento."""

    @model_validator(mode="after")
    def _no_duplicates(self) -> "VariablesConfig":
        for name in ("matching", "stratification", "evaluation"):
            vals = getattr(self, name)
            if len(set(vals)) != len(vals):
                raise ValueError(f"variables.{name} contiene duplicati.")
        overlap = set(self.matching) & set(self.evaluation)
        if overlap:
            raise ValueError(f"Variabili sia in matching sia in evaluation: {sorted(overlap)}")
        return self


class CaliperWindow(_Cfg):
    """Finestra di tolleranza hard su una variabile: |x_q - x_p| <= value."""

    variable: str
    value: float = Field(gt=0)
    scale: Literal["absolute", "standard_deviation"] = "absolute"


class GlobalCaliper(_Cfg):
    """Distanza massima ammessa per una coppia (sulla metrica scelta)."""

    value: float = Field(gt=0)
    scale: Literal["absolute", "standard_deviation"] = "absolute"
    """standard_deviation: value x SD della feature di matching (1-D) o delle distanze (multi-D)."""


class ConstraintsConfig(_Cfg):
    calipers: list[CaliperWindow] = Field(default_factory=list)
    global_caliper: GlobalCaliper | None = None


class MetricConfig(_Cfg):
    name: str = "euclidean"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "MetricConfig":
        key = self.name.strip().lower()
        self.name = _METRIC_ALIASES.get(key, key)
        if self.name not in METRIC_NAMES:
            raise ValueError(f"metrics.name '{self.name}' non supportato. Validi: {list(METRIC_NAMES)}")
        return self


class MethodConfig(_Cfg):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "MethodConfig":
        self.name = normalize_method_name(self.name)
        model = METHOD_PARAMS.get(self.name)
        if model is None:
            raise ValueError(f"method.name '{self.name}' non supportato. Validi: {sorted(METHOD_PARAMS)}")
        parsed = model.model_validate(self.params)
        self.params = parsed.model_dump()
        return self

    def parsed(self):
        return METHOD_PARAMS[self.name].model_validate(self.params)


class HookSpec(_Cfg):
    """Validazione esterna: callable 'pacchetto.modulo:funzione' (es. un evaluator SAFE)."""

    name: str
    callable: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ValidationConfig(_Cfg):
    smd_threshold: float = Field(default=0.1, gt=0)
    balance_metrics: list[
        Literal["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon", "wasserstein"]
    ] = Field(default_factory=lambda: ["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon"])
    plots: bool = True
    max_distribution_plots: int = Field(default=6, ge=0, le=24)
    hooks: list[HookSpec] = Field(default_factory=list)


class OptionsConfig(_Cfg):
    drop_rows_with_missing: bool = True
    """Esclude dal matching le righe con valori mancanti nelle variabili usate (e le riporta)."""
    random_state: int | None = 42


class MatchingConfig(_Cfg):
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    variables: VariablesConfig
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    metrics: MetricConfig = Field(default_factory=MetricConfig)
    method: MethodConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "distance" in data and "metrics" not in data:
                data["metrics"] = data.pop("distance")
            if "metric" in data and "metrics" not in data:
                data["metrics"] = data.pop("metric")
        return data

    @model_validator(mode="after")
    def _cross_checks(self) -> "MatchingConfig":
        cal_vars = [c.variable for c in self.constraints.calipers]
        if len(set(cal_vars)) != len(cal_vars):
            raise ValueError("constraints.calipers: variabile ripetuta.")
        if self.method.name == "propensity_score":
            ps: PropensityScoreParams = self.method.parsed()  # type: ignore[assignment]
            inner = METHOD_PARAMS.get(ps.algorithm)
            if inner is not None:
                inner.model_validate(ps.algorithm_params)
        return self

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchingConfig":
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"Configurazione matching non valida:\n{exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "MatchingConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"File di configurazione matching non trovato: {p}")
        try:
            parsed = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON non valido in '{p}': {exc}") from exc
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
