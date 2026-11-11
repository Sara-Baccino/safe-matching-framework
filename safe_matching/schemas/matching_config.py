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
    "SuccessCriteria",
    "EstimationConfig",
    "OutcomeSpec",
    "BootstrapConfig",
    "AnalysisConfig",
    "HookSpec",
    "METRIC_NAMES",
    "MissingPolicy",
    "HarmonizationPolicy"
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


class SuccessCriteria(_Cfg):
    """Soglie per giudicare riuscito il matching (None = criterio disattivato)."""

    min_match_rate: float | None = Field(default=0.8, ge=0, le=1)
    max_mean_abs_smd: float | None = Field(default=0.1, gt=0)
    max_abs_smd: float | None = Field(default=0.25, gt=0)
    """0.25 e' la soglia 'accettabile' di Stuart & Rubin per il singolo SMD."""
    min_pct_balanced: float | None = Field(default=90.0, ge=0, le=100)
    require_improvement: bool = True
    max_pca_smd: float | None = Field(default=None, gt=0)
    """Opzionale: max |SMD| sulle componenti principali trattenute (richiede analysis.pca)."""
    max_corr_diff: float | None = Field(default=None, gt=0)
    """Opzionale: massimo scarto medio |corr_trattati - corr_controlli| dopo il matching (richiede analysis.correlations)."""


class AnalysisConfig(_Cfg):
    """Analisi multivariate mostrate nel report (calcolate su matching + stratificazione + evaluation)."""

    pca: bool = True
    pca_variance_target: float = Field(default=0.8, gt=0, le=1)
    """Quota di varianza spiegata che determina quante componenti entrano nel riepilogo SMD delle PC."""
    pca_loadings_components: int = Field(default=3, ge=1, le=6)
    distributions: bool = True
    correlations: bool = True
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    max_correlation_variables: int = Field(default=25, ge=3, le=60)
    max_scatter_points: int = Field(default=3000, ge=200)
    ps_support: bool = True
    """Diagnostica di supporto comune/overlap del propensity score. Con metodi diversi da propensity_score il PS e'
    stimato (logistica) solo per la diagnostica e non influenza il matching."""
    ps_support_bins: int = Field(default=30, ge=10, le=100)


class OutcomeSpec(_Cfg):
    variable: str
    """new_name dell'outcome nella data config. NON deve essere una variabile di matching/stratificazione."""
    type: Literal["continuous", "binary"] = "continuous"
    """continuous: differenza di medie. binary (0/1): differenza di rischio e risk ratio."""
    label: str | None = None


class BootstrapConfig(_Cfg):
    n_boot: int = Field(default=200, ge=20, le=50000)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1)
    resample: Literal["full_pipeline", "matched_pairs"] = "full_pipeline"
    """full_pipeline: ricampiona le unita' (dentro ciascun gruppo) e RIESEGUE stima PS + matching a ogni replica
    (cattura la variabilita' del matching; lento). matched_pairs: ricampiona le coppie gia' formate (veloce; solo
    metodi a coppie; ignora la variabilita' del matching)."""
    random_state: int | None = None
    """None = usa options.random_state."""
    n_jobs: int = Field(default=1, ge=-1)
    """Repliche in parallelo (joblib); -1 = tutti i core. Su Windows main.py deve usare `if __name__ == "__main__"`."""


class EstimationConfig(_Cfg):
    """Stima di ATT / ATC / ATE sugli outcome indicati, con errori standard e IC da bootstrap."""

    enabled: bool = False
    estimands: list[Literal["ATT", "ATC", "ATE"]] = Field(default_factory=lambda: ["ATT"])
    """ATT: effetto sui trattati (matching treated_to_control). ATC: sui controlli (control_to_treated).
    ATE: combinazione di ATT e ATC pesata per le unita' abbinate (richiede entrambi i matching)."""
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    include_naive: bool = True
    """Aggiunge la differenza non aggiustata (popolazione intera) come riferimento."""
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @model_validator(mode="after")
    def _need_outcomes(self) -> "EstimationConfig":
        if self.enabled and not self.outcomes:
            raise ValueError("estimation.enabled=true richiede almeno un outcome in estimation.outcomes.")
        if len(set(self.estimands)) != len(self.estimands) or not self.estimands:
            raise ValueError("estimation.estimands: elenco non vuoto e senza duplicati.")
        return self


class ValidationConfig(_Cfg):
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    smd_threshold: float = Field(default=0.1, gt=0)
    balance_metrics: list[
        Literal["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon", "wasserstein"]
    ] = Field(default_factory=lambda: ["smd", "variance_ratio", "ks", "chi_square", "jensen_shannon"])
    plots: bool = True
    max_distribution_plots: int = Field(default=12, ge=0, le=60)
    """Numero di variabili (le piu' sbilanciate prima del matching) mostrate nelle griglie di distribuzioni."""
    hooks: list[HookSpec] = Field(default_factory=list)

class MissingPolicy(_Cfg):
    """Gestione dei valori mancanti nelle variabili usate (matching, stratificazione, evaluation).
    Gli outcome non si imputano MAI e non entrano nei modelli di imputazione."""

    impute: bool = True
    """False = nessuna imputazione: le righe con mancanti nelle variabili di matching sono escluse."""
    continuous_imputer: Literal["knn", "median", "mean"] = "knn"
    categorical_imputer: Literal["mode"] = "mode"
    """Nominali (categorical): valore piu' frequente."""
    ordinal_imputer: Literal["median", "mode"] = "median"
    """Ordinali (categorical + numerical): livello mediano (esiste sempre tra i livelli osservati) o moda."""
    knn_neighbors: int = Field(default=5, ge=1, le=50)
    include_group_in_imputation: bool = True
    """Con un solo dataset il gruppo entra come feature del knn, cosi' l'imputazione non cancella lo sbilanciamento."""
    use_column_imputer: bool = False
    """True = usa 'imputer' della data config per colonna (zero/mean/median/mode/knn) al posto di questi default."""
    low_threshold: float = Field(default=0.05, ge=0, le=1)
    """Fino a questa quota di mancanti: imputazione 'bassa' (nessun avviso)."""
    max_fraction: float = Field(default=0.30, gt=0, le=1)
    """Oltre questa quota la variabile e' 'alta': si applica over_threshold_action. Tra low e max: imputata con avviso."""
    over_threshold_action: Literal["drop_variable", "error", "impute"] = "drop_variable"
    """drop_variable: la variabile esce dal matching/valutazione (con nota); error: si ferma; impute: imputa comunque."""
    row_max_fraction: float = Field(default=0.5, ge=0, le=1)
    """Righe con piu' di questa quota di variabili mancanti (e almeno 2) NON vengono imputate: restano escluse."""
    add_flags: bool = True
    """Colonne imputed_<variabile> (0/1) nell'output, per le analisi di sensibilita'."""

class HarmonizationPolicy(_Cfg):
    """Controlli e armonizzazione tra i DUE dataset prima del matching (con un solo dataset non si applica)."""

    enabled: bool = True
    unify_codes: bool = True
    """Variabili nominali con codici diversi nei due config (stessa etichetta, codice diverso; o codice uguale per etichette
    diverse): il secondo dataset viene ricodificato su un mapping unificato. False = errore. Le ordinali non si unificano mai."""
    check_numeric: bool = True
    overlap_warn_below: float = Field(default=0.8, ge=0, le=1)
    """Avviso se meno di questa quota dei valori di un dataset cade nel range [min, max] dell'altro."""
    unit_check: bool = True
    """Segnala possibili unita' di misura diverse (rapporto tra mediane vicino a un fattore di conversione noto + SMD grezzo > 1)."""
    error_on_unit_mismatch: bool = False
    sentinel_values: list[float] = Field(default_factory=lambda: [-9, -99, -999, -9999, 99, 999, 9999])
    """Valori speciali (es. -99 = non misurato) da cercare se lontani dalla distribuzione."""

class OptionsConfig(_Cfg):
    harmonization: HarmonizationPolicy = Field(default_factory=HarmonizationPolicy)
    missing: MissingPolicy = Field(default_factory=MissingPolicy)
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
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # chiavi di commento ("_note", "_method_options", ...) e "$schema" sono ignorate
            data = {k: v for k, v in data.items() if not (str(k).startswith("_") or str(k).startswith("$"))}
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
        bad = [o.variable for o in self.estimation.outcomes
               if o.variable in self.variables.matching or o.variable in self.variables.stratification]
        if bad:
            raise ValueError(f"Gli outcome non possono essere variabili di matching/stratificazione: {bad}")
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
