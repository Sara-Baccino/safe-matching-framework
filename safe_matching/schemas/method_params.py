"""Parametri tipizzati dei metodi di matching (validati al caricamento del JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "GreedyNNParams",
    "HungarianParams",
    "SortedParams",
    "PartialOTParams",
    "PropensityScoreParams",
    "METHOD_PARAMS",
    "METHOD_ALIASES",
    "ALGORITHM_NAMES",
    "normalize_method_name",
]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GreedyNNParams(_Params):
    """Nearest neighbour greedy 1:K (con o senza reinserimento)."""

    ratio_k: int = Field(default=1, ge=1)
    replacement: bool = False
    order: Literal["given", "random", "hardest_first"] = "given"
    """Ordine di elaborazione delle unita' query (il greedy dipende dall'ordine)."""
    prefilter_k: int | None = Field(default=None, ge=1)


class HungarianParams(_Params):
    """Assegnazione ottima 1:1 (min costo globale, scipy.linear_sum_assignment)."""

    prefilter_k: int | None = Field(default=None, ge=1)
    """Se impostato, ogni query considera solo i k pool piu' vicini (riduce memoria/tempo)."""


class SortedParams(_Params):
    """Matching ottimo monotono su UNA variabile ordinabile (programmazione dinamica)."""

    sort_variable: str | None = None
    """Variabile di ordinamento. Obbligatoria se i matching variables sono piu' di uno."""


class PartialOTParams(_Params):
    """Optimal transport parziale: SELEZIONA un sottoinsieme del pool (non coppie)."""

    target_size: int | None = Field(default=None, gt=0)
    """Numero totale di unita' pool da selezionare (ripartito tra gli strati)."""
    target_ratio: float | None = Field(default=None, gt=0)
    """Alternativa a target_size: unita' pool selezionate per ogni unita' query."""
    nb_dummies: int = Field(default=10, ge=0)
    max_iterations: int = Field(default=1_000_000, gt=0)
    prefilter_k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_target(self) -> "PartialOTParams":
        if self.target_size is not None and self.target_ratio is not None:
            raise ValueError("Specificare solo uno tra target_size e target_ratio.")
        if self.target_size is None and self.target_ratio is None:
            self.target_ratio = 1.0
        return self


ALGORITHM_NAMES = ("greedy_nn", "hungarian", "sorted", "partial_ot")


class PropensityScoreParams(_Params):
    """Stima del propensity score + matching su logit(PS) (o PS)."""

    model: Literal["logistic", "random_forest", "xgboost", "gradient_boosting"] = "logistic"
    model_params: dict[str, Any] = Field(default_factory=dict)
    """Iperparametri passati al modello (sovrascrivono i default): es. {"n_estimators": 500, "max_depth": 3}
    per random_forest/xgboost/gradient_boosting; {"penalty": "l2"} per logistic."""
    poly_degree: int = Field(default=1, ge=1, le=4)
    """Solo logistic: grado dei termini polinomiali."""
    C: float = Field(default=0.5, gt=0)
    """Solo logistic: inverso della regolarizzazione L2."""
    max_iter: int = Field(default=5000, gt=0)
    """Solo logistic: iterazioni massime."""
    cross_fit: bool | None = None
    """PS fuori-campione (K-fold) per evitare l'overfitting dei modelli flessibili. None = automatico:
    attivo per random_forest/xgboost/gradient_boosting, spento per logistic."""
    cross_fit_folds: int = Field(default=5, ge=2, le=20)
    include_stratification: bool = True
    """Include le variabili di stratificazione tra le covariate del modello PS."""
    match_on: Literal["logit", "ps"] = "logit"
    caliper_sd: float | None = Field(default=0.2, gt=0)
    """Caliper sulla scala del logit(PS), in SD (Austin: 0.2). None = nessun caliper."""
    trim: Literal["none", "minmax", "quantile"] = "none"
    trim_quantile: float = Field(default=0.02, gt=0, lt=0.5)
    algorithm: Literal["greedy_nn", "hungarian", "sorted", "partial_ot"] = "greedy_nn"
    algorithm_params: dict[str, Any] = Field(default_factory=dict)


METHOD_PARAMS: dict[str, type[_Params]] = {
    "propensity_score": PropensityScoreParams,
    "greedy_nn": GreedyNNParams,
    "hungarian": HungarianParams,
    "sorted": SortedParams,
    "partial_ot": PartialOTParams,
}

METHOD_ALIASES: dict[str, str] = {
    "ps": "propensity_score",
    "propensity": "propensity_score",
    "nearest": "greedy_nn",
    "nearest_neighbor": "greedy_nn",
    "nearest_neighbour": "greedy_nn",
    "nn": "greedy_nn",
    "greedy": "greedy_nn",
    "optimal": "hungarian",
    "optimal_hungarian": "hungarian",
    "ot": "partial_ot",
    "optimal_transport": "partial_ot",
    "partial_optimal_transport": "partial_ot",
    "optimal_transport_selection": "partial_ot",
    "optimal_transport_sinkhorn": "partial_ot",
    "sorted_dp": "sorted",
}


def normalize_method_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return METHOD_ALIASES.get(key, key)
