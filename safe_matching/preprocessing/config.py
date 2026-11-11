"""Configurazione dati (JSON) - schema tollerante.

Chiave = nome della colonna NEL FILE; ``new_name`` = nome usato dal framework. Si accettano sia il formato attuale
(``mappings``, ``missing_data_management.missing_values``) sia quello precedente (``categorical_mappings``,
``condition``/``value``, ``gene``/``cytogenetic``/``clinical``, ``id``): i campi sconosciuti sono ignorati.

Tipo statistico della colonna (``kind``):
  categorical=true,  numerical=false -> "categorical"  (nominale)
  categorical=false, numerical=true  -> "numerical"
  categorical=true,  numerical=true  -> "ordinal"      (categorica ORDINATA: l'ordine e' quello dei codici in ``mappings``,
                                                         o dei valori numerici se non c'e' mapping)
``type`` descrive il tipo dei valori NEL FILE (int, float, str, bool, date, datetime). Se la colonna ha ``mappings``,
il risultato e' numerico (i codici) indipendentemente da ``type``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from safe_matching.exceptions import ConfigError

__all__ = ["ColumnConfig", "DataConfig", "MissingDataManagement", "canonical_type"]

_STRATEGIES = ("maintain", "impute", "drop", "replace")
_IMPUTERS = {"zero": "zero", "mean": "mean", "median": "median", "most_frequent": "mode", "mode": "mode",
             "moda": "mode", "knn": "knn", "iterative": "iterative"}
_TYPES = {
    **dict.fromkeys(("int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "integer"), "int"),
    **dict.fromkeys(("float", "float32", "float64", "double", "number", "numeric"), "float"),
    **dict.fromkeys(("str", "string", "object", "category", "text"), "str"),
    **dict.fromkeys(("bool", "boolean"), "bool"),
    "date": "date", "datetime": "datetime",
}


def canonical_type(t: str | None) -> str | None:
    if t is None:
        return None
    return _TYPES.get(str(t).strip().lower())


class MissingDataManagement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: str = "maintain"
    imputer: str = "knn"
    missing_values: list[Any] = Field(default_factory=list)
    """Token aggiuntivi da trattare come mancanti (oltre a vuoto, NA, N/A, NaN, null, None)."""
    missing_replace_value: Any = None

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "condition" in data and "missing_values" not in data:
                data["missing_values"] = data.pop("condition")
            if "value" in data and "missing_replace_value" not in data:
                data["missing_replace_value"] = data.pop("value")
        return data

    @model_validator(mode="after")
    def _check(self) -> "MissingDataManagement":
        self.strategy = self.strategy.strip().lower()
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy '{self.strategy}' non valida. Valide: {list(_STRATEGIES)}")
        imp = _IMPUTERS.get(self.imputer.strip().lower())
        if imp is None:
            raise ValueError(f"imputer '{self.imputer}' non valido. Validi: {sorted(set(_IMPUTERS))}")
        self.imputer = imp
        if self.strategy == "replace" and self.missing_replace_value is None:
            raise ValueError("strategy 'replace' richiede missing_replace_value.")
        return self


class ColumnConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    new_name: str
    active: bool = True
    id: bool = False
    categorical: bool = False
    numerical: bool = False
    mappings: dict[str, Any] = Field(default_factory=dict)
    missing_data_management: MissingDataManagement = Field(default_factory=MissingDataManagement)
    type: str | None = None
    multiplier: float = 1.0
    scale: bool = False
    """Ignorato: il framework non scala mai i dati (le trasformazioni avvengono dentro il matching)."""

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("mappings") and data.get("categorical_mappings"):
                data["mappings"] = data["categorical_mappings"]
        return data

    @model_validator(mode="after")
    def _check(self) -> "ColumnConfig":
        if self.type is not None and canonical_type(self.type) is None:
            raise ValueError(f"type '{self.type}' non riconosciuto (usa int, float, str, bool, date, datetime).")
        return self

    @property
    def kind(self) -> str | None:
        if self.id:
            return None
        if self.categorical and self.numerical:
            return "ordinal"
        if self.categorical:
            return "categorical"
        if self.numerical:
            return "numerical"
        return None


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: dict[str, ColumnConfig]

    @model_validator(mode="after")
    def _unique_active_names(self) -> "DataConfig":
        seen: dict[str, str] = {}
        for raw, c in self.columns.items():
            if c.active:
                if c.new_name in seen:
                    raise ValueError(f"new_name '{c.new_name}' duplicato tra colonne attive: '{seen[c.new_name]}' e '{raw}'.")
                seen[c.new_name] = raw
        return self

    def active_columns(self) -> dict[str, ColumnConfig]:
        return {r: c for r, c in self.columns.items() if c.active}

    def raw_name_of(self, name: str) -> str | None:
        if name in self.columns:
            return name
        for raw, c in self.active_columns().items():
            if c.new_name == name:
                return raw
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        payload = data if set(data) == {"columns"} else {"columns": data}
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ConfigError(f"Data config non valida:\n{exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "DataConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"File di configurazione dati non trovato: {p}")
        try:
            parsed = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"JSON non valido in '{p}': {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"'{p}': attesa una mappa {{colonna: {{...}}}}.")
        try:
            return cls.from_dict(parsed)
        except ConfigError as exc:
            raise ConfigError(f"{p}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")