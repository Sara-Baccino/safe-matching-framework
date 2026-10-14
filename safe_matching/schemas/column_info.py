"""Descrizione dichiarativa di una singola colonna del dataset.

Struttura compatibile con i file JSON SAFE (``new_name``, ``active``, ``id``,
``gene``/``cytogenetic``/``clinical``, ``multiplier``, ``missing_data_management``,
``categorical``/``numerical``, ``mappings``, ``type``). I flag legacy
``gene``/``cytogenetic``/``clinical`` vengono convertiti in ``semantic_roles``
(come in synapse-core), cosi' lo stesso file funziona in entrambi gli ecosistemi.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ColumnType",
    "MissingStrategy",
    "ImputerType",
    "MissingDataManagement",
    "ColumnInfo",
    "DEFAULT_MISSING_TOKENS",
]

DEFAULT_MISSING_TOKENS: list[str] = [
    "NA", "na", "", " ", ".", "NaN", "nan", "/", "n/a", "N/A", "null", "NULL",
]

_LEGACY_ROLE_FLAGS = ("gene", "cytogenetic", "clinical")


class ColumnType(str, Enum):
    INT = "int"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORY = "category"


class MissingStrategy(str, Enum):
    DROP = "drop"
    IMPUTE = "impute"
    REPLACE = "replace"
    MAINTAIN = "maintain"


class ImputerType(str, Enum):
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    KNN = "knn"
    ITERATIVE = "iterative"


class MissingDataManagement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: MissingStrategy = MissingStrategy.MAINTAIN
    value: Any = None
    condition: list[Any] = Field(default_factory=lambda: DEFAULT_MISSING_TOKENS.copy())
    imputer: ImputerType = ImputerType.ZERO

    @model_validator(mode="after")
    def _replace_needs_value(self) -> "MissingDataManagement":
        if self.strategy == MissingStrategy.REPLACE and self.value is None:
            raise ValueError("strategy='replace' richiede un 'value' non nullo.")
        return self


class ColumnInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_name: str
    active: bool = True
    id: bool = False
    categorical: bool = False
    numerical: bool = False
    semantic_roles: set[str] = Field(default_factory=set)
    multiplier: float = 1.0
    mappings: dict[str, Any] = Field(default_factory=dict)
    missing_data_management: MissingDataManagement = Field(default_factory=MissingDataManagement)
    type: ColumnType | None = None

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_role_flags(cls, data: Any) -> Any:
        """gene/cytogenetic/clinical (bool) -> semantic_roles (set[str])."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        roles = set(data.get("semantic_roles") or [])
        for flag in _LEGACY_ROLE_FLAGS:
            if flag in data and data.pop(flag):
                roles.add(flag)
        if roles:
            data["semantic_roles"] = roles
        return data

    @model_validator(mode="after")
    def _check_roles(self) -> "ColumnInfo":
        if self.categorical and self.numerical and not self.id:
            raise ValueError(
                f"Colonna '{self.new_name}': non puo' essere sia categorical sia numerical."
            )
        return self

    @property
    def kind(self) -> str | None:
        """'categorical' | 'numerical' | None (colonne id o non tipizzate)."""
        if self.id:
            return None
        if self.categorical:
            return "categorical"
        if self.numerical:
            return "numerical"
        return None
