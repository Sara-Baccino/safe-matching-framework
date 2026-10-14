"""Configurazione dati: collezione di ColumnInfo indicizzata per NOME ORIGINALE.

Diversamente da synapse-core (dove la chiave deve coincidere con ``new_name``),
qui la chiave e' il nome della colonna *nel file sorgente* e ``new_name`` e' il
nome dopo il preprocessing: e' la convenzione dei JSON SAFE (es. ``"Eta"`` ->
``"Age"``). Sono ammessi ``new_name`` duplicati solo tra colonne inattive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from safe_matching.exceptions import ConfigError
from safe_matching.schemas.column_info import ColumnInfo

__all__ = ["DataConfig"]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: dict[str, ColumnInfo]

    @model_validator(mode="after")
    def _unique_active_new_names(self) -> "DataConfig":
        seen: dict[str, str] = {}
        for raw, info in self.columns.items():
            if not info.active:
                continue
            if info.new_name in seen:
                raise ValueError(
                    f"new_name '{info.new_name}' duplicato tra colonne attive: "
                    f"'{seen[info.new_name]}' e '{raw}'."
                )
            seen[info.new_name] = raw
        return self

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def active_columns(self) -> dict[str, ColumnInfo]:
        """{nome_originale: ColumnInfo} per le sole colonne attive."""
        return {raw: info for raw, info in self.columns.items() if info.active}

    def new_names(self, *, include_id: bool = False) -> list[str]:
        return [i.new_name for i in self.active_columns().values() if include_id or not i.id]

    def id_new_names(self) -> list[str]:
        return [i.new_name for i in self.active_columns().values() if i.id]

    def kinds(self) -> dict[str, str]:
        """{new_name: 'categorical'|'numerical'} per le colonne attive non-id tipizzate."""
        out: dict[str, str] = {}
        for info in self.active_columns().values():
            if info.kind is not None:
                out[info.new_name] = info.kind
        return out

    def raw_name_of(self, name: str) -> str | None:
        """Nome originale a partire da un nome originale *o* da un new_name attivo."""
        if name in self.columns:
            return name
        for raw, info in self.active_columns().items():
            if info.new_name == name:
                return raw
        return None

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        payload = data if set(data.keys()) == {"columns"} else {"columns": data}
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ConfigError(f"DataConfig non valida: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "DataConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"File di configurazione dati non trovato: {p}")
        try:
            parsed = json.loads(p.read_text(encoding="utf-8"))
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

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
