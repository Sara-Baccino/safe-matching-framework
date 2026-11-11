"""Sorgente dati: file grezzo (testo) + dataset tipizzato + config, con conversione inversa per l'output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from safe_matching.exceptions import ConfigError
from safe_matching.preprocessing.config import DataConfig
from safe_matching.preprocessing.reader import read_table
from safe_matching.preprocessing.transform import TransformResult, UnmappedValuesError, apply_config, inverse_transform

__all__ = ["SourceData", "load_source", "column_kinds", "ordinal_columns", "resolve_raw_column"]


@dataclass
class SourceData:
    name: str
    raw: pl.DataFrame
    """Dataset come letto dal file: tutte le colonne come testo."""
    pre: pl.DataFrame
    """Colonne attive, rinominate e tipizzate (mapping applicato, mancanti = null, NON scalate)."""
    config: DataConfig
    config_ref: str
    transform: TransformResult
    mapping_additions: dict[str, dict[str, int]] = field(default_factory=dict)
    """{new_name: {etichetta: codice}} aggiunti automaticamente con on_unmapped='union'."""

    def to_original(self, df: pl.DataFrame, *, to_labels: bool = True) -> pl.DataFrame:
        """Riporta una tabella del framework a nomi di colonna e valori (etichette) del file originale."""
        return inverse_transform(df, self.config, self.transform, to_labels=to_labels)


def load_source(
    dataset: str | Path | pl.DataFrame,
    data_config: str | Path | DataConfig,
    name: str,
    *,
    read_options: dict[str, Any] | None = None,
    on_unmapped: str = "error",
    on_cast_error: str = "error",
    strict_columns: bool = True,
) -> SourceData:
    cfg = data_config if isinstance(data_config, DataConfig) else DataConfig.from_file(data_config)
    ref = str(data_config) if isinstance(data_config, (str, Path)) else "<in-memory>"
    raw = read_table(dataset, **(read_options or {}))
    additions: dict[str, dict[str, int]] = {}
    if on_unmapped == "union":
        cfg = cfg.model_copy(deep=True)
        try:
            res = apply_config(raw, cfg, name=name, on_unmapped="error", on_cast_error=on_cast_error, strict_columns=strict_columns)
        except UnmappedValuesError as exc:
            additions = _extend_mappings(cfg, exc, name)
            res = apply_config(raw, cfg, name=name, on_unmapped="error", on_cast_error=on_cast_error, strict_columns=strict_columns)
            res.warnings.append(f"Etichette aggiunte ai mappings (on_unmapped='union'): {additions}")
    else:
        res = apply_config(raw, cfg, name=name, on_unmapped=on_unmapped, on_cast_error=on_cast_error, strict_columns=strict_columns)
    return SourceData(name=name, raw=raw, pre=res.data, config=cfg, config_ref=ref, transform=res, mapping_additions=additions)


def _extend_mappings(cfg: DataConfig, exc: UnmappedValuesError, name: str) -> dict[str, dict[str, int]]:
    """Aggiunge ai mappings le etichette non tracciate delle colonne NOMINALI (codici successivi al massimo esistente).
    Le ordinali non si estendono: l'ordine delle nuove etichette non e' deducibile."""
    added: dict[str, dict[str, int]] = {}
    for new_name, labels in exc.unmapped.items():
        raw = cfg.raw_name_of(new_name)
        col = cfg.columns[raw]
        if col.kind == "ordinal":
            raise UnmappedValuesError(
                f"[{name}] '{new_name}' e' ordinale: le etichette non tracciate {list(labels)[:10]} non si possono aggiungere "
                "automaticamente (l'ordine non e' noto). Aggiungile a 'mappings' con il codice corretto.", unmapped=exc.unmapped)
        nxt = int(max([float(v) for v in col.mappings.values()] + [-1])) + 1
        added[new_name] = {}
        for lab in sorted(labels):
            col.mappings[lab] = nxt
            added[new_name][lab] = nxt
            nxt += 1
    return added


def column_kinds(cfg: DataConfig) -> dict[str, str]:
    """{new_name: 'categorical'|'numerical'}. Le ordinali valgono 'numerical' per il motore (codici ordinati)."""
    out: dict[str, str] = {}
    for c in cfg.active_columns().values():
        k = c.kind
        if k == "categorical":
            out[c.new_name] = "categorical"
        elif k in ("numerical", "ordinal"):
            out[c.new_name] = "numerical"
    return out


def ordinal_columns(cfg: DataConfig) -> set[str]:
    return {c.new_name for c in cfg.active_columns().values() if c.kind == "ordinal"}


def resolve_raw_column(raw: pl.DataFrame, cfg: DataConfig, name: str) -> str:
    if name in raw.columns:
        return name
    r = cfg.raw_name_of(name)
    if r is not None and r in raw.columns:
        return r
    raise ConfigError(f"Colonna '{name}' non trovata nel dataset (colonne: {raw.columns[:20]}).")