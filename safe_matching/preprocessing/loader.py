"""Lettura di un dataset da file in un DataFrame Polars grezzo."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from safe_matching.exceptions import DatasetLoadError

__all__ = ["Loader"]


def _read_csv(path: Path, **kw) -> pl.DataFrame:
    kw.setdefault("infer_schema_length", 10_000)
    kw.setdefault("try_parse_dates", False)
    return pl.read_csv(path, **kw)


def _read_tsv(path: Path, **kw) -> pl.DataFrame:
    kw.setdefault("separator", "\t")
    return _read_csv(path, **kw)


_READERS = {
    ".csv": _read_csv,
    ".tsv": _read_tsv,
    ".parquet": lambda p, **kw: pl.read_parquet(p, **kw),
    ".json": lambda p, **kw: pl.read_json(p, **kw),
    ".ndjson": lambda p, **kw: pl.read_ndjson(p, **kw),
    ".xlsx": lambda p, **kw: pl.read_excel(p, **kw),
    ".xls": lambda p, **kw: pl.read_excel(p, **kw),
}


class Loader:
    @staticmethod
    def supported_extensions() -> list[str]:
        return sorted(_READERS)

    @staticmethod
    def load(path: str | Path, **read_kwargs) -> pl.DataFrame:
        p = Path(path)
        if not p.is_file():
            raise DatasetLoadError(f"Dataset non trovato: {p}")
        reader = _READERS.get(p.suffix.lower())
        if reader is None:
            raise DatasetLoadError(
                f"Estensione '{p.suffix}' non supportata per '{p}'. Supportate: {Loader.supported_extensions()}"
            )
        try:
            df = reader(p, **read_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(f"Lettura di '{p}' fallita: {exc}") from exc
        if df.height == 0:
            raise DatasetLoadError(f"Il dataset '{p}' e' vuoto.")
        return df
