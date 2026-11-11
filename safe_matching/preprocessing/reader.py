"""Lettura dei dataset: TUTTE le colonne come testo, la tipizzazione e' compito di ``transform``.

Leggere tutto come stringa evita i problemi di inferenza dei tipi (colonne Excel a tipi misti, codifiche, virgole
decimali) e permette di segnalare con precisione i valori che non rispettano la config."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl

from safe_matching.exceptions import DatasetLoadError

__all__ = ["read_table"]


def _sniff_sep(first_line: str) -> str:
    counts = {s: first_line.count(s) for s in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _read_csv(path: Path, **opts: Any) -> pl.DataFrame:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    sep = opts.pop("separator", None) or _sniff_sep(text.splitlines()[0] if text else "")
    return pl.read_csv(io.BytesIO(text.encode("utf-8")), separator=sep, infer_schema_length=0, **opts)


def read_table(source: str | Path | pl.DataFrame, **read_options: Any) -> pl.DataFrame:
    """``read_options``: per Excel ``sheet_name``/``sheet_id``; per CSV ``separator`` (altrimenti rilevato)."""
    if isinstance(source, pl.DataFrame):
        df = source.clone()
    else:
        p = Path(source)
        if not p.is_file():
            raise DatasetLoadError(f"Dataset non trovato: {p}")
        ext = p.suffix.lower()
        try:
            if ext in (".csv", ".txt"):
                df = _read_csv(p, **read_options)
            elif ext == ".tsv":
                df = _read_csv(p, separator="\t", **{k: v for k, v in read_options.items() if k != "separator"})
            elif ext in (".xlsx", ".xlsm", ".xls"):
                df = pl.read_excel(p, engine="calamine", infer_schema_length=0, drop_empty_rows=False,
                                   drop_empty_cols=False, **read_options)
            elif ext == ".parquet":
                df = pl.read_parquet(p)
            else:
                raise DatasetLoadError(f"Estensione '{ext}' non supportata (csv, tsv, xlsx, xls, parquet).")
        except DatasetLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(f"Lettura di '{p}' fallita: {type(exc).__name__}: {exc}") from exc
    if df.height == 0:
        raise DatasetLoadError("Il dataset e' vuoto.")
    return df.rename({c: str(c).strip() for c in df.columns if str(c) != str(c).strip()}) if any(
        str(c) != str(c).strip() for c in df.columns) else df