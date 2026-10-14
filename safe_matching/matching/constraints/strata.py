"""Stratificazione (match esatto): partiziona la popolazione in strati indipendenti."""

from __future__ import annotations

import polars as pl

__all__ = ["partition_strata"]


def partition_strata(df: pl.DataFrame, columns: list[str]) -> dict[str, pl.DataFrame]:
    if not columns:
        return {"ALL": df}
    out: dict[str, pl.DataFrame] = {}
    for key, part in df.group_by(columns, maintain_order=True):
        vals = key if isinstance(key, tuple) else (key,)
        label = " | ".join(f"{c}={v}" for c, v in zip(columns, vals))
        out[label] = part
    return out
