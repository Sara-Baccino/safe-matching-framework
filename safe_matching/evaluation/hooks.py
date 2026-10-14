"""Validazione esterna opzionale (es. un evaluator di safe-validation-framework).

Una *hook* e' una funzione ``pacchetto.modulo:funzione`` che riceve
``before``, ``after`` (polars.DataFrame), ``group_col``, ``variables``, ``kinds`` piu' i
``params`` del JSON e restituisce un dict (-> metriche), un DataFrame polars/pandas
(-> tabella nel report) oppure None. Un errore nella hook NON interrompe la pipeline:
viene riportato nelle note del report.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from safe_matching.schemas.matching_config import HookSpec

__all__ = ["run_hooks", "HookResult"]


@dataclass
class HookResult:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    table: pl.DataFrame | None = None
    error: str | None = None


def _resolve(path: str):
    if ":" not in path:
        raise ValueError("callable deve avere la forma 'pacchetto.modulo:funzione'")
    mod, fn = path.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def run_hooks(
    hooks: list[HookSpec], *, before: pl.DataFrame, after: pl.DataFrame, group_col: str,
    variables: list[str], kinds: dict[str, str],
) -> list[HookResult]:
    out: list[HookResult] = []
    for h in hooks:
        if not h.enabled:
            continue
        res = HookResult(name=h.name)
        try:
            ret = _resolve(h.callable)(
                before=before, after=after, group_col=group_col, variables=variables, kinds=kinds, **h.params
            )
            if isinstance(ret, dict):
                res.metrics = {str(k): v for k, v in ret.items()}
            elif isinstance(ret, pl.DataFrame):
                res.table = ret
            elif ret is not None and hasattr(ret, "to_dict"):  # pandas
                res.table = pl.from_pandas(ret)
        except Exception as exc:  # noqa: BLE001
            res.error = f"{type(exc).__name__}: {exc}"
        out.append(res)
    return out
