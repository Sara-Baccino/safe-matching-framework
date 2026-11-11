"""Imputazione dei mancanti con tracciamento (politica: ``options.missing`` del config di matching).

* continue  -> knn su feature standardizzate (o mediana/media);  nominali -> moda;  ordinali -> livello mediano (o moda);
* un dataset per volta; con un solo dataset il gruppo e' una feature del knn (``include_group_in_imputation``);
* gli outcome non si imputano e non entrano nel knn (sono esclusi da ``targets``);
* ogni cella imputata e' tracciata con una colonna booleana ``__imp_<variabile>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import KNNImputer

from safe_matching.exceptions import ConfigError
from safe_matching.preprocessing.config import DataConfig
from safe_matching.schemas.matching_config import MissingPolicy

__all__ = ["IMP_PREFIX", "ColumnRule", "column_rules", "impute_frame", "REPORT_SCHEMA"]

IMP_PREFIX = "__imp_"
REPORT_SCHEMA = {
    "source": pl.Utf8, "variable": pl.Utf8, "role": pl.Utf8, "kind": pl.Utf8, "n_rows": pl.Int64, "n_missing": pl.Int64,
    "pct_missing": pl.Float64, "tier": pl.Utf8, "action": pl.Utf8, "method": pl.Utf8, "n_imputed": pl.Int64, "n_left_missing": pl.Int64,
}


@dataclass
class ColumnRule:
    strategy: str = "impute"
    imputer: str = "knn"
    replace_value: Any = None


def column_rules(cfg: DataConfig) -> dict[str, ColumnRule]:
    """{new_name: regola} dalla data config (``missing_data_management`` per colonna)."""
    return {c.new_name: ColumnRule(c.missing_data_management.strategy, c.missing_data_management.imputer,
                                   c.missing_data_management.missing_replace_value) for c in cfg.active_columns().values()}


def _tier(rate: float, n_missing: int, pol: MissingPolicy) -> str:
    if n_missing == 0:
        return "none"
    return "low" if rate <= pol.low_threshold else ("moderate" if rate <= pol.max_fraction else "high")


def _features(df: pl.DataFrame, cols: list[str], kinds: dict[str, str], ordinal: set[str], group_col: str | None):
    """Matrice standardizzata per il knn: numeriche/ordinali/binarie come sono, nominali >2 livelli in one-hot."""
    parts, index = [], {}
    for c in cols:
        s = df[c]
        if kinds.get(c) == "categorical" and c not in ordinal and s.drop_nulls().n_unique() > 2:
            for lv in sorted(s.drop_nulls().unique().to_list(), key=str):
                v = (s == lv).cast(pl.Float64).to_numpy().astype(float)
                v[s.is_null().to_numpy()] = np.nan
                parts.append(v)
        else:
            index[c] = len(parts)
            parts.append(s.cast(pl.Float64).fill_nan(None).to_numpy().astype(float))
    if group_col and group_col in df.columns:
        parts.append(df[group_col].cast(pl.Float64).to_numpy().astype(float))
    X = np.column_stack(parts)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    mu, sd = np.where(np.isnan(mu), 0.0, mu), np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    return (X - mu) / sd, mu, sd, index


def impute_frame(
    df: pl.DataFrame,
    *,
    source: str,
    targets: dict[str, str],
    kinds: dict[str, str],
    ordinal: set[str],
    policy: MissingPolicy,
    rules: dict[str, ColumnRule],
    group_col: str | None = None,
) -> tuple[pl.DataFrame, list[dict[str, Any]], list[str]]:
    """Imputa le colonne ``targets`` ({variabile: ruolo}). Ritorna (df, righe del report, variabili da escludere)."""
    n = df.height
    cols = [v for v in targets if v in df.columns]
    float_cols = [v for v in cols if df.schema[v].is_float()]
    if float_cols:
        df = df.with_columns([pl.col(v).fill_nan(None) for v in float_cols])

    report: list[dict[str, Any]] = []
    dropped: list[str] = []
    plan: dict[str, tuple[str, str]] = {}          # variabile -> (azione, metodo)
    for v in cols:
        nm = df[v].null_count()
        rate = nm / n if n else 0.0
        tier = _tier(rate, nm, policy)
        rule = rules.get(v, ColumnRule())
        kind = "ordinal" if v in ordinal else kinds.get(v, "numerical")
        base = {"source": source, "variable": v, "role": targets[v], "kind": kind, "n_rows": n, "n_missing": nm,
                "pct_missing": 100 * rate, "tier": tier, "action": "none", "method": "", "n_imputed": 0, "n_left_missing": nm}
        report.append(base)
        if nm == 0:
            continue
        if rule.strategy in ("maintain", "drop") or not policy.impute:
            base["action"] = "left_missing"
            base["method"] = "complete case" if (rule.strategy == "drop" or not policy.impute) else "maintain"
            continue
        if rule.strategy == "replace":
            plan[v] = ("replace", "replace")
            continue
        if nm == n:
            raise ConfigError(f"[{source}] '{v}' e' mancante in tutte le righe: impossibile imputarla.")
        if tier == "high":
            if policy.over_threshold_action == "error":
                raise ConfigError(f"[{source}] '{v}' ha {100 * rate:.1f}% di mancanti (> {100 * policy.max_fraction:.0f}%). "
                                  "Rimuovila dal matching o imposta options.missing.over_threshold_action.")
            if policy.over_threshold_action == "drop_variable":
                base["action"], base["method"] = "dropped_variable", f">{100 * policy.max_fraction:.0f}% missing"
                dropped.append(v)
                continue
        if kind == "numerical":
            m = policy.continuous_imputer
            if policy.use_column_imputer and rule.imputer in ("median", "mean", "zero"):
                m = rule.imputer
            m = "knn" if m == "iterative" else m
        elif kind == "ordinal":
            m = "mode" if (policy.use_column_imputer and rule.imputer == "mode") else policy.ordinal_imputer
        else:
            m = "mode"
        plan[v] = ("impute", m)

    live = [v for v in cols if v not in dropped]
    # righe con troppi mancanti: non imputate (restano escluse dal matching)
    skip = np.zeros(n, dtype=bool)
    if len(live) >= 2:
        miss = np.column_stack([df[v].is_null().to_numpy() for v in live])
        skip = (miss.sum(axis=1) >= 2) & (miss.mean(axis=1) > policy.row_max_fraction)

    knn_targets = [v for v, (a, m) in plan.items() if a == "impute" and m == "knn"]
    knn_out: dict[str, np.ndarray] = {}
    if knn_targets:
        Z, mu, sd, index = _features(df, live, kinds, ordinal, group_col if policy.include_group_in_imputation else None)
        k = int(max(1, min(policy.knn_neighbors, n - 1)))
        Zi = KNNImputer(n_neighbors=k, keep_empty_features=True).fit_transform(Z)
        for v in knn_targets:
            j = index[v]
            knn_out[v] = Zi[:, j] * sd[j] + mu[j]

    new_cols, flags = [], []
    by_var = {r["variable"]: r for r in report}
    for v, (action, method) in plan.items():
        s = df[v]
        was_null = s.is_null().to_numpy()
        vals = s.cast(pl.Float64).to_numpy().astype(float)
        obs = vals[~np.isnan(vals)]
        if action == "replace":
            fill = np.full(n, float(rules[v].replace_value))
        elif method == "knn":
            fill = np.clip(knn_out[v], obs.min(), obs.max())
        elif method == "median":
            fill = np.full(n, float(np.median(obs)))
        elif method == "mean":
            fill = np.full(n, float(obs.mean()))
        elif method == "zero":
            fill = np.zeros(n)
        elif method == "mode":
            u, c = np.unique(obs, return_counts=True)
            fill = np.full(n, float(u[np.argmax(c)]))
        else:
            raise ConfigError(f"Imputer '{method}' non supportato.")
        if v in ordinal and method == "median":
            fill = np.full(n, float(np.sort(obs)[(len(obs) - 1) // 2]))   # livello osservato (mediana inferiore)
        do = was_null & ~skip
        out = vals.copy()
        out[do] = fill[do]
        ser = pl.Series(v, out, nan_to_null=True)
        if s.dtype.is_integer():                      # colonne intere (eta', codici): il valore imputato resta intero
            ser = ser.round(0).cast(s.dtype)
        new_cols.append(ser)
        r = by_var[v]
        r["action"], r["method"] = ("replaced" if action == "replace" else "imputed"), method
        r["n_imputed"] = int(do.sum())
        r["n_left_missing"] = int((was_null & skip).sum())
        if policy.add_flags and do.any():
            flags.append(pl.Series(IMP_PREFIX + v, do))
    out_df = df.with_columns(new_cols + flags) if (new_cols or flags) else df
    return out_df, report, dropped