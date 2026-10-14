"""Diagnostiche di bilanciamento pre/post matching (pesate)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.spatial.distance import jensenshannon

from safe_matching.preprocessing.assembly import GROUP_COL

__all__ = ["balance_table", "summarize_balance", "BalanceResult"]

ALL_METRICS = ("smd", "variance_ratio", "ks", "chi_square", "jensen_shannon", "wasserstein")


@dataclass
class BalanceResult:
    table: pl.DataFrame
    summary: dict[str, float | int]


def _wmean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    sw = w.sum()
    if sw <= 0 or x.size == 0:
        return float("nan"), float("nan")
    m = float((w * x).sum() / sw)
    denom = sw - (w ** 2).sum() / sw
    v = float((w * (x - m) ** 2).sum() / denom) if denom > 0 else float("nan")
    return m, v


def _smd(m1: float, v1: float, m0: float, v0: float) -> float:
    pooled = np.sqrt((np.nan_to_num(v1) + np.nan_to_num(v0)) / 2.0)
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0 if abs(m1 - m0) < 1e-12 else float("nan")
    return float((m1 - m0) / pooled)


def _weighted_ks(x1, w1, x0, w0) -> float:
    if x1.size == 0 or x0.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x1, x0]))

    def cdf(x, w):
        o = np.argsort(x, kind="stable")
        xs, cw = x[o], np.cumsum(w[o])
        idx = np.searchsorted(xs, grid, side="right")
        return np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) / cw[-1]

    return float(np.max(np.abs(cdf(x1, w1) - cdf(x0, w0))))


def _uniform(w: np.ndarray) -> bool:
    return w.size > 0 and bool(np.allclose(w, w[0]))


def _evaluate(df: pl.DataFrame, var: str, level: Any | None, kind: str, weight_col: str | None) -> dict[str, float]:
    g = df[GROUP_COL].to_numpy()
    w_all = df[weight_col].to_numpy().astype(float) if weight_col else np.ones(df.height)
    s = df[var]
    if level is not None:
        x_all = (s == level).cast(pl.Float64).to_numpy()
    else:
        x_all = s.cast(pl.Float64).to_numpy()
    ok = ~np.isnan(x_all)
    t, c = (g == 1) & ok, (g == 0) & ok
    x1, w1, x0, w0 = x_all[t], w_all[t], x_all[c], w_all[c]
    m1, v1 = _wmean_var(x1, w1)
    m0, v0 = _wmean_var(x0, w0)
    out = {
        "mean_t": m1, "mean_c": m0, "smd": _smd(m1, v1, m0, v0),
        "variance_ratio": (v1 / v0) if (v0 and np.isfinite(v0) and v0 > 0) else float("nan"),
        "ks": float("nan"), "ks_p": float("nan"), "wasserstein": float("nan"),
    }
    if kind == "numerical" and x1.size and x0.size:
        out["ks"] = _weighted_ks(x1, w1, x0, w0)
        if _uniform(w1) and _uniform(w0):
            out["ks_p"] = float(stats.ks_2samp(x1, x0).pvalue)
        pooled = np.sqrt((np.nan_to_num(v1) + np.nan_to_num(v0)) / 2.0)
        if pooled > 0:
            out["wasserstein"] = float(stats.wasserstein_distance(x1, x0, w1, w0) / pooled)
    return out


def _categorical_overall(df: pl.DataFrame, var: str, weight_col: str | None) -> tuple[float, float]:
    """(Jensen-Shannon divergence, p-value chi-quadro) sulla variabile categorica intera."""
    g = df[GROUP_COL].to_numpy()
    w = df[weight_col].to_numpy().astype(float) if weight_col else np.ones(df.height)
    vals = df[var].cast(pl.Utf8).fill_null("NA").to_numpy()
    levels = np.unique(vals)
    tab = np.array([[w[(g == gr) & (vals == lv)].sum() for lv in levels] for gr in (1, 0)])
    if (tab.sum(axis=1) == 0).any():
        return float("nan"), float("nan")
    p, q = tab[0] / tab[0].sum(), tab[1] / tab[1].sum()
    js = float(jensenshannon(p, q, base=2) ** 2)
    pval = float("nan")
    if _uniform(w) and tab.shape[1] > 1 and (tab.sum(axis=0) > 0).all():
        try:
            pval = float(stats.chi2_contingency(tab)[1])
        except ValueError:
            pass
    return js, pval


def balance_table(
    before: pl.DataFrame,
    after: pl.DataFrame,
    variables: dict[str, str],
    kinds: dict[str, str],
    *,
    smd_threshold: float = 0.1,
    weight_col: str | None = "weight",
    metrics: tuple[str, ...] | list[str] = ALL_METRICS,
) -> BalanceResult:
    """``variables``: {nome: ruolo} con ruolo in {'matching','stratification','evaluation'}."""
    rows: list[dict[str, Any]] = []
    for var, role in variables.items():
        if var not in before.columns or var not in after.columns:
            continue
        dtype = before.schema[var]
        if not dtype.is_numeric() and dtype != pl.Boolean:
            continue
        kind = kinds.get(var) or "numerical"
        n_levels = before[var].drop_nulls().n_unique()
        levels: list[Any | None]
        if kind == "categorical" and n_levels > 2:
            levels = sorted(before[var].drop_nulls().unique().to_list())
        else:
            levels = [None]
        js_b = js_a = chi_b = chi_a = float("nan")
        if kind == "categorical":
            js_b, chi_b = _categorical_overall(before, var, None)
            js_a, chi_a = _categorical_overall(after, var, weight_col)
        for lv in levels:
            b = _evaluate(before, var, lv, kind, None)
            a = _evaluate(after, var, lv, kind, weight_col)
            rows.append({
                "variable": var if lv is None else f"{var}={lv}", "base_variable": var, "role": role, "type": kind,
                "mean_t_before": b["mean_t"], "mean_c_before": b["mean_c"],
                "mean_t_after": a["mean_t"], "mean_c_after": a["mean_c"],
                "smd_before": b["smd"], "smd_after": a["smd"],
                "abs_smd_before": abs(b["smd"]), "abs_smd_after": abs(a["smd"]),
                "variance_ratio_before": b["variance_ratio"], "variance_ratio_after": a["variance_ratio"],
                "ks_before": b["ks"], "ks_after": a["ks"], "ks_p_after": a["ks_p"],
                "wasserstein_before": b["wasserstein"], "wasserstein_after": a["wasserstein"],
                "js_before": js_b, "js_after": js_a, "chi2_p_before": chi_b, "chi2_p_after": chi_a,
            })
    table = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"variable": pl.Utf8})
    if not rows:
        return BalanceResult(table, {"n_variables": 0})
    table = table.with_columns((pl.col("abs_smd_after") < smd_threshold).alias("balanced_after"))
    drop = []
    if "smd" not in metrics:
        drop += ["smd_before", "smd_after"]
    if "variance_ratio" not in metrics:
        drop += ["variance_ratio_before", "variance_ratio_after"]
    if "ks" not in metrics:
        drop += ["ks_before", "ks_after", "ks_p_after"]
    if "wasserstein" not in metrics:
        drop += ["wasserstein_before", "wasserstein_after"]
    if "jensen_shannon" not in metrics:
        drop += ["js_before", "js_after"]
    if "chi_square" not in metrics:
        drop += ["chi2_p_before", "chi2_p_after"]
    table = table.drop([c for c in drop if c in table.columns])
    return BalanceResult(table, summarize_balance(table, smd_threshold))


def summarize_balance(table: pl.DataFrame, thr: float) -> dict[str, float | int]:
    if table.height == 0 or "abs_smd_before" not in table.columns:
        return {"n_variables": 0}
    b = table["abs_smd_before"].drop_nans().drop_nulls()
    a = table["abs_smd_after"].drop_nans().drop_nulls()
    return {
        "n_variables": int(table.height),
        "smd_threshold": thr,
        "mean_abs_smd_before": float(b.mean()) if b.len() else float("nan"),
        "mean_abs_smd_after": float(a.mean()) if a.len() else float("nan"),
        "max_abs_smd_before": float(b.max()) if b.len() else float("nan"),
        "max_abs_smd_after": float(a.max()) if a.len() else float("nan"),
        "pct_balanced_before": float((b < thr).mean() * 100) if b.len() else float("nan"),
        "pct_balanced_after": float((a < thr).mean() * 100) if a.len() else float("nan"),
        "n_unbalanced_after": int((a >= thr).sum()),
    }
