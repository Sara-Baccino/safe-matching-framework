"""Grafici PNG (matplotlib, backend Agg) per il report."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from safe_matching.preprocessing.assembly import GROUP_COL  # noqa: E402

__all__ = ["love_plot", "distance_hist", "ps_overlap", "distribution_panel", "strata_bars"]

_C_T, _C_C = "#1f4e79", "#c0504d"
_DPI = 150


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def love_plot(table: pl.DataFrame, thr: float, max_vars: int = 40) -> bytes:
    t = table.sort("abs_smd_before", descending=True, nulls_last=True).head(max_vars).reverse()
    names = t["variable"].to_list()
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.28 * len(names) + 1.2)))
    ax.scatter(t["abs_smd_before"].to_numpy(), y, c="#9e9e9e", marker="o", s=26, label="Before")
    ax.scatter(t["abs_smd_after"].to_numpy(), y, c=_C_T, marker="D", s=26, label="After")
    for i in range(len(y)):
        b, a = t["abs_smd_before"][i], t["abs_smd_after"][i]
        if b is not None and a is not None and np.isfinite(b) and np.isfinite(a):
            ax.plot([b, a], [y[i], y[i]], color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(thr, color=_C_C, ls="--", lw=1, label=f"threshold {thr:g}")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("|SMD|")
    ax.set_title("Covariate balance (Love plot)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return _png(fig)


def distance_hist(dist: np.ndarray, cap: float | None = None) -> bytes:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(dist, bins=min(40, max(8, int(np.sqrt(dist.size)))), color=_C_T, alpha=0.85)
    if cap is not None:
        ax.axvline(cap, color=_C_C, ls="--", label=f"caliper = {cap:.3g}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Pair distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of matched-pair distances", fontsize=10)
    return _png(fig)


def ps_overlap(pop: pl.DataFrame, matched: pl.DataFrame, col: str = "__logit_ps__") -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, sharex=True)
    for ax, df, title in ((axes[0], pop, "Before"), (axes[1], matched, "After")):
        g = df[GROUP_COL].to_numpy()
        x = df[col].to_numpy()
        w = df["weight"].to_numpy() if "weight" in df.columns else np.ones(df.height)
        bins = np.linspace(np.nanmin(pop[col].to_numpy()), np.nanmax(pop[col].to_numpy()), 30)
        ax.hist(x[g == 1], bins=bins, weights=w[g == 1], alpha=0.55, color=_C_T, label="Treated", density=True)
        ax.hist(x[g == 0], bins=bins, weights=w[g == 0], alpha=0.55, color=_C_C, label="Control", density=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("logit(PS)" if "logit" in col else "PS")
    axes[0].legend(fontsize=8)
    fig.suptitle("Propensity score overlap", fontsize=10)
    return _png(fig)


def distribution_panel(before: pl.DataFrame, after: pl.DataFrame, var: str, kind: str, label: str = "") -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    for ax, df, title in ((axes[0], before, "Before"), (axes[1], after, "After")):
        g = df[GROUP_COL].to_numpy()
        w = df["weight"].to_numpy() if "weight" in df.columns and title == "After" else np.ones(df.height)
        x = df[var].cast(pl.Float64).to_numpy()
        ok = ~np.isnan(x)
        if kind == "categorical":
            levels = np.unique(before[var].drop_nulls().to_numpy())
            width = 0.38
            for k, (gr, col, name) in enumerate(((1, _C_T, "Treated"), (0, _C_C, "Control"))):
                m = (g == gr) & ok
                tot = w[m].sum()
                props = [w[m & (x == lv)].sum() / tot if tot else 0 for lv in levels]
                ax.bar(np.arange(len(levels)) + (k - 0.5) * width, props, width, color=col, alpha=0.8, label=name)
            ax.set_xticks(np.arange(len(levels)))
            ax.set_xticklabels([str(int(l)) if float(l).is_integer() else str(l) for l in levels], fontsize=8)
        else:
            lo, hi = np.nanpercentile(before[var].cast(pl.Float64).to_numpy(), [0.5, 99.5])
            bins = np.linspace(lo, hi, 28)
            for gr, col, name in ((1, _C_T, "Treated"), (0, _C_C, "Control")):
                m = (g == gr) & ok
                ax.hist(x[m], bins=bins, weights=w[m], density=True, histtype="stepfilled", alpha=0.45, color=col, label=name)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{var} {label}".strip(), fontsize=10)
    return _png(fig)


def strata_bars(strata: pl.DataFrame, max_rows: int = 25) -> bytes:
    t = strata.sort("n_query", descending=True).head(max_rows).reverse()
    y = np.arange(t.height)
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.3 * t.height + 1)))
    ax.barh(y + 0.2, t["n_query"].to_numpy(), 0.4, color="#9e9e9e", label="Query units in stratum")
    ax.barh(y - 0.2, t["n_query_matched"].to_numpy(), 0.4, color=_C_T, label="Query units matched")
    ax.set_yticks(y)
    ax.set_yticklabels(t["stratum"].to_list(), fontsize=7)
    ax.set_title("Coverage by stratum", fontsize=10)
    ax.legend(fontsize=8)
    return _png(fig)
