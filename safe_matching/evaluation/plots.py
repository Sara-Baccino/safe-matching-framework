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


# --------------------------------------------------------------------------------------------------
# Analisi multivariate: PCA, griglie di distribuzioni, correlazioni
# --------------------------------------------------------------------------------------------------
from matplotlib.patches import Ellipse  # noqa: E402


def _ellipse(ax, xy: np.ndarray, w: np.ndarray, color: str) -> None:
    if xy.shape[0] < 5:
        return
    m = np.average(xy, axis=0, weights=w)
    c = np.cov(xy.T, aweights=w)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, 0, None)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(m, 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang, fill=False, ec=color, lw=1.4))
    ax.scatter(*m, marker="X", s=70, c=color, edgecolors="white", zorder=5)


def pca_scatter(scores_before, g_before, w_before, scores_after, g_after, w_after, evr, max_points: int = 3000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), sharex=True, sharey=True)
    for ax, S, g, w, title in ((axes[0], scores_before, g_before, w_before, "Before matching"),
                               (axes[1], scores_after, g_after, w_after, "After matching (weighted)")):
        idx = rng.choice(len(S), min(len(S), max_points), replace=False)
        for gr, col, name in ((0, _C_C, "Control"), (1, _C_T, "Treated")):
            m = g == gr
            ax.scatter(S[idx][m[idx], 0], S[idx][m[idx], 1], s=7, alpha=0.30, c=col, label=name, linewidths=0)
            _ellipse(ax, S[m][:, :2], w[m], col)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)")
    axes[0].legend(fontsize=8, markerscale=2.5)
    fig.suptitle("PCA of the analysis variables (ellipse = 2 SD, X = group centroid)", fontsize=10)
    return _png(fig)


def pca_variance(evr: np.ndarray, smd_before: np.ndarray, smd_after: np.ndarray, thr: float) -> bytes:
    k = len(smd_before)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    n = min(len(evr), 10)
    axes[0].bar(np.arange(1, n + 1), 100 * evr[:n], color=_C_T, alpha=0.85)
    ax2 = axes[0].twinx()
    ax2.plot(np.arange(1, n + 1), 100 * np.cumsum(evr)[:n], color=_C_C, marker="o", ms=3)
    ax2.set_ylim(0, 105)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Explained variance", fontsize=9)
    x = np.arange(1, k + 1)
    axes[1].bar(x - 0.2, np.abs(smd_before), 0.4, color="#9e9e9e", label="Before")
    axes[1].bar(x + 0.2, np.abs(smd_after), 0.4, color=_C_T, label="After")
    axes[1].axhline(thr, color=_C_C, ls="--", lw=1)
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("|SMD| of the scores")
    axes[1].set_title("Group imbalance along each PC", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _png(fig)


def pca_loadings(loadings: np.ndarray, names: list[str], comps: list[str], top: int = 18) -> bytes:
    order = np.argsort(-np.abs(loadings).max(axis=1))[:top]
    L = loadings[order]
    fig, ax = plt.subplots(figsize=(5.8, max(2.6, 0.27 * len(order) + 1.2)))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("PCA loadings (variables with the largest loadings)", fontsize=10)
    return _png(fig)


def distribution_grid(before: pl.DataFrame, after: pl.DataFrame, variables: list[tuple[str, str]], title: str = "") -> bytes:
    """Una colonna per variabile; riga 1 = prima del matching, riga 2 = dopo (pesato)."""
    n = len(variables)
    ncols = max(n, 3)   # larghezza costante anche se l'ultimo gruppo ha meno di 3 variabili
    fig, axes = plt.subplots(2, ncols, figsize=(2.65 * 3 + 0.6, 4.6), squeeze=False)
    for j in range(n, ncols):
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    for j, (var, kind) in enumerate(variables):
        ref = before[var].cast(pl.Float64).to_numpy()
        ref = ref[~np.isnan(ref)]
        for i, (df, row) in enumerate(((before, "Before"), (after, "After"))):
            ax = axes[i, j]
            g = df[GROUP_COL].to_numpy()
            w = df["weight"].to_numpy() if (i == 1 and "weight" in df.columns) else np.ones(df.height)
            x = df[var].cast(pl.Float64).to_numpy()
            ok = ~np.isnan(x)
            if kind == "categorical":
                levels = np.unique(ref)
                for k, (gr, col) in enumerate(((1, _C_T), (0, _C_C))):
                    m = (g == gr) & ok
                    tot = w[m].sum()
                    props = [w[m & (x == lv)].sum() / tot if tot else 0 for lv in levels]
                    ax.bar(np.arange(len(levels)) + (k - 0.5) * 0.38, props, 0.38, color=col, alpha=0.8)
                ax.set_xticks(np.arange(len(levels)))
                ax.set_xticklabels([str(int(l)) if float(l).is_integer() else f"{l:.1f}" for l in levels], fontsize=7)
                ax.set_ylim(0, 1)
            else:
                lo, hi = np.percentile(ref, [0.5, 99.5])
                bins = np.linspace(lo, hi, 24)
                for gr, col in ((1, _C_T), (0, _C_C)):
                    m = (g == gr) & ok
                    ax.hist(x[m], bins=bins, weights=w[m], density=True, histtype="stepfilled", alpha=0.45, color=col)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(var, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(row, fontsize=8.5)
    axes[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=_C_T, alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=_C_C, alpha=0.6)],
                      labels=["Treated", "Control"], fontsize=7, loc="upper right")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def _heat(ax, M: np.ndarray, names: list[str], vmax: float, title: str, annot: bool, cmap: str = "RdBu_r"):
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=5.5)
    return im


def corr_heatmaps(C: dict[str, np.ndarray], names: list[str], method: str) -> bytes:
    """C: chiavi t_before, c_before, t_after, c_after. Colonne: trattati | controlli | differenza."""
    annot = len(names) <= 12
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), constrained_layout=True)
    for r, tag in enumerate(("before", "after")):
        T, Cn = C[f"t_{tag}"], C[f"c_{tag}"]
        _heat(axes[r, 0], T, names, 1, f"Treated - {tag}", annot)
        im1 = _heat(axes[r, 1], Cn, names, 1, f"Control - {tag}", annot)
        im2 = _heat(axes[r, 2], T - Cn, names, 0.5, f"Treated - Control ({tag})", annot)
    fig.colorbar(im1, ax=axes[:, :2], shrink=0.7, pad=0.02, label=f"{method.title()} correlation")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.7, pad=0.02, label="Difference")
    fig.suptitle("Correlation structure of the analysis variables", fontsize=10)
    return _png(fig)


def corr_with_group(names: list[str], before: np.ndarray, after: np.ndarray, top: int = 30) -> bytes:
    order = np.argsort(-np.nan_to_num(np.abs(before)))[:top][::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.26 * len(order) + 1.2)))
    ax.barh(y + 0.2, before[order], 0.4, color="#9e9e9e", label="Before")
    ax.barh(y - 0.2, after[order], 0.4, color=_C_T, label="After")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("Correlation with group membership (treated = 1)")
    ax.set_title("Association of each variable with the group", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return _png(fig)


# --------------------------------------------------------------------------------------------------
# Propensity score: supporto comune / overlap e stime degli effetti
# --------------------------------------------------------------------------------------------------
def ps_support(
    ps_before: np.ndarray, g_before: np.ndarray, ps_after: np.ndarray, g_after: np.ndarray, w_after: np.ndarray,
    support: tuple[float, float], bins: int = 30, label: str = "Propensity score",
) -> bytes:
    """Istogrammi specchiati: trattati sopra l'asse, controlli sotto. Area ombreggiata = supporto comune."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharex=True)
    edges = np.linspace(min(ps_before.min(), ps_after.min()), max(ps_before.max(), ps_after.max()), bins + 1)
    ymax = 0.0
    panels = ((axes[0], ps_before, g_before, np.ones(len(ps_before)), "Before matching"),
              (axes[1], ps_after, g_after, w_after, "After matching (weighted)"))
    for ax, ps, g, w, title in panels:
        for gr, sign, col, name in ((1, 1, _C_T, "Treated"), (0, -1, _C_C, "Control")):
            m = g == gr
            h, _ = np.histogram(ps[m], bins=edges, weights=w[m], density=True)
            ax.bar(edges[:-1], sign * h, width=np.diff(edges), align="edge", color=col, alpha=0.75, label=name)
            ymax = max(ymax, h.max())
        ax.axvspan(support[0], support[1], color="#2e7d32", alpha=0.08, label="Common support")
        ax.axvline(support[0], color="#2e7d32", ls="--", lw=0.9)
        ax.axvline(support[1], color="#2e7d32", ls="--", lw=0.9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(label)
    for ax in axes:
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    axes[0].set_ylabel("Density (treated up, control down)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Propensity score overlap and common support", fontsize=10)
    fig.tight_layout()
    return _png(fig)


def effects_forest(table: pl.DataFrame, ci_level: float) -> bytes:
    t = table.filter(pl.col("measure") == "difference")
    labels = [f"{r['outcome']} | {r['estimand']}" + (" (naive)" if r["estimator"] == "naive" else "") for r in t.iter_rows(named=True)]
    est = t["estimate"].to_numpy()
    lo, hi = t["ci_low_pct"].to_numpy(), t["ci_high_pct"].to_numpy()
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.36 * len(labels) + 1.2)))
    for i, r in enumerate(t.iter_rows(named=True)):
        col = "#9e9e9e" if r["estimator"] == "naive" else _C_T
        ax.errorbar(est[i], y[i], xerr=[[est[i] - lo[i]], [hi[i] - est[i]]], fmt="o" if r["estimator"] != "naive" else "s",
                    color=col, capsize=3, ms=5)
    ax.axvline(0, color=_C_C, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Effect (treated - control), {100 * ci_level:.0f}% bootstrap percentile CI")
    ax.set_title("Treatment effect estimates", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    return _png(fig)


def effects_bootstrap(boot: dict, point: dict, keys: list, ci_level: float) -> bytes:
    n = len(keys)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 0.5, 2.5 * nrows + 0.4), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        arr = boot[k]
        ax.hist(arr, bins=25, color=_C_T, alpha=0.75)
        ax.axvline(point[k], color="black", lw=1.2)
        a = (1 - ci_level) / 2
        for q in np.quantile(arr, [a, 1 - a]):
            ax.axvline(q, color=_C_C, ls="--", lw=1)
        ax.set_title(f"{k[0]} | {k[1]}", fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle("Bootstrap distribution of the matched effect (black = estimate, red = CI)", fontsize=10)
    fig.tight_layout()
    return _png(fig)
