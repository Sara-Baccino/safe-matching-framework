"""Analisi multivariate pre/post matching: PCA, griglie di distribuzioni, correlazioni.

Le variabili analizzate sono quelle di ``variables`` (matching + stratificazione + evaluation); le
categoriche con >2 livelli sono espanse in indicatori one-hot (livelli fissati sulla popolazione).
Il dopo-matching usa i pesi ``weight`` (coppie 1:K e selezione OT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from safe_matching.evaluation import plots
from safe_matching.evaluation.balance import _smd, _wmean_var
from safe_matching.preprocessing.assembly import GROUP_COL
from safe_matching.schemas.matching_config import AnalysisConfig

__all__ = ["run_analysis", "AnalysisResult"]


@dataclass
class AnalysisResult:
    figures: list[tuple[str, str, bytes]] = field(default_factory=list)
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _design(df: pl.DataFrame, cols: list[str], kinds: dict[str, str], levels: dict[str, list]) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for c in cols:
        if c in levels:
            for lv in levels[c]:
                parts.append((df[c] == lv).cast(pl.Float64).to_numpy())
                names.append(f"{c}={lv:g}" if isinstance(lv, float) else f"{c}={lv}")
        else:
            parts.append(df[c].cast(pl.Float64).to_numpy())
            names.append(c)
    return np.column_stack(parts), names


def _weighted_corr(X: np.ndarray, w: np.ndarray, method: str) -> np.ndarray:
    if method == "spearman":
        X = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    sw = w.sum()
    m = (w[:, None] * X).sum(0) / sw
    Xc = X - m
    cov = (Xc * w[:, None]).T @ Xc / sw
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.outer(sd, sd)


def _wcorr_with(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m1, v1 = _wmean_var(x, w)
    m2, v2 = _wmean_var(y, w)
    if not (v1 > 0 and v2 > 0):
        return float("nan")
    sw = w.sum()
    return float((w * (x - m1) * (y - m2)).sum() / sw / np.sqrt(v1 * v2 * ((sw - (w ** 2).sum() / sw) / sw) ** 2))


def run_analysis(
    population: pl.DataFrame,
    matched: pl.DataFrame,
    variables: dict[str, str],
    kinds: dict[str, str],
    cfg: AnalysisConfig,
    *,
    smd_threshold: float,
    plots_enabled: bool = True,
    max_distribution_variables: int = 12,
    ranked_variables: list[str] | None = None,
    ps_data: tuple[pl.DataFrame, pl.DataFrame, str] | None = None,
) -> AnalysisResult:
    res = AnalysisResult()
    cols = [c for c in variables if c in population.columns and c in matched.columns
            and (population.schema[c].is_numeric() or population.schema[c] == pl.Boolean)]
    if len(cols) < 2:
        res.notes.append("Multivariate analyses skipped: at least 2 numeric/binary variables are needed.")
        return res
    levels = {c: sorted(population[c].drop_nulls().unique().to_list())[1:] for c in cols
              if kinds.get(c) == "categorical" and population[c].drop_nulls().n_unique() > 2}
    Xb, names = _design(population, cols, kinds, levels)
    Xa, _ = _design(matched, cols, kinds, levels)
    ok_b, ok_a = ~np.isnan(Xb).any(1), ~np.isnan(Xa).any(1)
    gb, ga = population[GROUP_COL].to_numpy()[ok_b], matched[GROUP_COL].to_numpy()[ok_a]
    Xb, Xa = Xb[ok_b], Xa[ok_a]
    wb = np.ones(len(Xb))
    wa = matched["weight"].to_numpy().astype(float)[ok_a] if "weight" in matched.columns else np.ones(len(Xa))

    # ----------------------------------------------------------------- PCA
    if cfg.pca:
        mu, sd = Xb.mean(0), Xb.std(0, ddof=1)
        keep = sd > 0
        if keep.sum() >= 2:
            Zb, Za = (Xb[:, keep] - mu[keep]) / sd[keep], (Xa[:, keep] - mu[keep]) / sd[keep]
            _, svals, Vt = np.linalg.svd(Zb, full_matrices=False)
            evr = svals ** 2 / (svals ** 2).sum()
            Sb, Sa = Zb @ Vt.T, Za @ Vt.T
            k = int(min(10, len(evr), max(2, np.searchsorted(np.cumsum(evr), cfg.pca_variance_target) + 1)))
            smd_b = np.array([_smd(*_wmean_var(Sb[gb == 1, j], wb[gb == 1]), *_wmean_var(Sb[gb == 0, j], wb[gb == 0])) for j in range(k)])
            smd_a = np.array([_smd(*_wmean_var(Sa[ga == 1, j], wa[ga == 1]), *_wmean_var(Sa[ga == 0, j], wa[ga == 0])) for j in range(k)])
            res.metrics.update({
                "pca_components_used": k, "pca_explained_variance_used": float(evr[:k].sum()),
                "pca_smd_euclid_before": float(np.sqrt((smd_b ** 2).sum())), "pca_smd_euclid_after": float(np.sqrt((smd_a ** 2).sum())),
                "pca_max_abs_smd_before": float(np.abs(smd_b).max()), "pca_max_abs_smd_after": float(np.abs(smd_a).max()),
            })
            res.tables["pca_variance"] = pl.DataFrame({
                "component": [f"PC{j + 1}" for j in range(k)], "explained_variance": evr[:k], "cumulative": np.cumsum(evr)[:k],
                "smd_before": smd_b, "smd_after": smd_a,
            })
            nl = min(cfg.pca_loadings_components, Vt.shape[0])
            L = Vt[:nl].T
            kn = [n for n, kp in zip(names, keep) if kp]
            res.tables["pca_loadings"] = pl.DataFrame({"variable": kn, **{f"PC{j + 1}": L[:, j] for j in range(nl)}})
            if plots_enabled:
                res.figures.append(("pca_scatter", "First two principal components (PCA fitted on the full eligible population, "
                                    "standardized variables); the matched data is projected on the same axes.",
                                    plots.pca_scatter(Sb, gb, wb, Sa, ga, wa, evr, cfg.max_scatter_points)))
                res.figures.append(("pca_variance", "Explained variance per component and group imbalance (|SMD|) of the component scores.",
                                    plots.pca_variance(evr, smd_b, smd_a, smd_threshold)))
                res.figures.append(("pca_loadings", "Loadings of the first components: which variables drive each axis.",
                                    plots.pca_loadings(L, kn, [f"PC{j + 1}" for j in range(nl)])))
        else:
            res.notes.append("PCA skipped: fewer than 2 variables with non-zero variance.")

    # ------------------------------------------------------- distribuzioni
    if cfg.distributions and plots_enabled and max_distribution_variables > 0:
        order = [c for c in (ranked_variables or cols) if c in cols][:max_distribution_variables]
        vk = [(c, kinds.get(c, "numerical")) for c in order]
        for i in range(0, len(vk), 3):
            chunk = vk[i:i + 3]
            res.figures.append((f"dist_grid_{i // 3 + 1}",
                                f"Distributions by group before (top row) and after (bottom row) matching: "
                                f"{', '.join(c for c, _ in chunk)}.",
                                plots.distribution_grid(population, matched, chunk)))

    # --------------------------------------------------------- correlazioni
    if cfg.correlations:
        sel = list(range(len(names)))
        if len(sel) > cfg.max_correlation_variables:
            res.notes.append(f"Correlations limited to the first {cfg.max_correlation_variables} of {len(names)} columns.")
            sel = sel[: cfg.max_correlation_variables]
        nm = [names[i] for i in sel]
        Xb_, Xa_ = Xb[:, sel], Xa[:, sel]
        C = {
            "t_before": _weighted_corr(Xb_[gb == 1], wb[gb == 1], cfg.correlation_method),
            "c_before": _weighted_corr(Xb_[gb == 0], wb[gb == 0], cfg.correlation_method),
            "t_after": _weighted_corr(Xa_[ga == 1], wa[ga == 1], cfg.correlation_method),
            "c_after": _weighted_corr(Xa_[ga == 0], wa[ga == 0], cfg.correlation_method),
        }
        iu = np.triu_indices(len(nm), 1)
        d_b = np.abs(C["t_before"] - C["c_before"])[iu]
        d_a = np.abs(C["t_after"] - C["c_after"])[iu]
        res.metrics.update({
            "corr_mean_abs_diff_before": float(np.nanmean(d_b)), "corr_mean_abs_diff_after": float(np.nanmean(d_a)),
            "corr_max_abs_diff_before": float(np.nanmax(d_b)), "corr_max_abs_diff_after": float(np.nanmax(d_a)),
            "corr_pct_pairs_close_after": float(np.nanmean(d_a < 0.1) * 100), "corr_n_pairs": int(d_a.size),
        })
        order = np.argsort(-np.nan_to_num(d_a))[:15]
        res.tables["correlation_pairs"] = pl.DataFrame({
            "variable_1": [nm[iu[0][o]] for o in order], "variable_2": [nm[iu[1][o]] for o in order],
            "corr_treated_after": [float(C["t_after"][iu][o]) for o in order],
            "corr_control_after": [float(C["c_after"][iu][o]) for o in order],
            "abs_diff_before": [float(d_b[o]) for o in order], "abs_diff_after": [float(d_a[o]) for o in order],
        })
        cg_b = np.array([_wcorr_with(Xb_[:, j], gb.astype(float), wb) for j in range(len(nm))])
        cg_a = np.array([_wcorr_with(Xa_[:, j], ga.astype(float), wa) for j in range(len(nm))])
        res.tables["correlation_with_group"] = pl.DataFrame({"variable": nm, "corr_before": cg_b, "corr_after": cg_a})
        if plots_enabled:
            res.figures.append(("corr_heatmaps", f"{cfg.correlation_method.title()} correlations among the analysis variables in treated and "
                                f"control units, before and after matching, and their difference.",
                                plots.corr_heatmaps(C, nm, cfg.correlation_method)))
            res.figures.append(("corr_group", "Correlation of each variable with group membership; values close to zero after matching "
                                "indicate that the group can no longer be predicted from that variable.",
                                plots.corr_with_group(nm, cg_b, cg_a)))

    # -------------------------------------------- propensity score: overlap
    if cfg.ps_support and ps_data is not None:
        _ps_support(res, ps_data, cfg, plots_enabled)
    return res


def _ps_support(res: AnalysisResult, ps_data, cfg: AnalysisConfig, plots_enabled: bool) -> None:
    pop, matched, col = ps_data
    g_b, ps_b = pop[GROUP_COL].to_numpy(), pop[col].to_numpy().astype(float)
    g_a, ps_a = matched[GROUP_COL].to_numpy(), matched[col].to_numpy().astype(float)
    w_a = matched["weight"].to_numpy().astype(float) if "weight" in matched.columns else np.ones(len(ps_a))
    t, c = ps_b[g_b == 1], ps_b[g_b == 0]
    lo, hi = max(t.min(), c.min()), min(t.max(), c.max())
    edges = np.linspace(ps_b.min(), ps_b.max(), cfg.ps_support_bins + 1)

    def ovl(p_, g_, w_):
        ht, _ = np.histogram(p_[g_ == 1], edges, weights=w_[g_ == 1])
        hc, _ = np.histogram(p_[g_ == 0], edges, weights=w_[g_ == 0])
        return float(np.minimum(ht / ht.sum(), hc / hc.sum()).sum())

    m = {
        "ps_auc": float(roc_auc_score(g_b, ps_b)),
        "ps_support_low": float(lo), "ps_support_high": float(hi),
        "ps_pct_treated_outside_support": float(np.mean((t < lo) | (t > hi)) * 100),
        "ps_pct_control_outside_support": float(np.mean((c < lo) | (c > hi)) * 100),
        "ps_overlap_coefficient_before": ovl(ps_b, g_b, np.ones(len(ps_b))),
        "ps_overlap_coefficient_after": ovl(ps_a, g_a, w_a),
        "ps_min_treated": float(t.min()), "ps_max_treated": float(t.max()),
        "ps_min_control": float(c.min()), "ps_max_control": float(c.max()),
    }
    res.metrics.update(m)
    ps_b_ = pop[col].to_numpy().astype(float)
    res.tables["ps_overlap_summary"] = pl.DataFrame({
        "group": ["treated", "control"], "n": [len(t), len(c)], "ps_min": [t.min(), c.min()],
        "ps_q05": [np.quantile(t, .05), np.quantile(c, .05)], "ps_median": [np.median(t), np.median(c)],
        "ps_q95": [np.quantile(t, .95), np.quantile(c, .95)], "ps_max": [t.max(), c.max()],
        "pct_outside_common_support": [m["ps_pct_treated_outside_support"], m["ps_pct_control_outside_support"]],
    })
    if plots_enabled:
        label = "logit(PS)" if "logit" in col else "Propensity score"
        res.figures.append(("ps_support", "Mirrored histograms of the propensity score: treated above and control below the axis. "
                            "The shaded band is the common support [max of the group minima, min of the group maxima].",
                            plots.ps_support(ps_b, g_b, ps_a, g_a, w_a, (lo, hi), cfg.ps_support_bins, label)))
