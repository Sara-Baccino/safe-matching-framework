"""Stima di ATT / ATC / ATE dopo il matching, con errori standard e intervalli da bootstrap.

Stimatore: differenza tra la media pesata dell'outcome nei trattati e nei controlli del dataset abbinato
(pesi ``weight``: 1 per le unita' query, 1/K per i K controlli di una coppia, massa OT per la selezione).
* ATT = matching treated_to_control, ATC = matching control_to_treated.
* ATE: E[Y(1)] e E[Y(0)] sono la media di quelle dei due matching pesata per il numero di unita' query abbinate.

Bootstrap (full_pipeline): si ricampionano le unita' CON reinserimento dentro ciascun gruppo e si RIESEGUE tutto
(stima del PS inclusa e matching). Avvertenza: Abadie & Imbens (2008) mostrano che il bootstrap non e' formalmente
valido per il matching nearest-neighbour con numero fisso di vicini; qui e' un'approssimazione pratica che tiene
conto della variabilita' del matching, da leggere come indicativa (confronta con 'matched_pairs' e con il naive).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from safe_matching.exceptions import ConfigError, MatchingError
from safe_matching.matching import MatchingEngine
from safe_matching.preprocessing.assembly import GROUP_COL, UID_COL
from safe_matching.schemas.matching_config import EstimationConfig, MatchingConfig

__all__ = ["estimate_effects", "EstimationResult"]
log = logging.getLogger("safe_matching")

_DIR = {"ATT": "treated_to_control", "ATC": "control_to_treated"}
Key = tuple[str, str, str, str]  # (outcome, estimand, estimator, measure)


@dataclass
class EstimationResult:
    table: pl.DataFrame
    boot: dict[Key, np.ndarray] = field(default_factory=dict)
    point: dict[Key, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
def _wmean(y: np.ndarray, w: np.ndarray) -> float:
    sw = w.sum()
    return float((w * y).sum() / sw) if sw > 0 else float("nan")


def _group_stats(matched: pl.DataFrame, outcome: str) -> tuple[float, float, int, int, int]:
    """(media trattati, media controlli, n query, n righe trattati, n righe controlli) sul dataset abbinato."""
    g = matched[GROUP_COL].to_numpy()
    y = matched[outcome].cast(pl.Float64).to_numpy()
    w = matched["weight"].to_numpy().astype(float)
    ok = ~np.isnan(y)
    nq = int((matched["match_role"] == "query").sum())
    t, c = (g == 1) & ok, (g == 0) & ok
    return _wmean(y[t], w[t]), _wmean(y[c], w[c]), nq, int((g == 1).sum()), int((g == 0).sum())


def _combine(stats: dict[str, tuple], outcome: str, kind: str, estimands: list[str]) -> dict[Key, float]:
    """stats: {direzione: (mt, mc, nq, ...)} -> stime per estimando e misura."""
    out: dict[Key, float] = {}

    def put(est: str, e1: float, e0: float) -> None:
        out[(outcome, est, "matched", "difference")] = e1 - e0
        if kind == "binary":
            out[(outcome, est, "matched", "risk_ratio")] = e1 / e0 if e0 > 0 else float("nan")

    for est in ("ATT", "ATC"):
        if est in estimands:
            mt, mc = stats[_DIR[est]][:2]
            put(est, mt, mc)
    if "ATE" in estimands:
        (mt_a, mc_a, na), (mt_b, mc_b, nb) = (stats[d][:3] for d in ("treated_to_control", "control_to_treated"))
        tot = na + nb
        put("ATE", (na * mt_a + nb * mt_b) / tot, (na * mc_a + nb * mc_b) / tot)
    return out


def _directions(estimands: list[str]) -> list[str]:
    d: set[str] = set()
    for e in estimands:
        d |= {"treated_to_control", "control_to_treated"} if e == "ATE" else {_DIR[e]}
    return sorted(d)


def _naive(pop: pl.DataFrame, outcome: str, kind: str) -> dict[Key, float]:
    g = pop[GROUP_COL].to_numpy()
    y = pop[outcome].cast(pl.Float64).to_numpy()
    ok = ~np.isnan(y)
    m1, m0 = y[(g == 1) & ok].mean(), y[(g == 0) & ok].mean()
    out = {(outcome, "unadjusted", "naive", "difference"): float(m1 - m0)}
    if kind == "binary":
        out[(outcome, "unadjusted", "naive", "risk_ratio")] = float(m1 / m0) if m0 > 0 else float("nan")
    return out


def _run_dir(pop: pl.DataFrame, cfg_dict: dict, kinds: dict, direction: str, seed: int) -> pl.DataFrame:
    c = MatchingConfig.from_dict(cfg_dict)
    c.population.direction = direction  # type: ignore[assignment]
    c.options.random_state = seed
    return MatchingEngine(c, kinds).run(pop).matched


def _replicate(pop, cfg_dict, kinds, outcomes, estimands, include_naive, seed):
    """Una replica full_pipeline; ritorna {Key: valore} oppure None se il matching fallisce."""
    rng = np.random.default_rng(seed)
    g = pop[GROUP_COL].to_numpy()
    it, ic = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    sel = np.concatenate([rng.choice(it, len(it)), rng.choice(ic, len(ic))])
    boot = pop[sel].with_columns(pl.Series(UID_COL, np.arange(len(sel)), dtype=pl.Int64))
    try:
        matched = {d: _run_dir(boot, cfg_dict, kinds, d, seed) for d in _directions(estimands)}
    except (MatchingError, ConfigError, ValueError):
        return None
    res: dict[Key, float] = {}
    for o in outcomes:
        st = {d: _group_stats(m, o["variable"]) for d, m in matched.items()}
        res.update(_combine(st, o["variable"], o["type"], estimands))
        if include_naive:
            res.update(_naive(boot, o["variable"], o["type"]))
    return res


# ---- bootstrap sulle coppie gia' formate ----------------------------------------------------------
def _pair_arrays(matched: pl.DataFrame, outcome: str):
    q = matched.filter(pl.col("match_role") == "query").select(["pair_id", GROUP_COL, outcome])
    p = matched.filter(pl.col("match_role") == "match").select(["pair_id", outcome, "weight"])
    p = p.filter(pl.col(outcome).is_not_null()).with_columns((pl.col(outcome) * pl.col("weight")).alias("_s"))
    agg = p.group_by("pair_id").agg(pl.col("_s").sum().alias("S"), pl.col("weight").sum().alias("W"))
    d = q.filter(pl.col(outcome).is_not_null()).join(agg, on="pair_id", how="inner").filter(pl.col("W") > 0)
    qgroup = int(d[GROUP_COL][0]) if d.height else 1
    return d[outcome].to_numpy().astype(float), d["S"].to_numpy(), d["W"].to_numpy(), qgroup


def _pairs_replicate(arrays, outcomes, estimands, include_naive, naive_pop, rng):
    res: dict[Key, float] = {}
    for o in outcomes:
        name = o["variable"]
        st = {}
        for d, (yq, S, W, qgroup) in arrays[name].items():
            c = np.bincount(rng.integers(0, len(yq), len(yq)), minlength=len(yq)).astype(float)
            mq, mp = (c * yq).sum() / c.sum(), (c * S).sum() / (c * W).sum()
            st[d] = (mq, mp, int(c.sum())) if d == "treated_to_control" else (mp, mq, int(c.sum()))
        res.update(_combine(st, name, o["type"], estimands))
    return res


# --------------------------------------------------------------------------------------------------
def estimate_effects(
    population: pl.DataFrame,
    cfg: MatchingConfig,
    kinds: dict[str, str],
    base_matched: dict[str, pl.DataFrame],
    *,
    common_columns: list[str] | None = None,
) -> EstimationResult:
    """``base_matched``: {direzione: dataset abbinato} gia' calcolati (almeno quello della config)."""
    ec: EstimationConfig = cfg.estimation
    outcomes = [{"variable": o.variable, "type": o.type, "label": o.label or o.variable} for o in ec.outcomes]
    estimands = list(ec.estimands)
    notes: list[str] = []
    for o in outcomes:
        v = o["variable"]
        if v not in population.columns:
            raise ConfigError(f"Outcome '{v}' assente dal dataset di lavoro (colonne: {[c for c in population.columns if not c.startswith('__')][:20]}).")
        if common_columns is not None and v not in common_columns:
            raise ConfigError(f"Outcome '{v}' non presente in entrambi i dataset.")
        if o["type"] == "binary":
            vals = set(population[v].drop_nulls().cast(pl.Float64).unique().to_list())
            if not vals <= {0.0, 1.0}:
                raise ConfigError(f"Outcome binario '{v}' deve contenere solo 0/1 (valori: {sorted(vals)[:6]}).")
        if population[v].null_count():
            notes.append(f"Outcome '{v}': {population[v].null_count()} missing values, excluded from the means (if the data config imputes this "
                         "column, imputed values are treated as observed: use strategy 'maintain' for outcomes).")

    dirs = _directions(estimands)
    cfg_dict = cfg.to_dict()
    matched = dict(base_matched)
    seed0 = ec.bootstrap.random_state if ec.bootstrap.random_state is not None else (cfg.options.random_state or 0)
    for d in dirs:
        if d not in matched:
            matched[d] = _run_dir(population, cfg_dict, kinds, d, seed0)

    # ---- stime puntuali ---------------------------------------------------------------------------
    point: dict[Key, float] = {}
    sizes: dict[tuple[str, str], tuple[int, int, int | None, int | None]] = {}
    n_t, n_c = int((population[GROUP_COL] == 1).sum()), int((population[GROUP_COL] == 0).sum())
    group_total = {"treated_to_control": n_t, "control_to_treated": n_c}
    low_cov: set[str] = set()
    for o in outcomes:
        st = {d: _group_stats(m, o["variable"]) for d, m in matched.items()}
        point.update(_combine(st, o["variable"], o["type"], estimands))
        for est in estimands:
            ds = dirs if est == "ATE" else [_DIR[est]]
            nq, tot = sum(st[d][2] for d in ds), sum(group_total[d] for d in ds)
            sizes[(o["variable"], est)] = (sum(st[d][3] for d in ds), sum(st[d][4] for d in ds), nq, tot)
            if nq / tot < 0.9:
                low_cov.add(est)
        if ec.include_naive:
            point.update(_naive(population, o["variable"], o["type"]))
            sizes[(o["variable"], "unadjusted")] = (n_t, n_c, None, None)
    if low_cov:
        notes.append(
            f"Coverage < 90% of the reference units for {sorted(low_cov)}: the estimate refers to the matched SUBSET, not to the whole group. "
            "With methods without replacement (hungarian, sorted, greedy_nn without replacement) and unbalanced groups, ATC/ATE are not "
            "identified on the whole population (hungarian returns the same pairing in both directions): for ATC/ATE use greedy_nn with "
            "replacement=true, or relax the calipers.")

    # ---- bootstrap ----------------------------------------------------------------------------------
    B = ec.bootstrap.n_boot
    mode = ec.bootstrap.resample
    if mode == "matched_pairs" and any(m["match_role"].is_in(["selected"]).any() for m in matched.values()):
        notes.append("matched_pairs is not applicable to selection methods (partial_ot): full_pipeline used.")
        mode = "full_pipeline"
    reps: list[dict[Key, float] | None]
    if mode == "full_pipeline":
        args = [(population, cfg_dict, kinds, outcomes, estimands, ec.include_naive, seed0 + 1 + b) for b in range(B)]
        if ec.bootstrap.n_jobs == 1:
            reps = []
            for b, a in enumerate(args):
                reps.append(_replicate(*a))
                if (b + 1) % max(1, B // 10) == 0:
                    log.info("bootstrap %d/%d", b + 1, B)
        else:
            from joblib import Parallel, delayed
            reps = Parallel(n_jobs=ec.bootstrap.n_jobs)(delayed(_replicate)(*a) for a in args)
    else:
        arrays = {o["variable"]: {d: _pair_arrays(matched[d], o["variable"]) for d in dirs} for o in outcomes}
        rng = np.random.default_rng(seed0 + 1)
        reps = []
        for _ in range(B):
            r = _pairs_replicate(arrays, outcomes, estimands, ec.include_naive, population, rng)
            if ec.include_naive:
                g = population[GROUP_COL].to_numpy()
                sel = np.concatenate([rng.choice(np.flatnonzero(g == 1), (g == 1).sum()), rng.choice(np.flatnonzero(g == 0), (g == 0).sum())])
                for o in outcomes:
                    r.update(_naive(population[sel], o["variable"], o["type"]))
            reps.append(r)

    ok = [r for r in reps if r is not None]
    n_ok = len(ok)
    if n_ok < max(10, B // 2):
        raise MatchingError(f"Bootstrap: solo {n_ok}/{B} repliche valide (matching fallito nelle altre): impossibile stimare gli errori.")
    if n_ok < B:
        notes.append(f"Bootstrap: {B - n_ok} of {B} replicates discarded because matching failed.")
    notes.append("Bootstrap standard errors and intervals (" + ("units resampled with replacement; PS estimation and matching are repeated at every replicate"
                 if mode == "full_pipeline" else "matched pairs resampled; matching variability is not included") +
                 "). With nearest-neighbour matching the bootstrap is not formally valid (Abadie & Imbens, 2008): read as indicative.")

    alpha = 1 - ec.bootstrap.ci_level
    z = norm.ppf(1 - alpha / 2)
    boot: dict[Key, np.ndarray] = {}
    rows = []
    for key, est in point.items():
        arr = np.array([r[key] for r in ok if key in r], dtype=float)
        arr = arr[np.isfinite(arr)]
        boot[key] = arr
        se = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        lo, hi = (np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if arr.size > 1 else (np.nan, np.nan))
        null = 1.0 if key[3] == "risk_ratio" else 0.0
        zz = (est - null) / se if se and se > 0 else float("nan")
        outc, estimand, estimator, measure = key
        nt, nc, nqm, nqt = sizes[(outc, estimand)]
        rows.append({
            "outcome": outc, "estimand": estimand, "estimator": estimator, "measure": measure, "estimate": est,
            "se_boot": se, "ci_low_pct": float(lo), "ci_high_pct": float(hi),
            "ci_low_norm": est - z * se, "ci_high_norm": est + z * se,
            "p_value": float(2 * (1 - norm.cdf(abs(zz)))) if np.isfinite(zz) else float("nan"),
            "n_rows_treated": nt, "n_rows_control": nc, "n_query_matched": nqm, "n_query_total": nqt,
            "coverage": (nqm / nqt) if nqt else None, "n_boot_ok": int(arr.size),
        })
    table = pl.DataFrame(rows)
    return EstimationResult(table, boot, point, notes, {"n_boot": B, "n_boot_ok": n_ok, "resample": mode, "ci_level": ec.bootstrap.ci_level})
