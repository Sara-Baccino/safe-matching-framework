"""Motore di matching: orchestra stratificazione, rappresentazione, distanza, vincoli e algoritmo."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from safe_matching.exceptions import ConfigError, MatchingError
from safe_matching.matching.algorithms import ALGORITHMS, build_algorithm
from safe_matching.matching.algorithms.base import RawMatch
from safe_matching.matching.constraints import PENALTY, PENALTY_THRESHOLD, apply_mask, partition_strata, window_mask
from safe_matching.matching.distances import build_distance
from safe_matching.matching.representation import design_matrix, fit_propensity
from safe_matching.preprocessing.assembly import GROUP_COL
from safe_matching.schemas.matching_config import MatchingConfig
from safe_matching.schemas.method_params import (
    METHOD_PARAMS,
    PropensityScoreParams,
    SortedParams,
)

__all__ = ["MatchingEngine", "MatchingOutcome", "PS_COL", "LOGIT_PS_COL"]

PS_COL = "__ps__"
LOGIT_PS_COL = "__logit_ps__"
_PAIR, _WEIGHT, _DIST, _ROLE, _STRATUM = "pair_id", "weight", "distance", "match_role", "stratum"


@dataclass
class MatchingOutcome:
    matched: pl.DataFrame
    """Righe abbinate: unita' query (match_role='query'), controparti ('match') o selezionate ('selected')."""
    population: pl.DataFrame
    """Popolazione eleggibile PRIMA del matching (dopo l'esclusione dei mancanti)."""
    unmatched_query: pl.DataFrame
    strata: pl.DataFrame
    metrics: dict[str, float | int | str | bool]
    method: str
    algorithm: str
    metric_name: str
    feature_columns: list[str]
    output_type: str
    query_label: str
    pool_label: str
    notes: list[str] = field(default_factory=list)
    ps_model: Any = None
    has_ps: bool = False


class MatchingEngine:
    def __init__(self, cfg: MatchingConfig, kinds: dict[str, str]) -> None:
        self.cfg = cfg
        self.kinds = dict(kinds)
        self.rng = np.random.default_rng(cfg.options.random_state)
        self._notes: list[str] = []

    # ------------------------------------------------------------------ #
    def run(self, df: pl.DataFrame) -> MatchingOutcome:
        cfg, kinds = self.cfg, self.kinds
        method = cfg.method.name
        v = cfg.variables
        calipers = cfg.constraints.calipers

        sort_var: str | None = None
        if method == "sorted":
            sp: SortedParams = cfg.method.parsed()  # type: ignore[assignment]
            sort_var = sp.sort_variable
            if sort_var is None:
                if len(v.matching) != 1:
                    raise ConfigError(
                        "method 'sorted' lavora su UNA variabile: indica params.sort_variable oppure usa un solo "
                        "matching variable (o 'propensity_score' con algorithm='sorted' per piu' variabili)."
                    )
                sort_var = v.matching[0]
            if sort_var not in v.matching:
                raise ConfigError(f"sort_variable '{sort_var}' deve essere tra variables.matching.")
            if len(v.matching) > 1:
                self._notes.append(
                    f"'sorted' ordina su '{sort_var}'; le altre variabili di matching sono usate solo per le diagnostiche."
                )

        self._check_columns(df, sort_var)

        # ---- eleggibilita' (mancanti) --------------------------------------
        needed = list(dict.fromkeys(v.matching + v.stratification + [c.variable for c in calipers]))
        valid = pl.all_horizontal([pl.col(c).is_not_null() & ~pl.col(c).is_nan() if df.schema[c].is_float() else pl.col(c).is_not_null() for c in needed])
        n_missing = int(df.filter(~valid).height)
        if n_missing and not cfg.options.drop_rows_with_missing:
            raise MatchingError(f"{n_missing} righe con mancanti nelle variabili di matching (options.drop_rows_with_missing=false).")
        pop = df.filter(valid)
        if n_missing:
            self._notes.append(f"{n_missing} righe escluse per valori mancanti nelle variabili di matching/stratificazione/caliper.")
        if pop.filter(pl.col(GROUP_COL) == 1).height == 0 or pop.filter(pl.col(GROUP_COL) == 0).height == 0:
            raise MatchingError("Dopo l'esclusione dei mancanti un gruppo e' vuoto.")

        work = pop
        n_trimmed = 0
        ps_model = None
        has_ps = False
        feature_cols: list[str]
        feature_kinds: dict[str, str]
        algo_name: str
        algo_params: Any

        # ---- rappresentazione ----------------------------------------------
        if method == "propensity_score":
            psp: PropensityScoreParams = cfg.method.parsed()  # type: ignore[assignment]
            cov = list(v.matching) + ([c for c in v.stratification if c not in v.matching] if psp.include_stratification else [])
            X, _ = design_matrix(work, cov, kinds)
            y = work[GROUP_COL].to_numpy().astype(int)
            ps, lg, ps_model = fit_propensity(X, y, psp, cfg.options.random_state)
            work = work.with_columns(pl.Series(PS_COL, ps), pl.Series(LOGIT_PS_COL, lg))
            has_ps = True
            pop = pop.with_columns(pl.Series(PS_COL, ps), pl.Series(LOGIT_PS_COL, lg))
            work, n_trimmed = self._trim(work, psp)
            feature_cols = [LOGIT_PS_COL if psp.match_on == "logit" else PS_COL]
            feature_kinds = {feature_cols[0]: "numerical"}
            algo_name = psp.algorithm
            algo_params = METHOD_PARAMS[algo_name].model_validate(psp.algorithm_params)
            metric_name = "euclidean (1-D su " + feature_cols[0].strip("_") + ")"
            feature_metric = build_distance("euclidean", {"scale": "none"})
        elif method == "sorted":
            feature_cols = [sort_var]  # type: ignore[list-item]
            feature_kinds = {sort_var: kinds.get(sort_var, "numerical")}  # type: ignore[index]
            algo_name, algo_params = "sorted", cfg.method.parsed()
            metric_name = f"|x_q - x_p| su {sort_var}"
            feature_metric = build_distance("euclidean", {"scale": "none"})
        else:
            feature_cols = list(v.matching)
            feature_kinds = {c: kinds[c] for c in feature_cols}
            algo_name, algo_params = method, cfg.method.parsed()
            metric_name = cfg.metrics.name
            feature_metric = build_distance(cfg.metrics.name, cfg.metrics.params)
            self._warn_multilevel(work, feature_cols)
        if method in ("propensity_score", "sorted") and cfg.metrics.name != "euclidean":
            self._notes.append(
                f"La metrica '{cfg.metrics.name}' e' ignorata: con '{method}' il matching e' 1-D su "
                f"{feature_cols[0]} (distanza assoluta)."
            )
        if algo_name not in ALGORITHMS:
            raise MatchingError(f"Algoritmo '{algo_name}' non registrato.")

        # ---- statistiche globali -------------------------------------------
        direction_tq = cfg.population.direction == "treated_to_control"
        q_group = 1 if direction_tq else 0
        is_query = (work[GROUP_COL].to_numpy() == q_group)
        Xall = work.select(feature_cols).cast(pl.Float64).to_numpy()
        feature_metric.fit(Xall, feature_cols, feature_kinds, is_query)
        sd_vars = {w.variable: float(work[w.variable].cast(pl.Float64).std() or 0.0) for w in cfg.constraints.calipers if kinds.get(w.variable) != "categorical"}
        cap = self._global_caliper(work, feature_cols, feature_metric, is_query, algo_name, cfg, method, cfg.method.parsed() if method == "propensity_score" else None)

        # ---- strati ----------------------------------------------------------
        strata_frames = partition_strata(work, v.stratification)
        plan: list[tuple[str, pl.DataFrame, pl.DataFrame]] = []
        records: list[dict[str, Any]] = []
        for key, part in strata_frames.items():
            q_df = part.filter(pl.col(GROUP_COL) == q_group)
            p_df = part.filter(pl.col(GROUP_COL) != q_group)
            if q_df.height == 0 or p_df.height == 0:
                records.append({"stratum": key, "n_query": q_df.height, "n_pool": p_df.height, "n_query_matched": 0,
                                "n_pool_rows": 0, "status": "skipped", "note": "un gruppo assente nello strato"})
                continue
            plan.append((key, q_df, p_df))
        if not plan:
            raise MatchingError("Nessuno strato contiene unita' di entrambi i gruppi.")

        targets = self._ot_targets(plan, algo_params) if algo_name == "partial_ot" else {}

        frames: list[pl.DataFrame] = []
        unmatched_frames: list[pl.DataFrame] = []
        pair_offset = 0
        output_type = "selection" if algo_name == "partial_ot" else "pairs"
        all_dist: list[np.ndarray] = []
        q_label = "treated" if direction_tq else "control"
        p_label = "control" if direction_tq else "treated"

        for key, q_df, p_df in plan:
            Xq = q_df.select(feature_cols).cast(pl.Float64).to_numpy()
            Xp = p_df.select(feature_cols).cast(pl.Float64).to_numpy()
            cost = feature_metric.pairwise(Xq, Xp)
            if calipers:
                cost = apply_mask(cost, window_mask(q_df, p_df, calipers, kinds, sd_vars))
            if cap is not None:
                cost = np.where((cost > cap) & (cost < PENALTY_THRESHOLD), PENALTY, cost)

            p_sub = p_df
            cand: np.ndarray | None = None
            pk = getattr(algo_params, "prefilter_k", None)
            if pk:
                need = targets.get(key, q_df.height) if algo_name == "partial_ot" else q_df.height
                cand = self._prefilter(cost, pk, need)
                cost = cost[:, cand]
                p_sub = p_df[cand]

            aux: dict[str, Any] = {}
            if algo_name == "sorted":
                scol = feature_cols[0]
                aux["sort_q"] = q_df[scol].cast(pl.Float64).to_numpy()
                aux["sort_p"] = p_sub[scol].cast(pl.Float64).to_numpy()
            if algo_name == "partial_ot":
                aux["target_size"] = min(targets[key], p_sub.height)

            raw = build_algorithm(algo_name, algo_params, self.rng).match(cost, aux)

            if raw.q_idx.size == 0:
                records.append({"stratum": key, "n_query": q_df.height, "n_pool": p_df.height, "n_query_matched": 0,
                                "n_pool_rows": 0, "status": "no_match", "note": "nessuna coppia ammissibile"})
                unmatched_frames.append(q_df.with_columns(pl.lit(key).alias(_STRATUM)))
                continue

            block, n_q_matched, offset_inc = self._assemble(q_df, p_sub, raw, key, pair_offset)
            pair_offset += offset_inc
            frames.append(block)
            all_dist.append(raw.dist)
            matched_q = np.unique(raw.q_idx)
            um = np.setdiff1d(np.arange(q_df.height), matched_q)
            if um.size:
                unmatched_frames.append(q_df[um].with_columns(pl.lit(key).alias(_STRATUM)))
            records.append({"stratum": key, "n_query": q_df.height, "n_pool": p_df.height,
                            "n_query_matched": n_q_matched, "n_pool_rows": int((block[_ROLE] != "query").sum()),
                            "status": "ok", "note": ""})

        if not frames:
            raise MatchingError("Nessuna coppia/selezione prodotta in alcuno strato (vincoli troppo stretti?).")

        matched = pl.concat(frames, how="diagonal_relaxed")
        unmatched = pl.concat(unmatched_frames, how="diagonal_relaxed") if unmatched_frames else q_df.head(0)
        strata_df = pl.DataFrame(records, schema={
            "stratum": pl.Utf8, "n_query": pl.Int64, "n_pool": pl.Int64, "n_query_matched": pl.Int64,
            "n_pool_rows": pl.Int64, "status": pl.Utf8, "note": pl.Utf8,
        })
        if method == "partial_ot" or algo_name == "partial_ot":
            strata_df = strata_df.join(
                pl.DataFrame({"stratum": list(targets), "target": list(targets.values())}), on="stratum", how="left"
            ).with_columns(pl.col("target").cast(pl.Int64))

        metrics = self._metrics(pop, matched, unmatched, strata_df, all_dist, n_missing, n_trimmed, output_type, q_label)
        if cap is not None:
            metrics["caliper_distance"] = float(cap)
        return MatchingOutcome(
            matched=matched, population=pop, unmatched_query=unmatched, strata=strata_df, metrics=metrics,
            method=method, algorithm=algo_name, metric_name=metric_name, feature_columns=feature_cols,
            output_type=output_type, query_label=q_label, pool_label=p_label, notes=list(self._notes),
            ps_model=ps_model, has_ps=has_ps,
        )

    # ------------------------------------------------------------------ #
    def _check_columns(self, df: pl.DataFrame, sort_var: str | None) -> None:
        v, kinds = self.cfg.variables, self.kinds
        for group_name, cols in (("matching", v.matching), ("stratification", v.stratification), ("evaluation", v.evaluation),
                                 ("caliper", [c.variable for c in self.cfg.constraints.calipers])):
            miss = [c for c in cols if c not in df.columns]
            if miss:
                raise ConfigError(
                    f"Variabili di '{group_name}' assenti dal dataset di lavoro: {miss}. "
                    f"Colonne disponibili: {[c for c in df.columns if not c.startswith('__')][:25]}"
                )
        untyped = [c for c in v.matching if c not in kinds]
        if untyped:
            raise ConfigError(
                f"Variabili di matching non tipizzate nella DataConfig (serve categorical o numerical): {untyped}"
            )
        non_numeric = [c for c in v.matching if not df.schema[c].is_numeric()]
        if non_numeric:
            raise ConfigError(
                f"Le variabili di matching devono essere numeriche (usa 'mappings' per codificare): {non_numeric}"
            )

    def _warn_multilevel(self, df: pl.DataFrame, cols: list[str]) -> None:
        if self.cfg.metrics.name in ("hybrid", "gower"):
            return
        multi = [c for c in cols if self.kinds.get(c) == "categorical" and df[c].n_unique() > 2]
        if multi:
            self._notes.append(
                f"Variabili categoriche con >2 livelli trattate come numeri nella metrica '{self.cfg.metrics.name}': {multi}. "
                "Considera 'hybrid' o la stratificazione."
            )

    def _trim(self, work: pl.DataFrame, psp: PropensityScoreParams) -> tuple[pl.DataFrame, int]:
        if psp.trim == "none":
            return work, 0
        g = work[GROUP_COL].to_numpy()
        ps = work[PS_COL].to_numpy()
        pt, pc = ps[g == 1], ps[g == 0]
        if psp.trim == "minmax":
            lo, hi = max(pt.min(), pc.min()), min(pt.max(), pc.max())
        else:
            q = psp.trim_quantile
            lo = max(np.quantile(pt, q), np.quantile(pc, q))
            hi = min(np.quantile(pt, 1 - q), np.quantile(pc, 1 - q))
        keep = (ps >= lo) & (ps <= hi)
        n = int((~keep).sum())
        if n:
            self._notes.append(f"Trimming del supporto comune ({psp.trim}): escluse {n} unita' con PS fuori [{lo:.3f}, {hi:.3f}].")
        out = work.filter(pl.Series(keep))
        if out.filter(pl.col(GROUP_COL) == 1).height == 0 or out.filter(pl.col(GROUP_COL) == 0).height == 0:
            raise MatchingError("Il trimming ha svuotato un gruppo: nessun supporto comune.")
        return out, n

    def _global_caliper(self, work, feature_cols, metric, is_query, algo_name, cfg, method, psp) -> float | None:
        caps: list[float] = []
        gc = cfg.constraints.global_caliper
        feat_sd = None
        if len(feature_cols) == 1:
            feat_sd = float(work[feature_cols[0]].cast(pl.Float64).std() or 0.0)
        if gc is not None:
            if gc.scale == "absolute":
                caps.append(gc.value)
            else:
                if feat_sd is None:
                    feat_sd = self._distance_sd(work, feature_cols, metric, is_query)
                caps.append(gc.value * feat_sd)
        if method == "propensity_score" and psp is not None and psp.caliper_sd is not None:
            caps.append(psp.caliper_sd * (feat_sd or 0.0))
        caps = [c for c in caps if c > 0]
        return min(caps) if caps else None

    def _distance_sd(self, work, cols, metric, is_query) -> float:
        X = work.select(cols).cast(pl.Float64).to_numpy()
        Xq, Xp = X[is_query], X[~is_query]
        rng = np.random.default_rng(0)
        iq = rng.choice(len(Xq), min(len(Xq), 1000), replace=False)
        ip = rng.choice(len(Xp), min(len(Xp), 1000), replace=False)
        d = metric.pairwise(Xq[iq], Xp[ip])
        return float(d.std())

    def _ot_targets(self, plan, params) -> dict[str, int]:
        keys = [k for k, _, _ in plan]
        n_q = np.array([q.height for _, q, _ in plan])
        n_p = np.array([p.height for _, _, p in plan])
        if params.target_ratio is not None:
            raw = params.target_ratio * n_q
        else:
            raw = params.target_size * n_q / n_q.sum()
        t = np.floor(raw).astype(int)
        rem = int(round(raw.sum())) - int(t.sum()) if params.target_size is not None else 0
        if rem > 0:
            order = np.argsort(-(raw - t))
            t[order[:rem]] += 1
        if params.target_ratio is not None:
            t = np.rint(raw).astype(int)
        t = np.maximum(t, 1)
        clipped = t > n_p
        if clipped.any():
            self._notes.append(
                f"partial_ot: target ridotto al pool disponibile in {int(clipped.sum())} strato/i "
                f"(richiesti {int(t.sum())}, disponibili al massimo {int(np.minimum(t, n_p).sum())})."
            )
        t = np.minimum(t, n_p)
        return dict(zip(keys, t.tolist()))

    @staticmethod
    def _prefilter(cost: np.ndarray, k: int, need: int) -> np.ndarray:
        n_q, n_p = cost.shape
        finite = np.where(cost < PENALTY_THRESHOLD, cost, np.inf)
        kk = min(k, n_p)
        while True:
            idx = np.argpartition(finite, kth=kk - 1, axis=1)[:, :kk]
            cand = np.unique(idx[np.isfinite(np.take_along_axis(finite, idx, axis=1))])
            if cand.size >= min(need, n_p) or kk >= n_p:
                return cand if cand.size else np.arange(n_p)
            kk = min(n_p, kk * 2)

    def _assemble(self, q_df, p_df, raw: RawMatch, key: str, offset: int):
        if raw.output_type == "selection":
            w = raw.weights if raw.weights is not None else np.ones(raw.p_idx.size)
            w = w / w.sum() * q_df.height
            q_rows = q_df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias(_PAIR), pl.lit(1.0).alias(_WEIGHT),
                pl.lit(None, dtype=pl.Float64).alias(_DIST), pl.lit("query").alias(_ROLE), pl.lit(key).alias(_STRATUM),
            )
            p_rows = p_df[raw.p_idx].with_columns(
                pl.lit(None, dtype=pl.Int64).alias(_PAIR), pl.Series(_WEIGHT, w), pl.Series(_DIST, raw.dist),
                pl.lit("selected").alias(_ROLE), pl.lit(key).alias(_STRATUM),
            )
            return pl.concat([q_rows, p_rows], how="diagonal_relaxed"), q_df.height, 0
        uq, first = np.unique(raw.pair_id, return_index=True)
        pid_map = {int(p): i + offset for i, p in enumerate(uq)}
        pair_ids = np.array([pid_map[int(p)] for p in raw.pair_id], dtype=np.int64)
        counts = np.bincount(raw.pair_id)[raw.pair_id].astype(float)
        q_rows = q_df[raw.q_idx[first]].with_columns(
            pl.Series(_PAIR, np.array([pid_map[int(p)] for p in uq], dtype=np.int64)), pl.lit(1.0).alias(_WEIGHT),
            pl.lit(None, dtype=pl.Float64).alias(_DIST), pl.lit("query").alias(_ROLE), pl.lit(key).alias(_STRATUM),
        )
        p_rows = p_df[raw.p_idx].with_columns(
            pl.Series(_PAIR, pair_ids), pl.Series(_WEIGHT, 1.0 / counts), pl.Series(_DIST, raw.dist),
            pl.lit("match").alias(_ROLE), pl.lit(key).alias(_STRATUM),
        )
        return pl.concat([q_rows, p_rows], how="diagonal_relaxed"), int(uq.size), int(uq.size)

    def _metrics(self, pop, matched, unmatched, strata, dists, n_missing, n_trimmed, output_type, q_label):
        n_t = int((pop[GROUP_COL] == 1).sum())
        n_c = int((pop[GROUP_COL] == 0).sum())
        q = matched.filter(pl.col(_ROLE) == "query")
        p = matched.filter(pl.col(_ROLE) != "query")
        n_query_total = int(strata["n_query"].sum())
        n_query_matched = int(q.height) if output_type == "pairs" else int(strata.filter(pl.col("status") == "ok")["n_query"].sum())
        d = np.concatenate(dists) if dists else np.array([])
        w = p[_WEIGHT].to_numpy()
        ess = float(w.sum() ** 2 / (w ** 2).sum()) if w.size else 0.0
        m: dict[str, float | int | str | bool] = {
            "n_treated_eligible": n_t, "n_control_eligible": n_c,
            "n_excluded_missing": n_missing, "n_trimmed": n_trimmed,
            "n_query_total": n_query_total, "n_query_matched": n_query_matched,
            "match_rate": n_query_matched / n_query_total if n_query_total else 0.0,
            "n_pool_rows": int(p.height), "n_unique_pool_units": int(p["__uid__"].n_unique()),
            "max_pool_reuse": int(p.group_by("__uid__").len()["len"].max()) if p.height else 0,
            "ess_pool": ess, "n_strata": int(strata.height),
            "n_strata_skipped": int(strata.filter(pl.col("status") != "ok").height),
            "output_type": output_type,
        }
        if d.size:
            m.update({"distance_mean": float(d.mean()), "distance_median": float(np.median(d)),
                      "distance_p95": float(np.quantile(d, 0.95)), "distance_max": float(d.max())})
        return m
