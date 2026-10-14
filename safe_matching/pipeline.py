"""Pipeline end-to-end: caricamento -> preprocessing -> matching -> validazione -> report PDF."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from safe_matching.evaluation import balance_table, run_hooks
from safe_matching.evaluation import plots
from safe_matching.evaluation.balance import BalanceResult
from safe_matching.evaluation.hooks import HookResult
from safe_matching.exceptions import ConfigError
from safe_matching.matching import MatchingEngine, MatchingOutcome
from safe_matching.preprocessing import Loader, PreparedData, prepare_datasets
from safe_matching.preprocessing.assembly import GROUP_COL
from safe_matching.reporting import generate_report
from safe_matching.schemas import DataConfig, MatchingConfig

__all__ = ["run_pipeline", "PipelineResult"]
log = logging.getLogger("safe_matching")


@dataclass
class PipelineResult:
    outcome: MatchingOutcome
    balance: BalanceResult
    prepared: PreparedData
    hooks: list[HookResult]
    output_dir: Path
    report_path: Path | None
    files: dict[str, Path] = field(default_factory=dict)


def _as_df(x: Any, kw: dict) -> tuple[pl.DataFrame, str]:
    if isinstance(x, pl.DataFrame):
        return x, "dataframe"
    return Loader.load(x, **kw), Path(x).stem


def run_pipeline(
    datasets: list[str | Path | pl.DataFrame],
    data_configs: list[str | Path | DataConfig],
    matching_config: str | Path | MatchingConfig,
    *,
    group_col: str | None = None,
    treated_value: Any | None = None,
    treated_dataset: int = 0,
    output_dir: str | Path = "outputs",
    lenient_columns: bool = False,
    generate_pdf: bool = True,
    loader_kwargs: dict | None = None,
) -> PipelineResult:
    """Esegue l'intera pipeline.

    * 1 dataset  -> ``group_col`` obbligatorio (``treated_value`` opzionale se il gruppo e' 0/1);
    * 2 dataset  -> il primo (o ``treated_dataset``) e' il gruppo trattato; 1 config dati viene
      riusata per entrambi, oppure se ne passano 2.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    cfg = matching_config if isinstance(matching_config, MatchingConfig) else MatchingConfig.from_file(matching_config)
    dcfgs = [c if isinstance(c, DataConfig) else DataConfig.from_file(c) for c in data_configs]
    loaded = [_as_df(d, loader_kwargs or {}) for d in datasets]
    if len(loaded) == 2 and loaded[0][1] == loaded[1][1]:
        loaded[1] = (loaded[1][0], loaded[1][1] + "_2")
    log.info("Dataset caricati: %s", [(n, df.shape) for df, n in loaded])

    prepared = prepare_datasets(
        [d for d, _ in loaded], dcfgs, names=[n for _, n in loaded], group_col=group_col,
        treated_value=treated_value, treated_dataset=treated_dataset, lenient_columns=lenient_columns,
        random_state=cfg.options.random_state,
    )
    v = cfg.variables
    if prepared.mode == "two":
        used = set(v.matching) | set(v.stratification) | {c.variable for c in cfg.constraints.calipers}
        missing = sorted(used - set(prepared.common_columns))
        if missing:
            raise ConfigError(
                f"Variabili di matching/stratificazione/caliper non presenti in ENTRAMBI i dataset: {missing}. "
                f"Colonne comuni: {prepared.common_columns}"
            )
    log.info("Dataset di lavoro: %s righe (trattati=%s)", prepared.data.height, int((prepared.data[GROUP_COL] == 1).sum()))

    outcome = MatchingEngine(cfg, prepared.kinds).run(prepared.data)

    roles: dict[str, str] = {c: "matching" for c in v.matching}
    roles.update({c: "stratification" for c in v.stratification if c not in roles})
    roles.update({c: "evaluation" for c in v.evaluation if c not in roles})
    if prepared.mode == "two":
        dropped = [c for c in roles if c not in prepared.common_columns]
        if dropped:
            outcome.notes.append(f"Variabili di valutazione non comuni ai due dataset, escluse dal bilanciamento: {dropped}")
            roles = {c: r for c, r in roles.items() if c in prepared.common_columns}
    bal = balance_table(outcome.population, outcome.matched, roles, prepared.kinds,
                        smd_threshold=cfg.validation.smd_threshold, metrics=cfg.validation.balance_metrics)

    hooks = run_hooks(cfg.validation.hooks, before=outcome.population, after=outcome.matched,
                      group_col=GROUP_COL, variables=list(roles), kinds=prepared.kinds)

    figures = _figures(outcome, bal, prepared, cfg) if cfg.validation.plots else []

    files: dict[str, Path] = {}
    files["matched_dataset.csv"] = _write_csv(outcome.matched, out_dir / "matched_dataset.csv")
    try:
        outcome.matched.write_parquet(out_dir / "matched_dataset.parquet")
        files["matched_dataset.parquet"] = out_dir / "matched_dataset.parquet"
    except Exception as exc:  # noqa: BLE001
        log.warning("Parquet non scritto: %s", exc)
    files["unmatched_query.csv"] = _write_csv(outcome.unmatched_query, out_dir / "unmatched_query.csv")
    files["strata.csv"] = _write_csv(outcome.strata, out_dir / "strata.csv")
    files["balance_table.csv"] = _write_csv(bal.table, out_dir / "balance_table.csv")
    (out_dir / "metrics.json").write_text(json.dumps({"matching": outcome.metrics, "balance": bal.summary,
                                                      "notes": outcome.notes}, indent=2, default=float), encoding="utf-8")
    files["metrics.json"] = out_dir / "metrics.json"
    (out_dir / "run_config.json").write_text(json.dumps({
        "run_id": run_id, "matching_config": cfg.to_dict(), "data_configs": [c.to_dict() for c in dcfgs],
        "group_col": group_col, "treated_value": treated_value, "treated_dataset": treated_dataset,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files["run_config.json"] = out_dir / "run_config.json"
    if figures:
        (out_dir / "figures").mkdir(exist_ok=True)
        for key, _, png in figures:
            (out_dir / "figures" / f"{key}.png").write_bytes(png)

    report_path = None
    if generate_pdf:
        report_path = generate_report(
            out_dir / "matching_report.pdf", outcome=outcome, balance=bal, cfg=cfg, prepared=prepared,
            figures=figures, hooks=hooks, run_info={"run_id": run_id},
            data_configs_json=[c.to_dict() for c in dcfgs],
        )
        files["matching_report.pdf"] = report_path
    return PipelineResult(outcome, bal, prepared, hooks, out_dir, report_path, files)


def _write_csv(df: pl.DataFrame, path: Path) -> Path:
    df.write_csv(path)
    return path


def _figures(outcome: MatchingOutcome, bal: BalanceResult, prepared: PreparedData, cfg: MatchingConfig):
    figs: list[tuple[str, str, bytes]] = []
    t = bal.table
    thr = cfg.validation.smd_threshold
    if t.height:
        figs.append(("love_plot", "Absolute SMD per variable before (grey) and after (blue) matching.",
                     plots.love_plot(t, thr)))
    if outcome.has_ps:
        col = outcome.feature_columns[0]
        figs.append(("ps_overlap", "Propensity score distributions by group, before and after matching (weighted).",
                     plots.ps_overlap(outcome.population, outcome.matched, col)))
    pool = outcome.matched.filter(pl.col("match_role") != "query")
    d = pool["distance"].drop_nulls().to_numpy()
    if d.size > 1:
        figs.append(("distance_hist", "Distance between each query unit and its match.",
                     plots.distance_hist(d, outcome.metrics.get("caliper_distance"))))
    if outcome.strata.height > 1:
        figs.append(("strata", "Query units available and matched in each stratum.", plots.strata_bars(outcome.strata)))
    n = cfg.validation.max_distribution_plots
    if n and t.height:
        seen: set[str] = set()
        for r in t.sort("abs_smd_before", descending=True, nulls_last=True).iter_rows(named=True):
            b = r["base_variable"]
            if b in seen or len(seen) >= n:
                continue
            seen.add(b)
            figs.append((f"dist_{b}".replace(" ", "_").replace("/", "_"),
                         f"Distribution of {b} by group, before and after matching (|SMD| {r['abs_smd_before']:.2f} -> {r['abs_smd_after']:.2f}).",
                         plots.distribution_panel(outcome.population, outcome.matched, b, r["type"])))
    return figs
