"""Pipeline end-to-end: lettura/preprocessing (safe-data-processing) -> matching -> validazione -> output + report."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from safe_matching.evaluation import balance_table, plots, run_hooks
from safe_matching.estimation import EstimationResult, estimate_effects
from safe_matching.evaluation.analysis import AnalysisResult, run_analysis
from safe_matching.evaluation.balance import BalanceResult
from safe_matching.evaluation.hooks import HookResult
from safe_matching.evaluation.success import SuccessResult, evaluate_success
from safe_matching.exceptions import ConfigError
from safe_matching.matching import MatchingEngine, MatchingOutcome
from safe_matching.outputs import build_output, write_table
from safe_matching.preprocessing import PreparedData, load_source, prepare_datasets
from safe_matching.preprocessing.harmonize import HarmonizationReport, harmonize_sources
from safe_matching.preprocessing.assembly import GROUP_COL
from safe_matching.reporting import generate_report
from safe_matching.schemas import MatchingConfig

__all__ = ["run_pipeline", "PipelineResult"]
log = logging.getLogger("safe_matching")


@dataclass
class PipelineResult:
    outcome: MatchingOutcome
    balance: BalanceResult
    success: SuccessResult
    analysis: AnalysisResult
    estimation: EstimationResult | None
    harmonization: HarmonizationReport | None
    prepared: PreparedData
    hooks: list[HookResult]
    matched_dataset: pl.DataFrame
    unmatched_dataset: pl.DataFrame
    output_dir: Path
    report_path: Path | None
    files: dict[str, Path] = field(default_factory=dict)


def run_pipeline(
    datasets: list[str | Path | pl.DataFrame],
    data_configs: list[str | Path | Any],
    matching_config: str | Path | dict | MatchingConfig,
    *,
    group_column: str | None = None,
    treated_value: Any | None = None,
    treated_dataset_index: int = 0,
    output_dir: str | Path = "outputs",
    output_name: str = "matching",
    output_values: str = "postprocessed",
    output_formats: tuple[str, ...] | list[str] = ("csv", "xlsx"),
    restore_inactive_columns: bool = True,
    generate_pdf: bool = True,
    read_options: dict[str, Any] | None = None,
    on_unmapped: str = "error",
    on_cast_error: str = "error",
) -> PipelineResult:
    """Esegue l'intera pipeline. Vedi ``main.py`` per il significato dei parametri.

    Con 2 dataset si puo' passare una sola config dati (riletta per ciascun dataset)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    if isinstance(matching_config, MatchingConfig):
        cfg = matching_config
    elif isinstance(matching_config, dict):
        cfg = MatchingConfig.from_dict(matching_config)
    else:
        cfg = MatchingConfig.from_file(matching_config)

    if len(datasets) not in (1, 2):
        raise ConfigError("Servono 1 oppure 2 dataset.")
    if len(data_configs) == 1 and len(datasets) == 2:
        data_configs = [data_configs[0], data_configs[0]]
    if len(data_configs) != len(datasets):
        raise ConfigError(f"Config dati ({len(data_configs)}) incoerenti con i dataset ({len(datasets)}).")

    names = [Path(d).stem if not isinstance(d, pl.DataFrame) else f"dataset_{i + 1}" for i, d in enumerate(datasets)]
    if len(names) == 2 and names[0] == names[1]:
        names[1] += "_2"
    sources = [load_source(d, c, n, read_options=read_options, on_unmapped=on_unmapped, on_cast_error=on_cast_error)
               for d, c, n in zip(datasets, data_configs, names)]
    v0 = cfg.variables
    targets = {**{c: "evaluation" for c in v0.evaluation}, **{c: "stratification" for c in v0.stratification},
               **{c: "matching" for c in v0.matching}}
    outcomes = {o.variable for o in cfg.estimation.outcomes}            # gli outcome non si imputano mai
    targets = {c: r for c, r in targets.items() if c not in outcomes}
    harm = None
    if len(sources) == 2 and cfg.options.harmonization.enabled:
        used = list(dict.fromkeys(list(targets) + sorted(outcomes)))
        sources, harm = harmonize_sources(sources, variables=used, policy=cfg.options.harmonization)
    prepared = prepare_datasets(sources, group_column=group_column, treated_value=treated_value,
                                treated_dataset_index=treated_dataset_index, missing_policy=cfg.options.missing, targets=targets)
    drop_notes: list[str] = []
    if prepared.dropped_variables:
        cfg, drop_notes = _drop_variables(cfg, prepared.dropped_variables)

    v = cfg.variables
    if prepared.mode == "two":
        used = set(v.matching) | set(v.stratification) | {c.variable for c in cfg.constraints.calipers}
        missing = sorted(used - set(prepared.common_columns))
        if missing:
            raise ConfigError(f"Variabili non presenti in ENTRAMBI i dataset: {missing}. Comuni: {prepared.common_columns}")
    log.info("Dataset di lavoro: %s righe (trattati=%s)", prepared.data.height, int((prepared.data[GROUP_COL] == 1).sum()))

    outcome = MatchingEngine(cfg, prepared.kinds).run(prepared.data)
    outcome.notes[:0] = drop_notes

    roles: dict[str, str] = {c: "matching" for c in v.matching}
    roles.update({c: "stratification" for c in v.stratification if c not in roles})
    roles.update({c: "evaluation" for c in v.evaluation if c not in roles})
    if prepared.mode == "two":
        dropped = [c for c in roles if c not in prepared.common_columns]
        if dropped:
            outcome.notes.append(f"Evaluation variables not present in both datasets, excluded from the balance analysis: {dropped}")
            roles = {c: r for c, r in roles.items() if c in prepared.common_columns}
    bal = balance_table(outcome.population, outcome.matched, roles, prepared.kinds,
                        smd_threshold=cfg.validation.smd_threshold, metrics=cfg.validation.balance_metrics)
    ranked = [b for b in dict.fromkeys(bal.table.sort("abs_smd_before", descending=True, nulls_last=True)["base_variable"].to_list())] \
        if bal.table.height else None
    ps_data = _ps_data(outcome, cfg, prepared.kinds) if cfg.validation.analysis.ps_support else None
    analysis = run_analysis(outcome.population, outcome.matched, roles, prepared.kinds, cfg.validation.analysis,
                            smd_threshold=cfg.validation.smd_threshold, plots_enabled=cfg.validation.plots,
                            max_distribution_variables=cfg.validation.max_distribution_plots, ranked_variables=ranked,
                            ps_data=ps_data)
    outcome.notes.extend(analysis.notes)
    estimation = None
    if cfg.estimation.enabled:
        estimation = estimate_effects(
            outcome.population, cfg, prepared.kinds, {cfg.population.direction: outcome.matched},
            common_columns=prepared.common_columns if prepared.mode == "two" else None)
        outcome.notes.extend(estimation.notes)
    success = evaluate_success(outcome.metrics, bal.summary, cfg.validation.success_criteria, cfg.validation.smd_threshold,
                               analysis.metrics)
    hooks = run_hooks(cfg.validation.hooks, before=outcome.population, after=outcome.matched,
                      group_col=GROUP_COL, variables=list(roles), kinds=prepared.kinds)
    figures = (_figures(outcome, bal, cfg) if cfg.validation.plots else []) + analysis.figures
    if estimation is not None and cfg.validation.plots:
        figures += _estimation_figures(estimation, cfg)

    # ---- output: colonne originali + metadati ----------------------------------
    matched_out = build_output(outcome.matched, prepared, values=output_values,
                               restore_inactive_columns=restore_inactive_columns,)
    unmatched_out = (build_output(outcome.unmatched_query, prepared, values=output_values,
                                  restore_inactive_columns=restore_inactive_columns,)
                     if outcome.unmatched_query.height else pl.DataFrame())
    files: dict[str, Path] = {}
    files.update(write_table(matched_out, out_dir / f"{output_name}_dataset", output_formats))
    if unmatched_out.height:
        files.update(write_table(unmatched_out, out_dir / f"{output_name}_unmatched", output_formats))
    for src in sources:   # config dati EFFETTIVA di ciascuna sorgente (mappings estesi/unificati inclusi)
        cp = out_dir / f"{output_name}_data_config_{src.name}.json"
        cp.write_text(json.dumps(src.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        files[cp.name] = cp
    if harm is not None:
        for nm, df_ in (("numeric", harm.numeric), ("categorical", harm.categorical)):
            df_.write_csv(out_dir / f"{output_name}_harmonization_{nm}.csv")
            files[f"{output_name}_harmonization_{nm}.csv"] = out_dir / f"{output_name}_harmonization_{nm}.csv"    
    prepared.missing_report.write_csv(out_dir / f"{output_name}_missing_report.csv")
    files[f"{output_name}_missing_report.csv"] = out_dir / f"{output_name}_missing_report.csv"
    outcome.strata.write_csv(out_dir / f"{output_name}_strata.csv")
    bal.table.write_csv(out_dir / f"{output_name}_balance_table.csv")
    for tname, tdf in analysis.tables.items():
        tdf.write_csv(out_dir / f"{output_name}_{tname}.csv")
        files[f"{output_name}_{tname}.csv"] = out_dir / f"{output_name}_{tname}.csv"
    files[f"{output_name}_strata.csv"] = out_dir / f"{output_name}_strata.csv"
    files[f"{output_name}_balance_table.csv"] = out_dir / f"{output_name}_balance_table.csv"
    if estimation is not None:
        estimation.table.write_csv(out_dir / f"{output_name}_effects.csv")
        files[f"{output_name}_effects.csv"] = out_dir / f"{output_name}_effects.csv"
    mpath = out_dir / f"{output_name}_metrics.json"
    mpath.write_text(json.dumps({
        "run_id": run_id, "matching": outcome.metrics, "balance": bal.summary, "analysis": analysis.metrics,
        "estimation": estimation.metrics if estimation else None,
        "success": {"passed": success.passed, "checks": success.checks}, "notes": outcome.notes,
    }, indent=2, default=float), encoding="utf-8")
    files[mpath.name] = mpath
    rpath = out_dir / f"{output_name}_run_config.json"
    rpath.write_text(json.dumps({"run_id": run_id, "matching_config": cfg.to_dict(), "datasets": names,
                                 "data_configs": [c if isinstance(c, (str, Path)) else "<in-memory>" for c in data_configs],
                                 "group_column": group_column, "treated_value": treated_value,
                                 "output_values": output_values}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    files[rpath.name] = rpath
    if figures:
        (out_dir / "figures").mkdir(exist_ok=True)
        for key, _, png in figures:
            (out_dir / "figures" / f"{output_name}_{key}.png").write_bytes(png)

    report_path = None
    if generate_pdf:
        report_path = generate_report(
            out_dir / f"{output_name}_report.pdf", outcome=outcome, balance=bal, success=success, cfg=cfg,
            prepared=prepared, figures=figures, analysis=analysis, estimation=estimation, harmonization=harm, hooks=hooks, run_info={"run_id": run_id},
        )
        files[report_path.name] = report_path
    return PipelineResult(outcome, bal, success, analysis, estimation, harm, prepared, hooks, matched_out, unmatched_out, out_dir, report_path, files)


def _figures(outcome: MatchingOutcome, bal: BalanceResult, cfg: MatchingConfig):
    figs: list[tuple[str, str, bytes]] = []
    t = bal.table
    thr = cfg.validation.smd_threshold
    if t.height:
        figs.append(("love_plot", "Absolute SMD per variable before (grey) and after (blue) matching.", plots.love_plot(t, thr)))
    d = outcome.matched.filter(pl.col("match_role") != "query")["distance"].drop_nulls().to_numpy()
    if d.size > 1:
        figs.append(("distance_hist", "Distance between each query unit and its match.",
                     plots.distance_hist(d, outcome.metrics.get("caliper_distance"))))
    if outcome.strata.height > 1:
        figs.append(("strata", "Query units available and matched in each stratum.", plots.strata_bars(outcome.strata)))
    return figs


def _ps_data(outcome: MatchingOutcome, cfg: MatchingConfig, kinds: dict[str, str]):
    """(popolazione, abbinati, colonna PS). Con metodi non-PS stima un PS logistico solo per la diagnostica."""
    from safe_matching.matching.engine import PS_COL
    if outcome.has_ps:
        return outcome.population, outcome.matched, PS_COL
    try:
        from safe_matching.matching.representation import design_matrix, fit_propensity
        from safe_matching.schemas.method_params import PropensityScoreParams
        v = cfg.variables
        cov = list(dict.fromkeys(v.matching + v.stratification))
        X, _ = design_matrix(outcome.population, cov, kinds)
        ps, _, _ = fit_propensity(X, outcome.population[GROUP_COL].to_numpy(), PropensityScoreParams(), cfg.options.random_state)
        pop = outcome.population.with_columns(pl.Series(PS_COL, ps))
        lut = pop.select(["__uid__", PS_COL])
        return pop, outcome.matched.join(lut, on="__uid__", how="left"), PS_COL
    except Exception as exc:  # noqa: BLE001
        outcome.notes.append(f"PS diagnostics not computed: {exc}")
        return None


def _estimation_figures(est: EstimationResult, cfg: MatchingConfig):
    ci = cfg.estimation.bootstrap.ci_level
    figs = [("eff_forest", f"Estimated effects with {100 * ci:.0f}% bootstrap percentile intervals; grey squares are the unadjusted differences.",
             plots.effects_forest(est.table, ci))]
    keys = [k for k in est.boot if k[2] == "matched" and k[3] == "difference"][:6]
    if keys:
        figs.append(("eff_boot", "Bootstrap distributions of the matched effect.", plots.effects_bootstrap(est.boot, est.point, keys, ci)))
    return figs

def _drop_variables(cfg: MatchingConfig, dropped: list[str]) -> tuple[MatchingConfig, list[str]]:
    """Toglie dal config le variabili con troppi mancanti (copia: l'oggetto dell'utente non viene modificato)."""
    from safe_matching.exceptions import ConfigError
    c = cfg.model_copy(deep=True)
    v = c.variables
    for name in ("matching", "stratification", "evaluation"):
        setattr(v, name, [x for x in getattr(v, name) if x not in dropped])
    c.constraints.calipers = [w for w in c.constraints.calipers if w.variable not in dropped]
    if not v.matching:
        raise ConfigError(f"Tutte le variabili di matching hanno troppi mancanti e sono state escluse: {dropped}.")
    note = (f"Variables excluded because of >{100 * cfg.options.missing.max_fraction:.0f}% missing values "
            f"(options.missing.over_threshold_action='drop_variable'): {dropped}.")
    return c, [note]
