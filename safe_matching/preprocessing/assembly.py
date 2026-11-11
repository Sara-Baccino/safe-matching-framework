"""Costruzione del dataset di lavoro (trattati + controlli) da 1 o 2 sorgenti gia' preprocessate.

* 1 dataset  -> contiene trattati e controlli; una colonna del dataset ORIGINALE identifica il gruppo
  (non serve che sia attiva nella data config).
* 2 dataset  -> il primo (o ``treated_dataset``) e' il gruppo trattato, l'altro il controllo; devono
  condividere almeno una colonna attiva (dopo ``new_name``).

Colonne riservate: ``__uid__`` id univoco, ``__group__`` (1 trattato / 0 controllo), ``__src__`` indice
della sorgente, ``__row__`` posizione della riga nella sorgente, ``__source__`` nome sorgente, ``__id__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from safe_matching.exceptions import ConfigError, PreprocessingError
from safe_matching.preprocessing.source import SourceData, column_kinds, ordinal_columns, resolve_raw_column
from safe_matching.preprocessing.transform import norm_text
from safe_matching.preprocessing.missing import IMP_PREFIX, REPORT_SCHEMA, column_rules, impute_frame
from safe_matching.schemas.matching_config import MissingPolicy


__all__ = ["PreparedData", "prepare_datasets", "GROUP_COL", "UID_COL", "SOURCE_COL", "ID_COL", "SRC_IDX_COL", "ROW_COL"]

GROUP_COL, UID_COL, SOURCE_COL, ID_COL = "__group__", "__uid__", "__source__", "__id__"
SRC_IDX_COL, ROW_COL = "__src__", "__row__"


@dataclass
class PreparedData:
    data: pl.DataFrame
    kinds: dict[str, str]
    common_columns: list[str]
    mode: str  # "single" | "two"
    sources: list[SourceData]
    summaries: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    treated_label: str = "treated"
    control_label: str = "control"
    missing_report: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=REPORT_SCHEMA))
    """Una riga per (sorgente, variabile usata): quota di mancanti, fascia, azione e celle imputate."""
    dropped_variables: list[str] = field(default_factory=list)
    """Variabili escluse dal matching perche' hanno troppi mancanti (options.missing.over_threshold_action)."""
    ordinal: set[str] = field(default_factory=set)
    """Categoriche ordinali (categorical+numerical nella config): per il motore sono numeriche."""


def key_string(s: pl.Series) -> pl.Series:
    """Testo stabile per confrontare valori ('1.0' -> '1'); vuoto -> null."""
    t = norm_text(s)
    return pl.DataFrame({"t": t}).select(pl.when(pl.col("t") == "").then(None).otherwise(pl.col("t")).alias(s.name)).to_series()

def _binarize(raw: pl.Series, treated_value: Any | None) -> tuple[pl.Series, Any, list[str]]:
    warns: list[str] = []
    keys = key_string(raw)
    if keys.null_count():
        warns.append(f"{keys.null_count()} rows with missing group: excluded.")

    uniq = set(keys.drop_nulls().unique().to_list())
    if treated_value is None:
        if uniq <= {"0", "1"}:
            treated_value = 1
        else:
            raise ConfigError(
                f"Impossibile dedurre il valore 'trattato' (valori nella colonna gruppo: {sorted(uniq)[:10]}). "
                "Imposta TREATED_VALUE nel main."
            )
    tv = key_string(pl.Series([treated_value])).item()
    if tv not in uniq:
        raise ConfigError(f"Il valore trattato '{treated_value}' non compare nella colonna gruppo (valori: {sorted(uniq)[:10]}).")
    g = pl.Series(GROUP_COL, [None if k is None else int(k == tv) for k in keys.to_list()], dtype=pl.Int8)
    return g, treated_value, warns


def prepare_datasets(
    sources: list[SourceData],
    *,
    group_column: str | None = None,
    treated_value: Any | None = None,
    treated_dataset_index: int = 0,
    missing_policy: MissingPolicy | None = None,
    targets: dict[str, str] | None = None,
) -> PreparedData:
    """``targets``: {variabile: ruolo} da imputare (matching/stratificazione/evaluation, MAI gli outcome). Senza
    ``missing_policy`` non si imputa."""
    if len(sources) not in (1, 2):
        raise ConfigError("Servono 1 dataset (con colonna gruppo) oppure 2 dataset (trattati, controlli).")
    warnings: list[str] = []
    frames: list[pl.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    if len(sources) == 1:
        if not group_column:
            raise ConfigError("Con un solo dataset e' obbligatorio indicare GROUP_COLUMN.")
        s = sources[0]
        graw = resolve_raw_column(s.raw, s.config, group_column)
        group, tval, gw = _binarize(s.raw[graw], treated_value)
        warnings += [f"[{s.name}] {w}" for w in gw]
        work = s.pre.with_columns(group, pl.int_range(0, s.pre.height).alias(ROW_COL), pl.lit(0).alias(SRC_IDX_COL))
        work = work.filter(pl.col(GROUP_COL).is_not_null())
        frames.append(work)
        kinds = column_kinds(s.config)
        common = [c for c in kinds if c in work.columns]
        treated_label, control_label, mode = f"{graw}={tval}", f"{graw}!={tval}", "single"
        summaries.append(_summary(s, "mixed", work.height))
    else:
        if treated_dataset_index not in (0, 1):
            raise ConfigError("TREATED_DATASET_INDEX deve essere 0 o 1.")
        for i, s in enumerate(sources):
            role = "treated" if i == treated_dataset_index else "control"
            frames.append(s.pre.with_columns(
                pl.lit(1 if i == treated_dataset_index else 0, dtype=pl.Int8).alias(GROUP_COL),
                pl.int_range(0, s.pre.height).alias(ROW_COL), pl.lit(i).alias(SRC_IDX_COL),
            ))
            summaries.append(_summary(s, role, s.pre.height))
        k0, k1 = column_kinds(sources[0].config), column_kinds(sources[1].config)
        common = [c for c in k0 if c in k1 and c in frames[0].columns and c in frames[1].columns]
        if not common:
            raise PreprocessingError(
                "I due dataset non hanno colonne attive in comune dopo il rename (new_name). "
                f"Dataset 1: {sorted(k0)[:12]}; dataset 2: {sorted(k1)[:12]}"
            )
        conflicts = [c for c in common if k0[c] != k1[c]]
        if conflicts:
            raise ConfigError(f"Tipo categorical/numerical discordante tra le due config per: {conflicts}")
        kinds = {**k1, **k0}
        extra = [sorted(set(k) - set(common))[:6] for k in (k0, k1)]
        if any(extra):
            warnings.append(f"Columns not common to both sources (null in the other one): {extra[0]} | {extra[1]}")
        treated_label = sources[treated_dataset_index].name
        control_label = sources[1 - treated_dataset_index].name
        mode = "two"

    names = [s.name for s in sources]
    reports: list[dict] = []
    dropped: list[str] = []
    if missing_policy is not None and targets:
        for i, (fr, src) in enumerate(zip(frames, sources)):
            frames[i], rows, dr = impute_frame(
                fr, source=src.name, targets=targets, kinds=column_kinds(src.config), ordinal=ordinal_columns(src.config),
                policy=missing_policy, rules=column_rules(src.config), group_col=GROUP_COL if mode == "single" else None)
            reports += rows
            dropped += [d for d in dr if d not in dropped]
    work = pl.concat(frames, how="diagonal_relaxed")
    flag_cols = [c for c in work.columns if c.startswith(IMP_PREFIX)]
    if flag_cols:
        work = work.with_columns([pl.col(c).fill_null(False) for c in flag_cols])
    work = work.with_columns(
        pl.col(SRC_IDX_COL).map_elements(lambda i: names[i], return_dtype=pl.Utf8).alias(SOURCE_COL),
    ).with_columns(
        (pl.col(SOURCE_COL) + ":" + pl.col(ROW_COL).cast(pl.Utf8)).alias(ID_COL),
    ).with_row_index(UID_COL).with_columns(pl.col(UID_COL).cast(pl.Int64))

    n1, n0 = int((work[GROUP_COL] == 1).sum()), int((work[GROUP_COL] == 0).sum())
    if n1 == 0 or n0 == 0:
        raise PreprocessingError(f"Servono unita' in entrambi i gruppi (trattati={n1}, controlli={n0}).")
    return PreparedData(
        data=work, kinds=kinds, common_columns=common, mode=mode, sources=sources, summaries=summaries,
        warnings=warnings, treated_label=treated_label, control_label=control_label,        
        ordinal=set().union(*(ordinal_columns(x.config) for x in sources)),
        missing_report=pl.DataFrame(reports, schema=REPORT_SCHEMA) if reports else pl.DataFrame(schema=REPORT_SCHEMA),
        dropped_variables=dropped,
    )


def _summary(s: SourceData, role: str, n_after_group: int) -> dict[str, Any]:
    return {
        "name": s.name, "role": role, "config": s.config_ref,
        "n_rows_raw": s.raw.height, "n_rows_usable": n_after_group,
        "n_columns_raw": s.raw.width, "n_columns_active": s.pre.width,
    }
