"""Costruzione del dataset di lavoro (trattati + controlli) da 1 o 2 sorgenti.

* 1 dataset  -> contiene trattati e controlli; una colonna identifica il gruppo.
* 2 dataset  -> il primo (o ``treated_dataset``) e' il gruppo trattato, l'altro il
  controllo; devono condividere almeno una colonna attiva (dopo il rename).

Colonne riservate aggiunte al dataset di lavoro:
``__uid__`` (id univoco globale), ``__group__`` (1 = trattato, 0 = controllo),
``__source__`` (nome della sorgente), ``__id__`` (id leggibile, testo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from safe_matching.exceptions import ConfigError, PreprocessingError
from safe_matching.preprocessing.preprocess import Preprocessor, PreprocessReport, key_string
from safe_matching.schemas.data_config import DataConfig

__all__ = ["PreparedData", "prepare_datasets", "GROUP_COL", "UID_COL", "SOURCE_COL", "ID_COL"]

GROUP_COL = "__group__"
UID_COL = "__uid__"
SOURCE_COL = "__source__"
ID_COL = "__id__"
_ROW = "__row__"
_GRAW = "__group_raw__"


@dataclass
class PreparedData:
    data: pl.DataFrame
    kinds: dict[str, str]
    common_columns: list[str]
    mode: str  # "single" | "two"
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    treated_label: str = "treated"
    control_label: str = "control"


def _resolve_group_raw(df: pl.DataFrame, cfg: DataConfig, group_col: str) -> str:
    if group_col in df.columns:
        return group_col
    raw = cfg.raw_name_of(group_col)
    if raw is not None and raw in df.columns:
        return raw
    raise ConfigError(
        f"Colonna gruppo '{group_col}' non trovata nel dataset (colonne: {df.columns[:15]}...)."
    )


def _binarize_group(raw: pl.Series, treated_value: Any | None) -> tuple[pl.Series, Any, list[str]]:
    warns: list[str] = []
    n_null = raw.null_count()
    if n_null:
        warns.append(f"{n_null} righe con gruppo mancante: escluse.")
    values = raw.drop_nulls().unique().to_list()
    if treated_value is None:
        if set(map(str, key_string(raw.drop_nulls()).unique().to_list())) <= {"0", "1"}:
            treated_value = 1
        elif raw.dtype == pl.Boolean:
            treated_value = True
        else:
            raise ConfigError(
                f"Impossibile dedurre il valore 'trattato' dalla colonna gruppo (valori: {values[:10]}). "
                "Specificare --treated-value."
            )
    keys = key_string(raw)
    tv = key_string(pl.Series([treated_value])).item()
    group = pl.Series(
        "g", [None if k is None else int(k == tv) for k in keys.to_list()], dtype=pl.Int8
    )
    return group.alias(GROUP_COL), treated_value, warns


def _finish_ids(df: pl.DataFrame, cfg: DataConfig) -> pl.DataFrame:
    id_cols = [c for c in cfg.id_new_names() if c in df.columns]
    if id_cols:
        return df.with_columns(pl.col(id_cols[0]).cast(pl.Utf8).alias(ID_COL))
    return df.with_columns(pl.col(_ROW).cast(pl.Utf8).alias(ID_COL))


def prepare_datasets(
    datasets: list[pl.DataFrame],
    configs: list[DataConfig],
    *,
    names: list[str] | None = None,
    group_col: str | None = None,
    treated_value: Any | None = None,
    treated_dataset: int = 0,
    lenient_columns: bool = False,
    apply_imputation: bool = True,
    random_state: int | None = 42,
) -> PreparedData:
    if len(datasets) not in (1, 2):
        raise ConfigError("Servono 1 dataset (con colonna gruppo) oppure 2 dataset (trattati, controlli).")
    if len(configs) == 1 and len(datasets) == 2:
        configs = [configs[0], configs[0]]
    if len(configs) != len(datasets):
        raise ConfigError(
            f"Numero di config dati ({len(configs)}) incoerente con il numero di dataset ({len(datasets)})."
        )
    names = names or [f"dataset_{i + 1}" for i in range(len(datasets))]
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ single
    if len(datasets) == 1:
        if not group_col:
            raise ConfigError("Con un solo dataset e' obbligatorio indicare la colonna gruppo (group_col).")
        df, cfg = datasets[0], configs[0]
        graw = _resolve_group_raw(df, cfg, group_col)
        work = df.with_columns(
            pl.int_range(0, df.height).alias(_ROW),
            pl.col(graw).alias(_GRAW),
        )
        work, rep = Preprocessor.run(
            work, cfg, keep_extra=[_ROW, _GRAW], lenient_columns=lenient_columns,
            apply_imputation=apply_imputation, random_state=random_state,
        )
        warnings += [f"[{names[0]}] {w}" for w in rep.warnings]
        group, tval, gw = _binarize_group(work[_GRAW], treated_value)
        warnings += [f"[{names[0]}] {w}" for w in gw]
        work = work.with_columns(group).filter(pl.col(GROUP_COL).is_not_null())
        work = _finish_ids(work, cfg).with_columns(pl.lit(names[0]).alias(SOURCE_COL)).drop(_GRAW)
        sources.append(_summary(names[0], df.height, rep, work, "mixed"))
        kinds = cfg.kinds()
        common = [c for c in cfg.new_names() if c in work.columns]
        treated_label, control_label = f"{group_col}={tval}", f"{group_col}!={tval}"
        mode = "single"
    # --------------------------------------------------------------------- two
    else:
        if treated_dataset not in (0, 1):
            raise ConfigError("treated_dataset deve essere 0 o 1.")
        frames, reports = [], []
        for i, (df, cfg) in enumerate(zip(datasets, configs)):
            work = df.with_columns(pl.int_range(0, df.height).alias(_ROW))
            work, rep = Preprocessor.run(
                work, cfg, keep_extra=[_ROW], lenient_columns=lenient_columns,
                apply_imputation=apply_imputation, random_state=random_state,
            )
            warnings += [f"[{names[i]}] {w}" for w in rep.warnings]
            work = _finish_ids(work, cfg).with_columns(
                pl.lit(names[i]).alias(SOURCE_COL),
                pl.lit(1 if i == treated_dataset else 0, dtype=pl.Int8).alias(GROUP_COL),
            )
            frames.append(work)
            reports.append(rep)
            sources.append(_summary(names[i], df.height, rep, work, "treated" if i == treated_dataset else "control"))

        sets = [set(c.new_names()) & set(f.columns) for c, f in zip(configs, frames)]
        common = [c for c in configs[0].new_names() if c in sets[0] and c in sets[1]]
        if not common:
            raise PreprocessingError(
                "I due dataset non hanno colonne attive in comune dopo il rename (new_name). "
                f"Colonne dataset 1: {sorted(sets[0])[:10]}; dataset 2: {sorted(sets[1])[:10]}"
            )
        k0, k1 = configs[0].kinds(), configs[1].kinds()
        conflicts = [c for c in common if k0.get(c) != k1.get(c)]
        if conflicts:
            raise ConfigError(
                f"Tipo (categorical/numerical) discordante tra le due config per: {conflicts}"
            )
        kinds = {c: k for c, k in k0.items()}
        kinds.update({c: k for c, k in k1.items() if c not in kinds})
        only = [sorted(s - set(common))[:6] for s in sets]
        if any(only):
            warnings.append(
                f"Colonne non comuni (presenti in una sola sorgente, nulle nell'altra): {only[0]} | {only[1]}"
            )
        work = pl.concat(frames, how="diagonal_relaxed")
        treated_label = names[treated_dataset]
        control_label = names[1 - treated_dataset]
        mode = "two"

    work = work.with_row_index(UID_COL).with_columns(pl.col(UID_COL).cast(pl.Int64))
    n1 = int((work[GROUP_COL] == 1).sum())
    n0 = int((work[GROUP_COL] == 0).sum())
    if n1 == 0 or n0 == 0:
        raise PreprocessingError(f"Servono unita' in entrambi i gruppi (trattati={n1}, controlli={n0}).")
    return PreparedData(
        data=work, kinds=kinds, common_columns=common, mode=mode, sources=sources,
        warnings=warnings, treated_label=treated_label, control_label=control_label,
    )


def _summary(name: str, n_in: int, rep: PreprocessReport, work: pl.DataFrame, role: str) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "n_rows_raw": n_in,
        "n_rows_after_preprocessing": work.height,
        "n_dropped_by_strategy": rep.n_dropped_by_strategy,
        "imputed_values": int(sum(rep.imputed_counts.values())),
        "columns_imputed": len(rep.imputed_counts),
    }
