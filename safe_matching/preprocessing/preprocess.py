"""Applicazione di una DataConfig a un DataFrame grezzo.

Ordine: selezione colonne attive -> rename -> mappings -> token mancanti -> cast
-> multiplier -> drop/replace -> imputazione. Le colonne ``keep_extra`` (es. il
gruppo e l'indice di riga) attraversano il preprocessing senza modifiche.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from safe_matching.exceptions import PreprocessingError
from safe_matching.preprocessing.imputation import impute_dataframe
from safe_matching.schemas.column_info import ColumnInfo, ColumnType, MissingStrategy
from safe_matching.schemas.data_config import DataConfig

__all__ = ["Preprocessor", "PreprocessReport", "key_string"]

_INT_TYPES = {ColumnType.INT, ColumnType.INT32, ColumnType.INT64}
_FLOAT_TYPES = {ColumnType.FLOAT, ColumnType.FLOAT32, ColumnType.FLOAT64}


@dataclass
class PreprocessReport:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_dropped_by_strategy: int = 0
    dropped_inactive_missing_columns: list[str] = field(default_factory=list)
    imputed_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def key_string(s: pl.Series) -> pl.Series:
    """Rappresentazione testuale stabile per confrontare/mappare valori ('1.0' -> '1')."""
    if s.dtype == pl.Boolean:
        s = s.cast(pl.Int64)
    if s.dtype.is_float():
        as_int = s.cast(pl.Int64, strict=False)
        integral = s.is_not_null() & (s == s.floor())
        return pl.select(
            pl.when(integral).then(as_int.cast(pl.Utf8)).otherwise(s.cast(pl.Utf8))
        ).to_series().alias(s.name)
    return s.cast(pl.Utf8)


class Preprocessor:
    @staticmethod
    def run(
        df: pl.DataFrame,
        config: DataConfig,
        *,
        keep_extra: list[str] | None = None,
        lenient_columns: bool = False,
        apply_imputation: bool = True,
        random_state: int | None = 42,
    ) -> tuple[pl.DataFrame, PreprocessReport]:
        report = PreprocessReport(n_rows_in=df.height)
        extras = list(keep_extra or [])
        active = dict(config.active_columns())

        missing = [raw for raw in active if raw not in df.columns]
        if missing:
            if not lenient_columns:
                raise PreprocessingError(
                    f"Colonne attive nella DataConfig assenti dal dataset: {missing}. "
                    "Imposta 'active': false oppure usa --lenient-columns."
                )
            report.dropped_inactive_missing_columns = missing
            report.warnings.append(f"Colonne attive assenti dal dataset e ignorate: {missing}")
            active = {raw: info for raw, info in active.items() if raw not in missing}
        if not active:
            raise PreprocessingError("Nessuna colonna attiva disponibile nel dataset.")

        collisions = set(extras) & {i.new_name for i in active.values()}
        if collisions:
            raise PreprocessingError(f"Colonne riservate in conflitto con new_name: {sorted(collisions)}")

        out = df.select(list(active.keys()) + [c for c in extras if c not in active])
        rename = {raw: i.new_name for raw, i in active.items() if raw != i.new_name}
        if rename:
            out = out.rename(rename)

        for info in active.values():
            name = info.new_name
            s = out[name]
            if info.mappings:
                s = Preprocessor._apply_mapping(s, info, report)
            s = Preprocessor._normalize_missing(s, info)
            s = Preprocessor._cast(s, info)
            if info.numerical and info.multiplier != 1 and s.dtype.is_numeric():
                s = s * info.multiplier
            out = out.with_columns(s.alias(name))

        # drop / replace
        drop_cols = [i.new_name for i in active.values() if i.missing_data_management.strategy == MissingStrategy.DROP]
        if drop_cols:
            before = out.height
            out = out.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in drop_cols]))
            report.n_dropped_by_strategy = before - out.height
        for info in active.values():
            mdm = info.missing_data_management
            if mdm.strategy == MissingStrategy.REPLACE:
                out = out.with_columns(pl.col(info.new_name).fill_null(mdm.value))

        if apply_imputation:
            out, counts, warns = impute_dataframe(out, active, random_state=random_state)
            report.imputed_counts = counts
            report.warnings.extend(warns)

        report.n_rows_out = out.height
        return out, report

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_mapping(s: pl.Series, info: ColumnInfo, report: PreprocessReport) -> pl.Series:
        str_map = {str(k): str(v) for k, v in info.mappings.items()}
        keys = key_string(s)
        mapped = keys.replace(str_map)
        known = set(str_map) | set(str_map.values())
        unknown = (
            mapped.drop_nulls().filter(~mapped.drop_nulls().is_in(list(known))).unique().to_list()
        )
        if unknown:
            report.warnings.append(
                f"'{info.new_name}': valori non coperti da 'mappings' (diventeranno mancanti se il cast fallisce): "
                f"{unknown[:8]}"
            )
        return mapped.alias(s.name)

    @staticmethod
    def _normalize_missing(s: pl.Series, info: ColumnInfo) -> pl.Series:
        mdm = info.missing_data_management
        if mdm.strategy == MissingStrategy.MAINTAIN or not mdm.condition:
            return s
        if s.dtype not in (pl.Utf8, pl.String, pl.Categorical):
            return s
        tokens = [str(t) for t in mdm.condition]
        txt = s.cast(pl.Utf8)
        return pl.select(
            pl.when(txt.str.strip_chars().is_in([t.strip() for t in tokens]) | txt.is_in(tokens))
            .then(None)
            .otherwise(txt)
        ).to_series().alias(s.name)

    @staticmethod
    def _cast(s: pl.Series, info: ColumnInfo) -> pl.Series:
        target = info.type
        is_text = s.dtype in (pl.Utf8, pl.String, pl.Categorical)
        if target is None:
            if is_text and (info.numerical or info.categorical):
                numeric = s.cast(pl.Utf8).cast(pl.Float64, strict=False)
                if numeric.null_count() == s.null_count():
                    return Preprocessor._shrink_if_integral(numeric)
            return s
        if target in _INT_TYPES:
            base = s.cast(pl.Utf8).cast(pl.Float64, strict=False) if is_text else s
            dtype = {ColumnType.INT: pl.Int64, ColumnType.INT64: pl.Int64, ColumnType.INT32: pl.Int32}[target]
            return base.cast(dtype, strict=False)
        if target in _FLOAT_TYPES:
            dtype = pl.Float32 if target == ColumnType.FLOAT32 else pl.Float64
            base = s.cast(pl.Utf8) if is_text else s
            return base.cast(dtype, strict=False)
        if target == ColumnType.STRING:
            return s.cast(pl.Utf8)
        if target == ColumnType.BOOL:
            if is_text:
                return s.cast(pl.Utf8).str.to_lowercase().replace({"true": "1", "false": "0"}).cast(pl.Float64, strict=False).cast(pl.Boolean, strict=False)
            return s.cast(pl.Boolean, strict=False)
        if target == ColumnType.DATE:
            return s.str.to_date(strict=False) if is_text else s.cast(pl.Date, strict=False)
        if target == ColumnType.DATETIME:
            return s.str.to_datetime(strict=False) if is_text else s.cast(pl.Datetime, strict=False)
        if target == ColumnType.CATEGORY:
            return s.cast(pl.Utf8).cast(pl.Categorical)
        return s

    @staticmethod
    def _shrink_if_integral(s: pl.Series) -> pl.Series:
        nn = s.drop_nulls()
        if nn.len() and bool((nn == nn.floor()).all()):
            return s.cast(pl.Int64, strict=False)
        return s
