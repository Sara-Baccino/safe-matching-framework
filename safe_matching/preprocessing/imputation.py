"""Imputazione dei valori mancanti secondo ``missing_data_management``."""

from __future__ import annotations

import numpy as np
import polars as pl

from safe_matching.schemas.column_info import ColumnInfo, ImputerType, MissingStrategy

__all__ = ["impute_dataframe"]


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _round_like_categorical(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    out = np.rint(values)
    if observed.size:
        out = np.clip(out, observed.min(), observed.max())
    return out


def impute_dataframe(
    df: pl.DataFrame,
    columns: dict[str, ColumnInfo],
    *,
    random_state: int | None = 42,
    knn_neighbors: int = 5,
) -> tuple[pl.DataFrame, dict[str, int], list[str]]:
    """Imputa le colonne con strategy='impute'.

    :param columns: {nome_originale: ColumnInfo}; si usa ``new_name`` per accedere al df.
    :return: (df imputato, {colonna: n_valori_imputati}, warning)
    """
    warnings: list[str] = []
    counts: dict[str, int] = {}

    # NaN float -> null, cosi' un solo concetto di "mancante"
    float_cols = [c for c in df.columns if df.schema[c].is_float()]
    if float_cols:
        df = df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])

    targets: dict[str, tuple[ColumnInfo, ImputerType]] = {}
    for info in columns.values():
        mdm = info.missing_data_management
        if info.id or mdm.strategy != MissingStrategy.IMPUTE:
            continue
        if info.new_name not in df.columns or df[info.new_name].null_count() == 0:
            continue
        targets[info.new_name] = (info, mdm.imputer)

    if not targets:
        return df, counts, warnings

    # colonne non numeriche: solo valore piu' frequente
    for name in list(targets):
        if not _is_numeric(df.schema[name]):
            if df.schema[name] in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
                mode = df[name].drop_nulls().mode()
                if mode.len():
                    n = df[name].null_count()
                    df = df.with_columns(pl.col(name).fill_null(mode[0]))
                    counts[name] = n
                    warnings.append(f"'{name}': colonna non numerica imputata con il valore piu' frequente.")
            else:
                warnings.append(f"'{name}': tipo {df.schema[name]} non imputabile, lasciata invariata.")
            targets.pop(name)

    # imputer semplici per colonna
    for name, (info, imp) in list(targets.items()):
        if imp not in (ImputerType.ZERO, ImputerType.MEAN, ImputerType.MEDIAN, ImputerType.MOST_FREQUENT):
            continue
        s = df[name]
        n = s.null_count()
        if imp == ImputerType.ZERO:
            fill = 0
        elif imp == ImputerType.MEAN:
            fill = s.mean()
        elif imp == ImputerType.MEDIAN:
            fill = s.median()
        else:
            mode = s.drop_nulls().mode()
            fill = mode[0] if mode.len() else 0
        if fill is None:
            fill = 0
        if info.categorical and imp in (ImputerType.MEAN, ImputerType.MEDIAN):
            fill = round(float(fill))
        filled = s.fill_null(fill)
        if s.dtype.is_integer() and not float(fill).is_integer():
            filled = s.cast(pl.Float64).fill_null(float(fill))
        df = df.with_columns(filled.alias(name))
        counts[name] = n
        targets.pop(name)

    # imputer multivariati (knn / iterative) su tutte le colonne numeriche non-id attive
    if targets:
        from sklearn.impute import KNNImputer

        feature_cols = [
            i.new_name
            for i in columns.values()
            if not i.id and i.new_name in df.columns and _is_numeric(df.schema[i.new_name])
        ]
        X = df.select(feature_cols).cast(pl.Float64).to_numpy().astype(float)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
        Z = (X - mu) / sd

        want_iterative = any(imp == ImputerType.ITERATIVE for _, imp in targets.values())
        want_knn = any(imp == ImputerType.KNN for _, imp in targets.values())
        Zk = Zi = None
        if want_knn:
            k = max(1, min(knn_neighbors, df.height - 1))
            Zk = KNNImputer(n_neighbors=k, keep_empty_features=True).fit_transform(Z)
        if want_iterative:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            Zi = IterativeImputer(max_iter=10, random_state=random_state, keep_empty_features=True).fit_transform(Z)

        new_cols = []
        for name, (info, imp) in targets.items():
            j = feature_cols.index(name)
            filled = (Zk if imp == ImputerType.KNN else Zi)[:, j] * sd[j] + mu[j]
            observed = X[:, j][~np.isnan(X[:, j])]
            if info.categorical or df.schema[name].is_integer():
                filled = _round_like_categorical(filled, observed)
            original = df[name]
            n = original.null_count()
            merged = np.where(np.isnan(X[:, j]), filled, X[:, j])
            series = pl.Series(name, merged)
            if original.dtype.is_integer():
                series = series.cast(original.dtype, strict=False)
            new_cols.append(series)
            counts[name] = n
        df = df.with_columns(new_cols)

    return df, counts, warnings
