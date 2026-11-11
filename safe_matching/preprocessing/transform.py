"""Applicazione della data config: rename, token mancanti, mapping, cast, multiplier - e conversione inversa in uscita.

NON si scala mai e non si imputa qui (l'imputazione e' un passo separato). Gli errori sui dati sono raccolti per colonna e
sollevati insieme, con i valori incriminati e il loro conteggio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from safe_matching.exceptions import ConfigError, PreprocessingError
from safe_matching.preprocessing.config import ColumnConfig, DataConfig, canonical_type

__all__ = ["apply_config", "TransformResult", "ColumnQC", "UnmappedValuesError", "CastError", "norm_text", "infer_types",
           "inverse_transform", "norm_scalar", "build_inverse"]

BASE_MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})
_TRUE = {"true", "t", "yes", "y", "si", "sì", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


class UnmappedValuesError(PreprocessingError):
    """Valori presenti nei dati ma assenti da ``mappings``. ``unmapped`` = {new_name: {valore: conteggio}}."""

    def __init__(self, message: str, unmapped: dict[str, dict[str, int]] | None = None) -> None:
        super().__init__(message)
        self.unmapped = unmapped or {}


class CastError(PreprocessingError):
    """Valori non convertibili nel tipo dichiarato."""


@dataclass
class ColumnQC:
    raw_name: str
    new_name: str
    kind: str | None
    n_rows: int
    n_missing: int = 0
    n_unmapped: int = 0
    n_code_passthrough: int = 0
    n_cast_failed: int = 0
    dtype: str = ""


@dataclass
class TransformResult:
    data: pl.DataFrame
    qc: list[ColumnQC]
    inverse: dict[str, dict[str, str]]
    """{new_name: {codice (testo normalizzato): etichetta originale}} per le colonne con mappings."""
    order: dict[str, list[tuple[float, str]]]
    """{new_name: [(codice, etichetta)] ordinati per codice} - per etichettare grafici e tabelle."""
    unmapped: dict[str, dict[str, int]] = field(default_factory=dict)
    cast_failures: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------
def norm_text(s: pl.Series) -> pl.Series:
    """Testo confrontabile: spazi tolti, numeri interi come '1' (non '1.0'), NaN -> null. Vuoto resta ''."""
    name = s.name
    if s.dtype == pl.Boolean:
        s = s.cast(pl.Int64)
    if s.dtype.is_numeric():
        f = s.cast(pl.Float64).fill_nan(None)
        txt = f.cast(pl.Utf8)
    else:
        txt = s.cast(pl.Utf8).str.strip_chars()
        f = txt.cast(pl.Float64, strict=False)
    ok = f.is_not_null() & f.is_finite() & (f == f.floor()) & (f.abs() < 1e15)
    out = pl.DataFrame({"t": txt, "f": f, "ok": ok}).select(
        pl.when(pl.col("ok")).then(pl.col("f").cast(pl.Int64).cast(pl.Utf8)).otherwise(pl.col("t")).alias(name))
    return out.to_series()


def _norm_scalar(x: Any) -> str:
    return norm_text(pl.Series("x", [x], dtype=pl.Utf8 if isinstance(x, str) else None)).item()

norm_scalar = _norm_scalar


def build_inverse(mappings: dict[str, Any]) -> tuple[dict[str, str], list[tuple[float, str]]]:
    """({codice normalizzato: prima etichetta}, [(codice, etichetta)] ordinati per codice) da un dict di mappings."""
    inv: dict[str, str] = {}
    for k, v in mappings.items():
        inv.setdefault(_norm_scalar(v), str(k))
    order = sorted(((float(v), inv[_norm_scalar(v)]) for v in mappings.values()), key=lambda t: t[0])
    return inv, list(dict.fromkeys(order))

def _tokens(cfg: ColumnConfig) -> set[str]:
    return set(BASE_MISSING) | {str(t).strip().lower() for t in cfg.missing_data_management.missing_values}


def _to_number(txt: pl.Series, decimal_comma: bool = True) -> pl.Series:
    t = txt.str.replace(r"^\s*(-?\d+),(\d+)\s*$", "${1}.${2}") if decimal_comma else txt
    return t.cast(pl.Float64, strict=False)


def _examples(txt: pl.Series, mask: pl.Series, k: int = 15) -> dict[str, int]:
    vc = txt.filter(mask).value_counts(sort=True).head(k)
    return {str(a): int(b) for a, b in vc.iter_rows()}


# --------------------------------------------------------------------------------------------------
def apply_config(
    raw: pl.DataFrame,
    cfg: DataConfig,
    *,
    name: str = "dataset",
    on_unmapped: str = "error",
    on_cast_error: str = "error",
    accept_codes: bool = True,
    case_sensitive: bool = False,
    strict_columns: bool = True,
) -> TransformResult:
    """``on_unmapped``/``on_cast_error``: 'error' (default) oppure 'missing' (il valore diventa mancante, con avviso).
    ``accept_codes``: un valore che non e' una chiave ma coincide con un CODICE del mapping (dati gia' codificati) e' accettato."""
    if on_unmapped not in ("error", "missing") or on_cast_error not in ("error", "missing"):
        raise ConfigError("on_unmapped / on_cast_error devono essere 'error' o 'missing'.")
    active = cfg.active_columns()
    missing_cols = [r for r in active if r not in raw.columns]
    warnings: list[str] = []
    if missing_cols:
        if strict_columns:
            raise PreprocessingError(
                f"[{name}] colonne attive della data config assenti dal file: {missing_cols}. "
                f"Colonne del file (prime 25): {raw.columns[:25]}. Correggi la config (active: false) o il file.")
        warnings.append(f"Colonne attive assenti dal file e ignorate: {missing_cols}")
        active = {r: c for r, c in active.items() if r in raw.columns}
    if not active:
        raise PreprocessingError(f"[{name}] nessuna colonna attiva disponibile nel file.")

    cols: list[pl.Series] = []
    qc: list[ColumnQC] = []
    inverse: dict[str, dict[str, str]] = {}
    order: dict[str, list[tuple[float, str]]] = {}
    unmapped: dict[str, dict[str, int]] = {}
    failures: dict[str, dict[str, int]] = {}

    for rawname, c in active.items():
        n = raw.height
        txt = norm_text(raw[rawname])
        low = txt.str.to_lowercase()
        is_missing = txt.is_null() | low.is_in(list(_tokens(c)))
        txt = pl.DataFrame({"t": txt, "m": is_missing}).select(pl.when(pl.col("m")).then(None).otherwise(pl.col("t")).alias(rawname)).to_series()
        info = ColumnQC(rawname, c.new_name, c.kind, n, n_missing=int(txt.null_count()))
        present = txt.is_not_null()

        if c.mappings:
            keys = {(_norm_scalar(k) if case_sensitive else _norm_scalar(k).lower()): v for k, v in c.mappings.items()}
            codes_numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in keys.values())
            if not codes_numeric:
                raise ConfigError(f"[{name}] '{rawname}': i valori di 'mappings' devono essere numerici (codici). Trovati: "
                                  f"{[v for v in keys.values() if not isinstance(v, (int, float))][:5]}")
            cmp = txt if case_sensitive else txt.str.to_lowercase()
            mapped = cmp.replace_strict(list(keys), [float(v) for v in keys.values()], default=None, return_dtype=pl.Float64)
            miss_map = present & mapped.is_null()
            n_pass = 0
            if accept_codes and miss_map.any():
                code_strs = {_norm_scalar(v) for v in keys.values()}
                pt = miss_map & cmp.is_in(list(code_strs))
                if pt.any():
                    mapped = pl.DataFrame({"m": mapped, "p": pt, "n": txt.cast(pl.Float64, strict=False)}).select(
                        pl.when(pl.col("p")).then(pl.col("n")).otherwise(pl.col("m"))).to_series()
                    n_pass = int(pt.sum())
                    miss_map = present & mapped.is_null()
            info.n_code_passthrough, info.n_unmapped = n_pass, int(miss_map.sum())
            if miss_map.any():
                unmapped[c.new_name] = _examples(txt, miss_map)
            s = mapped
            if all(float(v).is_integer() for v in keys.values()):
                s = s.cast(pl.Int64, strict=False)
            inverse[c.new_name], order[c.new_name] = build_inverse(c.mappings)
        else:
            s, failed = _cast(txt, present, c)
            info.n_cast_failed = int(failed.sum())
            if failed.any():
                failures[c.new_name] = _examples(txt, failed)
            if c.kind == "ordinal" and s.dtype not in (pl.Int64, pl.Float64):
                raise ConfigError(f"[{name}] '{rawname}' e' ordinale (categorical+numerical) ma senza 'mappings' i valori devono essere numerici.")

        if c.multiplier != 1 and s.dtype.is_numeric() and not c.mappings:
            s = s.cast(pl.Float64) * c.multiplier
        if c.scale:
            warnings.append(f"'{rawname}': scale=true ignorato (il framework non scala i dati).")
        info.dtype = str(s.dtype)
        cols.append(s.alias(c.new_name))
        qc.append(info)

    problems: list[str] = []
    if unmapped and on_unmapped == "error":
        for col, vals in unmapped.items():
            problems.append(f"  - '{col}' (file: '{cfg.raw_name_of(col)}'): valori non presenti in 'mappings': {vals}")
        raise UnmappedValuesError(
            f"[{name}] valori non mappati.\n" + "\n".join(problems) +
            "\nAggiungili a 'mappings' della data config (o usa on_unmapped='missing' per trattarli come mancanti, "
            "'union' per aggiungerli automaticamente alle variabili nominali).", unmapped=unmapped)
    if unmapped:
        warnings.append(f"Valori non mappati resi mancanti: { {k: sum(v.values()) for k, v in unmapped.items()} }")
    if failures and on_cast_error == "error":
        for col, vals in failures.items():
            problems.append(f"  - '{col}' (file: '{cfg.raw_name_of(col)}'): non convertibili in {canonical_type(cfg.columns[cfg.raw_name_of(col)].type) or 'numero'}: {vals}")
        raise CastError(f"[{name}] valori non convertibili nel tipo dichiarato.\n" + "\n".join(problems) +
                        "\nCorreggi i dati o il 'type' nella config (o usa on_cast_error='missing').")
    if failures:
        warnings.append(f"Valori non convertibili resi mancanti: { {k: sum(v.values()) for k, v in failures.items()} }")
    for q in qc:
        if q.n_code_passthrough:
            warnings.append(f"'{q.new_name}': {q.n_code_passthrough} valori gia' espressi come codice e accettati come tali.")

    data = pl.DataFrame(cols)
    return TransformResult(data, qc, inverse, order, unmapped, failures, warnings)


def _cast(txt: pl.Series, present: pl.Series, c: ColumnConfig) -> tuple[pl.Series, pl.Series]:
    """Cast di una colonna SENZA mapping. Ritorna (serie tipizzata, maschera dei valori non convertibili)."""
    t = canonical_type(c.type)
    if t is None:
        t = "float" if c.kind in ("numerical", "ordinal") else None
    if t == "str" or (t is None and c.kind is None and not c.id):
        return txt, pl.Series([False] * len(txt))
    if t is None:  # categorica senza tipo: numerica se possibile, altrimenti testo
        num = _to_number(txt)
        if not (present & num.is_null()).any():
            return _shrink(num), pl.Series([False] * len(txt))
        return txt, pl.Series([False] * len(txt))
    if t in ("int", "float"):
        num = _to_number(txt)
        bad = present & num.is_null()
        if t == "int":
            frac = num.is_not_null() & (num != num.floor())
            clean = pl.DataFrame({"n": num, "f": frac}).select(pl.when(pl.col("f")).then(None).otherwise(pl.col("n"))).to_series()
            return clean.cast(pl.Int64, strict=False), bad | frac
        return num, bad
    if t == "bool":
        low = txt.str.to_lowercase()
        val = pl.DataFrame({"l": low}).select(
            pl.when(pl.col("l").is_in(list(_TRUE))).then(1).when(pl.col("l").is_in(list(_FALSE))).then(0).otherwise(None).alias("v")).to_series()
        return val.cast(pl.Int64), present & val.is_null()
    dt = txt.str.to_datetime(strict=False)
    return dt, present & dt.is_null()


def _shrink(num: pl.Series) -> pl.Series:
    nn = num.drop_nulls()
    return num.cast(pl.Int64, strict=False) if nn.len() and bool((nn == nn.floor()).all()) else num


def infer_types(df: pl.DataFrame) -> pl.DataFrame:
    """Colonne testuali interamente numeriche (a meno dei vuoti) -> numeriche; le altre restano testo."""
    out = []
    for c in df.columns:
        s = df[c]
        if s.dtype in (pl.Utf8, pl.String):
            t = norm_text(s)
            present = t.is_not_null() & (t != "")
            num = _to_number(s.str.strip_chars())
            if present.any() and not (present & num.is_null()).any():
                out.append(_shrink(num).alias(c))
                continue
            out.append(s.str.strip_chars().replace("", None).alias(c))
        else:
            out.append(s)
    return pl.DataFrame(out)


def inverse_transform(df: pl.DataFrame, cfg: DataConfig, result: TransformResult, *, to_labels: bool = True) -> pl.DataFrame:
    """Dai nomi/valori del framework a quelli del file: rename a nome originale, codici -> etichette, tipi del file."""
    out: list[pl.Series] = []
    for raw_name, c in cfg.active_columns().items():
        if c.new_name not in df.columns:
            continue
        s = df[c.new_name]
        inv = result.inverse.get(c.new_name)
        if inv and to_labels:
            keys = norm_text(s)
            s = keys.replace_strict(list(inv), list(inv.values()), default=None, return_dtype=pl.Utf8)
        elif c.mappings is None or not c.mappings:
            t = canonical_type(c.type)
            if t == "int" and s.dtype.is_float():
                s = s.round(0).cast(pl.Int64, strict=False)
        out.append(s.alias(raw_name))
    return pl.DataFrame(out) if out else pl.DataFrame()