"""Armonizzazione tra DUE dataset prima del matching.

* categoriche nominali: se i due config assegnano codici diversi alle stesse etichette (o lo stesso codice a etichette
  diverse) il secondo dataset e' ricodificato su un mapping unificato (A comanda); le ordinali non si unificano mai;
* tabella dei livelli per variabile categorica (quota nei due dataset, livelli presenti in uno solo);
* controllo numerico: statistiche affiancate, sovrapposizione dei range, possibili unita' di misura diverse, valori speciali.
Le etichette originali restano quelle di ciascun file: cambiano solo i codici interni (e il config effettivo)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from safe_matching.exceptions import PreprocessingError
from safe_matching.preprocessing.source import SourceData, column_kinds, ordinal_columns
from safe_matching.preprocessing.transform import build_inverse, norm_scalar
from safe_matching.schemas.matching_config import HarmonizationPolicy

__all__ = ["harmonize_sources", "HarmonizationReport", "HarmonizationError", "UNIT_FACTORS"]


class HarmonizationError(PreprocessingError):
    """I due dataset non sono conciliabili con le impostazioni correnti."""


UNIT_FACTORS: dict[float, str] = {
    10: "x10", 100: "x100", 1000: "x1000", 18.016: "glucose mg/dL <-> mmol/L", 38.67: "cholesterol mg/dL <-> mmol/L",
    88.4: "creatinine umol/L <-> mg/dL", 2.2046: "kg <-> lb", 2.54: "cm <-> inch", 12: "years <-> months", 365.25: "years <-> days",
}

NUMERIC_SCHEMA = {
    "variable": pl.Utf8, "kind": pl.Utf8, "n_a": pl.Int64, "n_b": pl.Int64, "pct_missing_a": pl.Float64, "pct_missing_b": pl.Float64,
    "min_a": pl.Float64, "median_a": pl.Float64, "max_a": pl.Float64, "mean_a": pl.Float64, "sd_a": pl.Float64,
    "min_b": pl.Float64, "median_b": pl.Float64, "max_b": pl.Float64, "mean_b": pl.Float64, "sd_b": pl.Float64,
    "smd_raw": pl.Float64, "median_ratio": pl.Float64, "pct_a_outside_b": pl.Float64, "pct_b_outside_a": pl.Float64,
    "dtype_a": pl.Utf8, "dtype_b": pl.Utf8, "flags": pl.Utf8, "severity": pl.Utf8,
}
CATEGORICAL_SCHEMA = {
    "variable": pl.Utf8, "code": pl.Float64, "label": pl.Utf8, "n_a": pl.Int64, "n_b": pl.Int64, "pct_a": pl.Float64,
    "pct_b": pl.Float64, "only_in": pl.Utf8,
}


@dataclass
class HarmonizationReport:
    numeric: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=NUMERIC_SCHEMA))
    categorical: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=CATEGORICAL_SCHEMA))
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    names: tuple[str, str] = ("A", "B")


# --------------------------------------------------------------------------------------------------
def _mapping(src: SourceData, name: str) -> dict[str, float]:
    raw = src.config.raw_name_of(name)
    if raw is None:
        return {}
    return {norm_scalar(k).lower(): float(v) for k, v in src.config.columns[raw].mappings.items()}


def _recode(src: SourceData, name: str, old_to_new: dict[float, float], new_labels: dict[str, float]) -> None:
    """Ricodifica la colonna e riallinea config effettivo e mappa inversa (le etichette restano quelle del file)."""
    s = src.pre[name]
    f = s.cast(pl.Float64)
    new = f.replace_strict(list(old_to_new), list(old_to_new.values()), default=f, return_dtype=pl.Float64)
    src.pre = src.pre.with_columns((new.cast(s.dtype) if s.dtype.is_integer() else new).alias(name))
    raw = src.config.raw_name_of(name)
    col = src.config.columns[raw]
    col.mappings = {k: (int(new_labels[norm_scalar(k).lower()]) if float(new_labels[norm_scalar(k).lower()]).is_integer()
                        else new_labels[norm_scalar(k).lower()]) for k in col.mappings}
    inv, order = build_inverse(col.mappings)
    src.transform.inverse[name], src.transform.order[name] = inv, order


def _unify(a: SourceData, b: SourceData, name: str, ordinal: bool, pol: HarmonizationPolicy, rep: HarmonizationReport) -> None:
    ma, mb = _mapping(a, name), _mapping(b, name)
    if not ma and not mb:
        return
    if bool(ma) != bool(mb):          # uno solo ha i mappings: l'altro deve essere gia' espresso in codici
        mapped, plain, who = (a, b, b.name) if ma else (b, a, a.name)
        codes = set((ma or mb).values())
        vals = set(plain.pre[name].drop_nulls().cast(pl.Float64).unique().to_list())
        if not vals <= codes:
            raise HarmonizationError(
                f"'{name}': {mapped.name} ha 'mappings' ma {who} no, e i suoi valori {sorted(vals - codes)[:8]} non sono codici di {mapped.name} "
                f"({sorted(codes)}). Aggiungi i 'mappings' anche alla config di {who}.")
        rep.actions.append(f"'{name}': {who} ha gia' i valori espressi come codici di {mapped.name}: accettati.")
        return
    shared = set(ma) & set(mb)
    conflicts = {lab: (ma[lab], mb[lab]) for lab in shared if ma[lab] != mb[lab]}
    codes_a = set(ma.values())
    collisions = {lab: c for lab, c in mb.items() if lab not in ma and c in codes_a}
    if not conflicts and not collisions:
        return
    disp: dict[str, str] = {}                      # etichette come scritte nei config (non normalizzate)
    for src_ in (b, a):
        for k in src_.config.columns[src_.config.raw_name_of(name)].mappings:
            disp[norm_scalar(k).lower()] = str(k)
    detail = []
    if conflicts:
        detail.append("stessa etichetta con codici diversi {" + ", ".join(
            f"{disp.get(k, k)}: {a.name}={x:g} / {b.name}={y:g}" for k, (x, y) in list(conflicts.items())[:6]) + "}")
    if collisions:
        detail.append("stesso codice per etichette diverse {" + ", ".join(f"{disp.get(k, k)} ({c:g})" for k, c in list(collisions.items())[:6]) + "}")
    if ordinal or not pol.unify_codes:
        why = "e' ordinale: l'ordine dei livelli non si unifica automaticamente" if ordinal else "options.harmonization.unify_codes=false"
        raise HarmonizationError(f"'{name}': mappings incompatibili tra {a.name} e {b.name} ({'; '.join(detail)}); {why}. "
                                 "Allinea i codici nelle due data config.")
    used = set(ma.values())
    nxt = int(max(used)) + 1 if used else 0
    unified: dict[str, float] = {}
    for lab, cb in sorted(mb.items(), key=lambda t: t[1]):
        if lab in ma:
            unified[lab] = ma[lab]
        elif cb not in used:
            unified[lab] = cb
            used.add(cb)
        else:
            unified[lab] = float(nxt)
            used.add(nxt)
            nxt += 1
    old_to_new: dict[float, float] = {}
    for lab, cb in mb.items():           # etichette fuse nello stesso codice di B ma separate in A: non separabili
        if old_to_new.setdefault(cb, unified[lab]) != unified[lab]:
            raise HarmonizationError(f"'{name}': in {b.name} le etichette con codice {cb:g} sono fuse, ma in {a.name} sono separate: "
                                     "i dati di " + b.name + " non permettono di distinguerle.")
    _recode(b, name, old_to_new, unified)
    rep.actions.append(f"'{name}': codici di {b.name} ricodificati sul mapping di {a.name} ({'; '.join(detail)}).")


def _categorical_table(a: SourceData, b: SourceData, name: str, rep_rows: list[dict]) -> None:
    sa, sb = a.pre[name].drop_nulls(), b.pre[name].drop_nulls()
    if sa.len() == 0 or sb.len() == 0:
        return
    ca = dict(zip(*[x.to_list() for x in sa.cast(pl.Float64).value_counts().get_columns()]))
    cb = dict(zip(*[x.to_list() for x in sb.cast(pl.Float64).value_counts().get_columns()]))
    labels = {}
    for src in (b, a):
        for code, lab in src.transform.inverse.get(name, {}).items():
            labels[float(code)] = lab
    na, nb = sa.len(), sb.len()
    for code in sorted(set(ca) | set(cb))[:40]:
        x, y = int(ca.get(code, 0)), int(cb.get(code, 0))
        rep_rows.append({"variable": name, "code": float(code), "label": labels.get(float(code), f"{code:g}"), "n_a": x, "n_b": y,
                         "pct_a": 100 * x / na, "pct_b": 100 * y / nb, "only_in": "A" if y == 0 else ("B" if x == 0 else "")})


def _numeric_row(a: SourceData, b: SourceData, name: str, kind: str, pol: HarmonizationPolicy) -> dict | None:
    xa = a.pre[name].cast(pl.Float64).drop_nulls().to_numpy()
    xb = b.pre[name].cast(pl.Float64).drop_nulls().to_numpy()
    if xa.size < 2 or xb.size < 2:
        return None
    sa, sb = float(xa.std(ddof=1)), float(xb.std(ddof=1))
    pooled = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
    smd = float((xa.mean() - xb.mean()) / pooled) if pooled > 0 else 0.0
    ma, mb = float(np.median(xa)), float(np.median(xb))
    ratio = ma / mb if ma * mb > 0 else float("nan")
    out_a = float(np.mean((xa < xb.min()) | (xa > xb.max())) * 100)
    out_b = float(np.mean((xb < xa.min()) | (xb > xa.max())) * 100)
    flags: list[str] = []
    if min(100 - out_a, 100 - out_b) < 100 * pol.overlap_warn_below:
        flags.append(f"low range overlap ({100 - out_a:.0f}% of A inside B, {100 - out_b:.0f}% of B inside A)")
    if pol.unit_check and abs(smd) > 1 and np.isfinite(ratio):
        r = max(ratio, 1 / ratio)
        for f, lab in UNIT_FACTORS.items():
            if abs(r / f - 1) < 0.08:
                flags.append(f"possible unit mismatch (median ratio {r:.1f} ~ x{f:g}: {lab})")
                break
    for tag, x in (("A", xa), ("B", xb)):
        med = float(np.median(x))
        rsd = float(np.subtract(*np.percentile(x, [75, 25])) / 1.349)
        if rsd > 0:
            for sv in pol.sentinel_values:
                cnt = int((x == sv).sum())
                if cnt >= 3 and abs(sv - med) / rsd > 6:
                    flags.append(f"sentinel-like value {sv:g} in {tag} (n={cnt})")
    return {
        "variable": name, "kind": kind, "n_a": int(xa.size), "n_b": int(xb.size),
        "pct_missing_a": 100 * (1 - xa.size / a.pre.height), "pct_missing_b": 100 * (1 - xb.size / b.pre.height),
        "min_a": float(xa.min()), "median_a": ma, "max_a": float(xa.max()), "mean_a": float(xa.mean()), "sd_a": sa,
        "min_b": float(xb.min()), "median_b": mb, "max_b": float(xb.max()), "mean_b": float(xb.mean()), "sd_b": sb,
        "smd_raw": smd, "median_ratio": ratio, "pct_a_outside_b": out_a, "pct_b_outside_a": out_b,
        "dtype_a": str(a.pre.schema[name]), "dtype_b": str(b.pre.schema[name]), "flags": "; ".join(flags),
        "severity": "warn" if flags else "ok",
    }


def harmonize_sources(
    sources: list[SourceData], *, variables: list[str], policy: HarmonizationPolicy,
) -> tuple[list[SourceData], HarmonizationReport]:
    """``variables``: new_name delle variabili usate (matching, stratificazione, evaluation, outcome). Modifica i SourceData
    in place (ricodifica) e ritorna anche il report."""
    if len(sources) != 2:
        return sources, HarmonizationReport()
    a, b = sources
    rep = HarmonizationReport(names=(a.name, b.name))
    ka, kb = column_kinds(a.config), column_kinds(b.config)
    oa, ob = ordinal_columns(a.config), ordinal_columns(b.config)
    common = [v for v in dict.fromkeys(variables) if v in a.pre.columns and v in b.pre.columns]
    skipped = [v for v in dict.fromkeys(variables) if v not in common]
    if skipped:
        rep.warnings.append(f"Variabili presenti in un solo dataset (non confrontabili): {skipped}")
    cat_rows: list[dict] = []
    num_rows: list[dict] = []
    for v in common:
        kind_a = ka.get(v)
        if (v in oa) != (v in ob):
            raise HarmonizationError(f"'{v}' e' ordinale in un solo config: dichiarala allo stesso modo (categorical+numerical) in entrambi.")
        is_ord = v in oa
        if kind_a == "categorical" or is_ord or kb.get(v) == "categorical":
            _unify(a, b, v, is_ord, policy, rep)
            _categorical_table(a, b, v, cat_rows)
        elif kind_a == "numerical" and policy.check_numeric:
            row = _numeric_row(a, b, v, "numerical", policy)
            if row:
                num_rows.append(row)
    rep.numeric = pl.DataFrame(num_rows, schema=NUMERIC_SCHEMA) if num_rows else pl.DataFrame(schema=NUMERIC_SCHEMA)
    rep.categorical = pl.DataFrame(cat_rows, schema=CATEGORICAL_SCHEMA) if cat_rows else pl.DataFrame(schema=CATEGORICAL_SCHEMA)
    for src in (a, b):
        if src.mapping_additions:
            rep.actions.append(f"[{src.name}] etichette non tracciate aggiunte ai mappings (on_unmapped='union'): {src.mapping_additions}")
    for r in rep.numeric.filter(pl.col("severity") == "warn").iter_rows(named=True):
        rep.warnings.append(f"'{r['variable']}': {r['flags']}")
        if policy.error_on_unit_mismatch and "unit mismatch" in r["flags"]:
            raise HarmonizationError(f"'{r['variable']}': {r['flags']}. Converti le unita' (o imposta error_on_unit_mismatch=false).")
    return sources, rep