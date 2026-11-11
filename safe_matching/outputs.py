"""Dataset di output: colonne ORIGINALI (nomi, etichette, tipi del file) + metadati di matching."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from safe_matching.preprocessing.assembly import PreparedData, ROW_COL, SRC_IDX_COL
from safe_matching.preprocessing.transform import infer_types

__all__ = ["build_output", "write_table", "META_COLUMNS"]
log = logging.getLogger("safe_matching")

META_COLUMNS: dict[str, str] = {
    "match_role": "match_role",          # query | match | selected
    "pair_id": "match_pair_id",
    "weight": "match_weight",
    "distance": "match_distance",
    "stratum": "match_stratum",
    "__group__": "match_group",          # 1 = trattato, 0 = controllo
    "__source__": "match_source",
    "__row__": "source_row",             # posizione nel file originale (0-based)
    "__id__": "match_id",
    "__uid__": "match_uid",
    "__ps__": "match_ps",
    "__logit_ps__": "match_logit_ps",
}


def build_output(
    table: pl.DataFrame,
    prepared: PreparedData,
    *,
    values: str = "postprocessed",
    restore_inactive_columns: bool = True,
) -> pl.DataFrame:
    """Converte una tabella del matching (nomi nuovi) nel formato dei dati originali.

    ``values='postprocessed'``: colonne attive riportate a nomi ed etichette del file (``SourceData.to_original``; i valori
    imputati restano) piu', se ``restore_inactive_columns``, le colonne del file non attive/non configurate.
    ``values='raw'``: le righe del file cosi' come sono (i mancanti restano tali).
    In entrambi i casi si aggiungono i metadati ``match_*``."""
    if values not in ("postprocessed", "raw"):
        raise ValueError("values deve essere 'postprocessed' o 'raw'")
    parts: list[pl.DataFrame] = []
    for idx, src in enumerate(prepared.sources):
        part = table.filter(pl.col(SRC_IDX_COL) == idx)
        if part.height == 0:
            continue
        raw_rows = infer_types(src.raw[part[ROW_COL].to_list()])
        if values == "raw":
            body = raw_rows
        else:
            names = [c.new_name for c in src.config.active_columns().values() if c.new_name in part.columns]
            post = src.to_original(part.select(list(dict.fromkeys(names))))
            if restore_inactive_columns:
                extra = [c for c in src.raw.columns if c not in post.columns]
                body = pl.concat([post, raw_rows.select(extra)], how="horizontal_extend") if extra else post
                body = body.select([c for c in src.raw.columns if c in body.columns] +
                                   [c for c in body.columns if c not in src.raw.columns])
            else:
                body = post
        meta_src = [c for c in META_COLUMNS if c in part.columns]
        meta = part.select(meta_src).rename({c: META_COLUMNS[c] for c in meta_src})
        imp_cols = [c for c in part.columns if c.startswith("__imp_")]
        if imp_cols:   # tracciamento dei valori imputati: imputed_<nome originale> = 1 se la cella e' stata imputata
            imp = part.select([pl.col(c).cast(pl.Int8) for c in imp_cols]).rename(
                {c: "imputed_" + (src.config.raw_name_of(c[6:]) or c[6:]) for c in imp_cols})
            meta = pl.concat([meta, imp], how="horizontal_extend")
        clash = [c for c in meta.columns if c in body.columns]
        if clash:
            body = body.rename({c: f"{c}_orig" for c in clash})
        parts.append(pl.concat([body, meta], how="horizontal_extend"))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def write_table(df: pl.DataFrame, base: Path, formats: tuple[str, ...] | list[str]) -> dict[str, Path]:
    """Scrive ``df`` in uno o piu' formati ('csv', 'xlsx', 'parquet'). Ritorna {nome_file: percorso}."""
    out: dict[str, Path] = {}
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        try:
            if fmt == "csv":
                df.write_csv(path)
            elif fmt == "xlsx":
                df.write_excel(path)
            elif fmt == "parquet":
                df.write_parquet(path)
            else:
                log.warning("Formato di output '%s' non supportato (usa csv, xlsx, parquet).", fmt)
                continue
            out[path.name] = path
        except Exception as exc:  # noqa: BLE001
            log.warning("Scrittura %s fallita: %s", path.name, exc)
    return out