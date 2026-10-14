"""Report PDF (ReportLab/Platypus) dei risultati di matching."""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from xml.sax.saxutils import escape

from safe_matching.evaluation.balance import BalanceResult
from safe_matching.evaluation.hooks import HookResult
from safe_matching.exceptions import ReportError
from safe_matching.matching.engine import MatchingOutcome
from safe_matching.preprocessing.assembly import PreparedData
from safe_matching.schemas.matching_config import MatchingConfig
from safe_matching.version import __version__

__all__ = ["generate_report"]

_NAVY, _GREY, _LIGHT = colors.HexColor("#1A2B4C"), colors.HexColor("#4A5568"), colors.HexColor("#F4F6F9")
_OK, _WARN, _BAD = colors.HexColor("#E6F4EA"), colors.HexColor("#FFF4E0"), colors.HexColor("#FDE8E8")
_W = 17.5 * cm


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "-" if math.isnan(x) else (f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.3g}")
    return str(x)


def _styles() -> dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                                textColor=_NAVY, alignment=0, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=_GREY, leading=12),
        "h1": ParagraphStyle("h1", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=_NAVY,
                             spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, textColor=_NAVY,
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12.5),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.5, leading=9.5),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=7.8, leading=9.6),
        "cellb": ParagraphStyle("cb", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=9.6,
                                textColor=colors.white),
        "cap": ParagraphStyle("cap", parent=ss["Italic"], fontSize=8, textColor=_GREY, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8),
    }


def _table(rows: list[list[Any]], widths: list[float], st: dict, *, header: bool = True,
           row_colors: dict[int, Any] | None = None, zebra: bool = True) -> Table:
    cells = []
    for r, row in enumerate(rows):
        style = st["cellb"] if (header and r == 0) else st["cell"]
        cells.append([Paragraph(escape(_fmt(c)), style) if not isinstance(c, Paragraph) else c for c in row])
    t = Table(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE1E7")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), _NAVY))
    if zebra:
        cmds.append(("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, _LIGHT]))
    for r, c in (row_colors or {}).items():
        cmds.append(("BACKGROUND", (0, r), (-1, r), c))
    t.setStyle(TableStyle(cmds))
    return t


def _kv(pairs: list[tuple[str, Any]], st: dict, w1: float = 5.5 * cm) -> Table:
    rows = [[Paragraph(f"<b>{escape(k)}</b>", st["cell"]), Paragraph(escape(_fmt(v)), st["cell"])] for k, v in pairs]
    t = Table(rows, colWidths=[w1, _W - w1], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE1E7")),
        ("BACKGROUND", (0, 0), (0, -1), _LIGHT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _image(png: bytes, width: float = 16 * cm) -> Image:
    w, h = ImageReader(io.BytesIO(png)).getSize()
    return Image(io.BytesIO(png), width=width, height=width * h / w)


def _banner(text: str, color, st: dict) -> Table:
    t = Table([[Paragraph(text, st["body"])]], colWidths=[_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B8C0CC")),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def _params_text(d: dict[str, Any]) -> str:
    shown = {k: v for k, v in d.items() if v not in (None, {}, [])}
    return ", ".join(f"{k}={v}" for k, v in shown.items()) or "(defaults)"


def generate_report(
    path: str | Path,
    *,
    outcome: MatchingOutcome,
    balance: BalanceResult,
    cfg: MatchingConfig,
    prepared: PreparedData,
    figures: list[tuple[str, str, bytes]],
    hooks: list[HookResult] | None = None,
    run_info: dict[str, Any] | None = None,
    data_configs_json: list[dict[str, Any]] | None = None,
    max_table_rows: int = 40,
) -> Path:
    """Compila il PDF. ``figures``: lista (chiave, didascalia, png)."""
    target = Path(path)
    st = _styles()
    run_info = run_info or {}
    thr = cfg.validation.smd_threshold
    m = outcome.metrics
    s = balance.summary
    story: list[Any] = []

    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        # ---- header -------------------------------------------------------
        story.append(Paragraph("SAFE Matching Report", st["title"]))
        meta = [f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"safe-matching-framework v{__version__}"]
        if run_info.get("run_id"):
            meta.append(f"run {run_info['run_id']}")
        story.append(Paragraph(" | ".join(meta), st["sub"]))
        story.append(Spacer(1, 8))

        # ---- verdict ------------------------------------------------------
        n_unb = int(s.get("n_unbalanced_after", 0)) if s.get("n_variables") else None
        if n_unb is None:
            story.append(_banner("<b>No balance diagnostics available.</b>", _WARN, st))
        elif n_unb == 0:
            story.append(_banner(f"<b>All {s['n_variables']} evaluated variables are balanced after matching</b> "
                                 f"(|SMD| &lt; {thr:g}).", _OK, st))
        else:
            story.append(_banner(f"<b>{n_unb} of {s['n_variables']} variables remain above the |SMD| threshold of "
                                 f"{thr:g}</b> after matching (max |SMD| = {_fmt(s.get('max_abs_smd_after'))}).",
                                 _WARN if n_unb <= max(1, s['n_variables'] // 4) else _BAD, st))
        story.append(Spacer(1, 8))

        # ---- 1. summary ---------------------------------------------------
        story.append(Paragraph("1. Summary", st["h1"]))
        unit = "selected pool units" if outcome.output_type == "selection" else "matched pool rows"
        story.append(_kv([
            ("Method", f"{outcome.method}" + (f" (algorithm: {outcome.algorithm})" if outcome.method != outcome.algorithm else "")),
            ("Distance / feature space", outcome.metric_name),
            ("Direction", f"{outcome.query_label} (query) -> {outcome.pool_label} (pool)"),
            ("Eligible treated / control", f"{m['n_treated_eligible']} / {m['n_control_eligible']}"),
            ("Query units matched", f"{m['n_query_matched']} of {m['n_query_total']} ({_fmt(100 * m['match_rate'], 1)}%)"),
            (f"Result size ({unit})", f"{m['n_pool_rows']} ({m['n_unique_pool_units']} unique units; effective sample size {_fmt(m['ess_pool'], 1)})"),
            ("Mean |SMD| before -> after", f"{_fmt(s.get('mean_abs_smd_before'))} -> {_fmt(s.get('mean_abs_smd_after'))}"),
            ("Max |SMD| before -> after", f"{_fmt(s.get('max_abs_smd_before'))} -> {_fmt(s.get('max_abs_smd_after'))}"),
            (f"Variables with |SMD| < {thr:g}", f"{_fmt(s.get('pct_balanced_before'), 1)}% -> {_fmt(s.get('pct_balanced_after'), 1)}%"),
            ("Strata (skipped)", f"{m['n_strata']} ({m['n_strata_skipped']})"),
        ], st))

        # ---- 2. data ------------------------------------------------------
        story.append(Paragraph("2. Input data", st["h1"]))
        rows = [["Source", "Role", "Raw rows", "After preprocessing", "Dropped (strategy)", "Imputed values"]]
        for src in prepared.sources:
            rows.append([src["name"], src["role"], src["n_rows_raw"], src["n_rows_after_preprocessing"],
                         src["n_dropped_by_strategy"], src["imputed_values"]])
        story.append(_table(rows, [4.5 * cm, 2.2 * cm, 2.4 * cm, 3.2 * cm, 2.8 * cm, 2.4 * cm], st))
        story.append(Spacer(1, 4))
        story.append(_kv([
            ("Mode", "single dataset with group column" if prepared.mode == "single" else "two datasets"),
            ("Treated group", prepared.treated_label), ("Control group", prepared.control_label),
            ("Rows excluded (missing in matching variables)", m["n_excluded_missing"]),
            ("Rows trimmed (common support)", m["n_trimmed"]),
        ], st))

        # ---- 3. configuration --------------------------------------------
        story.append(Paragraph("3. Matching configuration", st["h1"]))
        v, c = cfg.variables, cfg.constraints
        cal = "; ".join(f"{w.variable} within {w.value:g}" + (" SD" if w.scale == "standard_deviation" else "") for w in c.calipers) or "none"
        gc = (f"{c.global_caliper.value:g}" + (" SD" if c.global_caliper.scale == "standard_deviation" else "")) if c.global_caliper else "none"
        story.append(_kv([
            ("Matching variables", ", ".join(v.matching)),
            ("Stratification variables", ", ".join(v.stratification) or "none"),
            ("Evaluation-only variables", ", ".join(v.evaluation) or "none"),
            ("Per-variable calipers", cal), ("Distance caliper (resolved)", "none" if outcome.metrics.get("caliper_distance") is None
                                              else f"{_fmt(outcome.metrics['caliper_distance'], 4)} (constraints.global_caliper: {gc}; PS caliper_sd if applicable)"),
            ("Metric", f"{cfg.metrics.name} ({_params_text(cfg.metrics.params)})"),
            ("Method", f"{cfg.method.name} ({_params_text(cfg.method.params)})"),
            ("Random state", cfg.options.random_state),
        ], st))

        # ---- 4. matching results -----------------------------------------
        story.append(Paragraph("4. Matching results", st["h1"]))
        dist_rows = [(k.replace("distance_", "").title(), m[k]) for k in ("distance_mean", "distance_median", "distance_p95", "distance_max") if k in m]
        if dist_rows:
            story.append(Paragraph("Pair distances", st["h2"]))
            story.append(_table([["Statistic", "Value"]] + [[a, _fmt(b, 4)] for a, b in dist_rows], [5 * cm, 4 * cm], st))
        sdf = outcome.strata
        if sdf.height:
            story.append(Paragraph("Strata", st["h2"]))
            cols = [c for c in ("stratum", "n_query", "n_pool", "n_query_matched", "n_pool_rows", "target", "status", "note") if c in sdf.columns]
            ws = {"stratum": 5.0, "n_query": 1.7, "n_pool": 1.7, "n_query_matched": 2.2, "n_pool_rows": 2.0, "target": 1.5, "status": 1.8, "note": 3.6}
            tot = sum(ws[c_] for c_ in cols)
            head = sdf.select(cols).head(max_table_rows)
            story.append(_table([cols] + [list(r) for r in head.iter_rows()], [ws[c_] / tot * _W for c_ in cols], st))
            if sdf.height > max_table_rows:
                story.append(Paragraph(f"(First {max_table_rows} of {sdf.height} strata; full table in strata.csv.)", st["small"]))
        if outcome.notes:
            story.append(Paragraph("Notes", st["h2"]))
            for n in outcome.notes:
                story.append(Paragraph(f"- {escape(n)}", st["body"]))

        # ---- 5. balance ---------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("5. Covariate balance", st["h1"]))
        tbl = balance.table
        if tbl.height:
            story.append(Paragraph(
                f"Standardized mean differences (SMD) use the pooled standard deviation; rows highlighted in red have "
                f"|SMD| &gt;= {thr:g} after matching. VR = variance ratio (treated/control); KS = Kolmogorov-Smirnov "
                f"statistic (numeric variables); the last column reports the Jensen-Shannon divergence and, when "
                f"weights are uniform, the chi-square p-value (categorical variables).", st["small"]))
            story.append(Spacer(1, 4))
            has = lambda n: n in tbl.columns  # noqa: E731
            head = ["Variable", "Role", "SMD before", "SMD after", "VR before", "VR after", "KS after", "JS / chi2 p"]
            rows, red = [head], {}
            for i, r in enumerate(tbl.head(max_table_rows * 2).iter_rows(named=True), start=1):
                cat = ""
                if r["type"] == "categorical":
                    cat = f"{_fmt(r.get('js_after'), 3)} / {_fmt(r.get('chi2_p_after'), 3)}" if has("js_after") else ""
                rows.append([r["variable"], r["role"], _fmt(r.get("smd_before")), _fmt(r.get("smd_after")),
                             _fmt(r.get("variance_ratio_before"), 2), _fmt(r.get("variance_ratio_after"), 2),
                             _fmt(r.get("ks_after")), cat])
                if r.get("balanced_after") is False:
                    red[i] = _BAD
            ws = [4.2, 2.0, 1.7, 1.7, 1.6, 1.6, 1.6, 3.1]
            story.append(_table(rows, [w / sum(ws) * _W for w in ws], st, row_colors=red))
            if tbl.height > max_table_rows * 2:
                story.append(Paragraph(f"(First {max_table_rows * 2} of {tbl.height} rows; full table in balance_table.csv.)", st["small"]))

        for key, caption, png in figures:
            story.append(Spacer(1, 8))
            story.append(KeepTogether([_image(png), Paragraph(escape(caption), st["cap"])]))

        # ---- 6. external hooks -------------------------------------------
        if hooks:
            story.append(PageBreak())
            story.append(Paragraph("6. External validation", st["h1"]))
            for h in hooks:
                story.append(Paragraph(escape(h.name), st["h2"]))
                if h.error:
                    story.append(_banner(f"Hook failed: {escape(h.error)}", _BAD, st))
                if h.metrics:
                    story.append(_kv([(k, v) for k, v in h.metrics.items()], st))
                if h.table is not None and h.table.height:
                    t = h.table.head(max_table_rows)
                    n = len(t.columns)
                    story.append(_table([t.columns] + [[_fmt(x, 4) for x in r] for r in t.iter_rows()], [_W / n] * n, st))

        # ---- appendix -----------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("Appendix A. Warnings from data preparation", st["h1"]))
        if prepared.warnings:
            for w in prepared.warnings:
                story.append(Paragraph(f"- {escape(w)}", st["small"]))
        else:
            story.append(Paragraph("None.", st["body"]))
        story.append(Paragraph("Appendix B. Matching configuration (JSON)", st["h1"]))
        story.append(Preformatted(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), st["code"]))
        if data_configs_json:
            for i, dc in enumerate(data_configs_json, 1):
                active = {k: {"new_name": v["new_name"], "categorical": v["categorical"], "numerical": v["numerical"]}
                          for k, v in dc["columns"].items() if v["active"]}
                story.append(Paragraph(f"Appendix C.{i}. Active columns of data config {i}", st["h1"]))
                story.append(Preformatted(json.dumps(active, indent=1, ensure_ascii=False), st["code"]))

        def _footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(_GREY)
            canvas.drawString(1.75 * cm, 1.1 * cm, f"safe-matching-framework v{__version__}")
            canvas.drawRightString(A4[0] - 1.75 * cm, 1.1 * cm, f"Page {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(str(target), pagesize=A4, title="SAFE Matching Report",
                                leftMargin=1.75 * cm, rightMargin=1.75 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm)
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return target
    except Exception as exc:  # noqa: BLE001
        raise ReportError(f"Generazione del PDF fallita ({target}): {exc}") from exc
