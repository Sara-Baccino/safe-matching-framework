"""Valutazione della riuscita del matching rispetto a criteri configurabili."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from safe_matching.schemas.matching_config import SuccessCriteria

__all__ = ["evaluate_success", "SuccessResult"]


@dataclass
class SuccessResult:
    checks: list[dict[str, Any]]
    passed: bool
    n_passed: int
    n_checks: int


def _nan(x: Any) -> bool:
    return x is None or x != x


def evaluate_success(matching: dict[str, Any], balance: dict[str, Any], crit: SuccessCriteria, smd_threshold: float,
                     analysis: dict[str, Any] | None = None) -> SuccessResult:
    checks: list[dict[str, Any]] = []

    def add(name: str, observed: Any, op: str, threshold: float | None, ok: bool | None):
        if threshold is None:
            return
        checks.append({"criterion": name, "observed": observed, "rule": f"{op} {threshold:g}",
                       "passed": bool(ok) if ok is not None else False})

    mr = matching.get("match_rate")
    add("Match rate (query units matched)", mr, ">=", crit.min_match_rate, None if _nan(mr) else mr >= (crit.min_match_rate or 0))
    if balance.get("n_variables"):
        ma, mx, pb = balance["mean_abs_smd_after"], balance["max_abs_smd_after"], balance["pct_balanced_after"]
        add("Mean |SMD| after matching", ma, "<=", crit.max_mean_abs_smd, None if _nan(ma) else ma <= (crit.max_mean_abs_smd or 0))
        add("Max |SMD| after matching", mx, "<=", crit.max_abs_smd, None if _nan(mx) else mx <= (crit.max_abs_smd or 0))
        add(f"% variables with |SMD| < {smd_threshold:g}", pb, ">=", crit.min_pct_balanced, None if _nan(pb) else pb >= (crit.min_pct_balanced or 0))
        if crit.require_improvement:
            mb = balance["mean_abs_smd_before"]
            checks.append({"criterion": "Mean |SMD| improves over unmatched", "observed": f"{mb:.3f} -> {ma:.3f}",
                           "rule": "after < before", "passed": bool(ma < mb)})
    analysis = analysis or {}
    if crit.max_pca_smd is not None:
        v = analysis.get("pca_max_abs_smd_after")
        add("Max |SMD| on principal components", v, "<=", crit.max_pca_smd, None if _nan(v) else v <= crit.max_pca_smd)
    if crit.max_corr_diff is not None:
        v = analysis.get("corr_mean_abs_diff_after")
        add("Mean |corr(treated) - corr(control)|", v, "<=", crit.max_corr_diff, None if _nan(v) else v <= crit.max_corr_diff)
    passed = all(c["passed"] for c in checks) if checks else False
    return SuccessResult(checks, passed, sum(c["passed"] for c in checks), len(checks))
