"""Esempio di hook di validazione esterna. Da collegare in matching.json:

    "validation": {"hooks": [{"name": "esempio", "callable": "safe_matching.integrations.example_hook:smd_summary"}]}

Una hook reale per safe-validation-framework avra' la stessa firma e chiamera' il suo evaluator.
"""

from __future__ import annotations

import polars as pl


def smd_summary(*, before: pl.DataFrame, after: pl.DataFrame, group_col: str, variables: list[str], kinds: dict, **_) -> dict:
    return {"n_rows_before": before.height, "n_rows_after": after.height, "n_variables": len(variables)}
