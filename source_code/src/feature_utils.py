from __future__ import annotations

import numpy as np
import pandas as pd

from src.real_policies import normalize_real_policy, real_policy_labels


def safe_filename(value: str, fallback: str = "feature", max_length: int = 120) -> str:
    """Make a value safe to use as a filename."""
    chars = []
    for char in value:
        if char.isalnum() or char in {"_", "-", "."}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_")[:max_length] or fallback


def feature_to_variable(feature: str) -> str:
    """Map an encoded feature name back to the original variable."""
    if feature.endswith("__missing"):
        return feature[: -len("__missing")]
    if "__" in feature:
        return feature.split("__", 1)[0]
    return feature


def format_policy_label(policy: str) -> str:
    """Return a readable real-cohort policy label."""
    canonical_policy = normalize_real_policy(policy)
    return real_policy_labels().get(canonical_policy, canonical_policy.replace("_", " "))


def aggregate_encoded_importance(
    feature_importance: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
    count_col: str | None = None,
    sort_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate encoded-feature importances to original variables.

    Optional grouping columns are preserved, and encoded-feature counts can be
    added for auditability.
    """
    group_cols = list(group_cols or [])
    output_cols = [*group_cols, "variable", value_col]
    if count_col is not None:
        output_cols.append(count_col)
    if feature_importance.empty:
        return pd.DataFrame(columns=output_cols)

    df = feature_importance.copy()
    df["variable"] = df["feature"].map(feature_to_variable)
    agg_spec = {value_col: (value_col, "sum")}
    if count_col is not None:
        agg_spec[count_col] = ("feature", "count")

    grouped = df.groupby([*group_cols, "variable"], as_index=False).agg(**agg_spec)
    if sort_cols is None:
        sort_cols = [value_col]
        ascending = [False]
    else:
        ascending = [True] * (len(sort_cols) - 1) + [False]
    return grouped.sort_values(sort_cols, ascending=ascending)


def add_rank_and_pct(
    df: pd.DataFrame,
    value_col: str,
    pct_col: str,
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Add one-based rank and percentage columns based on an importance column.

    Rows are sorted by descending importance before ranks are assigned.
    """
    ranked = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    ranked.insert(0, rank_col, np.arange(1, len(ranked) + 1))
    total = float(ranked[value_col].sum())
    ranked[pct_col] = 100 * ranked[value_col] / total if total > 0 else np.nan
    return ranked
