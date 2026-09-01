"""Minimum cleaning required before computing SOFA."""

from __future__ import annotations

import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    DATE_COLUMNS,
    SOFA_CRITICAL_CARE_RETURN_MIN_HOURS,
    SOFA_DATASETS_DIR,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    SOFA_PRE_DIR,
)
from .figure_style import PALETTE, apply_report_style, save_report_figure


SOFA_FALLBACK_NORMAL = {
    "FIO2": 21,
    "GLASGOW": 15,
    "bilirubina_total": 0.8,
}

SOFA_FORWARD_FILL_COLUMNS = [
    "O2SAT",
    "plaquetes",
    "bilirubina_total",
    "creatinina",
    "TAM",
]

LAB_FORWARD_FILL_COLUMNS = {"plaquetes", "bilirubina_total", "creatinina"}

PRE_SOFA_PHYSIOLOGIC_RANGES = {
    # Broad ranges: only invalidate physiologically impossible values or values
    # very likely caused by unit/load errors, not plausible clinical extremes.
    "edat": (0, 110),
    "PAS": (30, 300),
    "PAD": (10, 200),
    "TAM": (20, 250),
    "FC": (20, 260),
    "RESP": (4, 90),
    "O2SAT": (1, 100),
    "TEMP": (25, 45),
    "GLASGOW": (3, 15),
    "FIO2": (21, 100),
    "pao2_arterial": (20, 700),
    "paco2_arterial": (5, 250),
    "ph_arterial": (6.5, 8.0),
    "bicarbonat_arterial": (1, 80),
    "lactat_arterial": (0, 50),
    "lactat_venos": (0, 50),
    "creatinina": (0, 25),
    "bilirubina_total": (0, 50),
    "plaquetes": (1, 2000),
    "leucocits": (0, 500),
    "DIURESIS": (0, 10000),
    "temps_cirurgia": (0, 168),
}

PRE_SOFA_BINARY_COLUMNS = [
    "vasopressor_qualsevol",
    "vasopressor_multiple",
    "vasopressor_dobutamina",
    "vasopressor_dopamina",
    "vasopressor_noradrenalina",
    "vasopressor_adrenalina",
    "porta_o2",
]


def clean_data_for_sofa(
    df: pd.DataFrame,
    episode_missingness_threshold: float = 0.80,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
    episode_gap_exclusion_threshold_days: int | None = SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    max_allowed_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply the minimum cleaning required before SOFA computation.

    This function does not select predictive variables. It only applies quality
    controls, estimates whether SOFA components are calculable, and excludes
    episodes that are too incomplete. Strong model decisions are left for train.
    """
    _validate_required_columns(df)

    df_clean, pre_sofa_audit = _apply_pre_cleaning_quality_checks(df, max_allowed_date)

    df_clean["data_index"] = pd.to_datetime(df_clean["data_index"], errors="coerce")
    df_clean = df_clean.sort_values(["Episodi", "data_index"]).reset_index(drop=True)
    _add_time_gaps(df_clean)

    availability_df = df_clean.copy()
    _prepare_sofa_availability(
        availability_df,
        lab_ffill_limit_days,
        vitals_ffill_limit_days,
    )
    availability = _calculate_sofa_component_availability(availability_df)
    df_clean = pd.concat([df_clean, availability], axis=1)

    episode_summary = _summarize_missingness_by_episode(df_clean)
    gap_summary = _summarize_gaps_by_episode(
        df_clean,
        episode_gap_exclusion_threshold_days,
    )
    episode_summary = episode_summary.merge(gap_summary, on="Episodi", how="left")
    _mark_episode_exclusions(
        episode_summary,
        episode_missingness_threshold,
        episode_gap_exclusion_threshold_days,
    )
    excluded_episodes = episode_summary.loc[
        episode_summary["exclude_episode"] == 1,
        "Episodi",
    ]

    df_clean = df_clean.loc[~df_clean["Episodi"].isin(excluded_episodes)].copy()
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, episode_summary.copy(), pre_sofa_audit


def save_sofa_cleaning_outputs(
    df_clean: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    pre_sofa_audit: dict[str, pd.DataFrame] | None = None,
    episode_gap_exclusion_threshold_days: int | None = SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
) -> None:
    """Save the clean pre-SOFA dataset and excluded-episode summaries."""
    SOFA_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SOFA_PRE_DIR.mkdir(parents=True, exist_ok=True)
    _clear_sofa_pre_outputs()

    global_summary = {
        "n_rows_after_cleaning": int(len(df_clean)),
        "n_episodes_after_cleaning": int(df_clean["Episodi"].nunique()),
        "n_excluded_episodes": int((exclusion_summary["exclude_episode"] == 1).sum()),
        "n_episodes_excluded_for_missing_components": int(
            (exclusion_summary["exclude_for_missing_components"] == 1).sum()
        ) if "exclude_for_missing_components" in exclusion_summary.columns else 0,
        "n_episodes_excluded_for_temporal_gap": int(
            (exclusion_summary["exclude_for_temporal_gap"] == 1).sum()
        ) if "exclude_for_temporal_gap" in exclusion_summary.columns else 0,
        "pct_excluded_episodes": round(
            100 * (exclusion_summary["exclude_episode"] == 1).mean(), 2
        ) if len(exclusion_summary) else 0.0,
        "episode_gap_exclusion_threshold_days": episode_gap_exclusion_threshold_days,
        "temporal_gap_criterion": "large_gap_without_critical_care_record",
        "temporal_gap_exclusion_enabled": episode_gap_exclusion_threshold_days is not None,
        "lab_ffill_limit_days": lab_ffill_limit_days,
        "vitals_ffill_limit_days": vitals_ffill_limit_days,
        "critical_care_return_min_hours": SOFA_CRITICAL_CARE_RETURN_MIN_HOURS,
    }
    if pre_sofa_audit and "summary" in pre_sofa_audit:
        global_summary["pre_sofa"] = _audit_summary_to_dict(pre_sofa_audit["summary"])

    with open(SOFA_PRE_DIR / "sofa_cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    _save_excel_csv(exclusion_summary, SOFA_PRE_DIR / "excluded_sofa_episodes.csv")
    _save_temporal_gap_decision_outputs(
        exclusion_summary,
        episode_gap_exclusion_threshold_days,
    )
    df_clean.to_csv(SOFA_DATASETS_DIR / "daily_sepsis_model_clean_sofa.csv", index=False)

    if pre_sofa_audit:
        for name, table in pre_sofa_audit.items():
            _save_excel_csv(table, SOFA_PRE_DIR / f"pre_sofa_audit_{name}.csv")


def _mark_episode_exclusions(
    episode_summary: pd.DataFrame,
    episode_missingness_threshold: float,
    episode_gap_exclusion_threshold_days: int | None,
) -> None:
    """Mark episodes without enough information to compute SOFA.

    A temporal gap only excludes an episode when it is large and unexplained by
    any critical-care record in the episode.
    """
    episode_summary["exclude_for_missing_components"] = (
        episode_summary["pct_missing_sofa_components"] >= episode_missingness_threshold * 100
    ).astype(int)

    if episode_gap_exclusion_threshold_days is None:
        episode_summary["large_temporal_gap"] = 0
        episode_summary["exclude_for_temporal_gap"] = 0
    else:
        large_gap = episode_summary["max_day_gap"] > episode_gap_exclusion_threshold_days
        has_critical_care_record = (
            episode_summary["has_critical_care_record"].fillna(False).astype(bool)
            if "has_critical_care_record" in episode_summary.columns
            else pd.Series(False, index=episode_summary.index)
        )
        episode_summary["large_temporal_gap"] = large_gap.astype(int)
        episode_summary["exclude_for_temporal_gap"] = (
            large_gap & ~has_critical_care_record
        ).astype(int)

    episode_summary["exclude_episode"] = (
        (episode_summary["exclude_for_missing_components"] == 1)
        | (episode_summary["exclude_for_temporal_gap"] == 1)
    ).astype(int)


def _apply_pre_cleaning_quality_checks(
    df: pd.DataFrame,
    max_allowed_date: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply conservative controls before SOFA calculation and return audit tables."""
    df_clean = df.copy()
    max_date = _normalize_max_date(max_allowed_date)
    original_row_count = len(df_clean)

    date_audit = _convert_dates_and_audit(df_clean, max_date)

    invalid_data_index = df_clean["data_index"].isna()
    future_data_index = df_clean["data_index"].gt(max_date)
    invalid_data_index_count = int(invalid_data_index.sum())
    future_data_index_count = int(future_data_index.sum())
    df_clean = df_clean.loc[~invalid_data_index & ~future_data_index].copy()

    exact_duplicates = df_clean.duplicated(keep="first")
    exact_duplicate_count = int(exact_duplicates.sum())
    if exact_duplicate_count:
        df_clean = df_clean.loc[~exact_duplicates].copy()

    variable_audit = _invalidate_impossible_values(df_clean)
    binary_audit = _normalize_binary_flags_and_audit(df_clean)
    invalid_numeric_count = (
        int(variable_audit["n_invalid_values_total"].sum())
        if "n_invalid_values_total" in variable_audit.columns
        else 0
    )
    invalid_binary_count = (
        int(binary_audit["n_invalid_values"].sum())
        if "n_invalid_values" in binary_audit.columns
        else 0
    )

    summary = pd.DataFrame(
        [
            {"metric": "max_allowed_date", "value": max_date.date().isoformat()},
            {"metric": "n_original_rows", "value": original_row_count},
            {"metric": "n_rows_removed_invalid_data_index", "value": invalid_data_index_count},
            {"metric": "n_rows_removed_future_data_index", "value": future_data_index_count},
            {"metric": "n_exact_duplicate_rows_removed", "value": exact_duplicate_count},
            {"metric": "n_numeric_values_set_to_nan", "value": invalid_numeric_count},
            {"metric": "n_binary_values_set_to_nan", "value": invalid_binary_count},
            {"metric": "n_rows_before_sofa_episode_exclusions", "value": len(df_clean)},
        ]
    )

    return df_clean.reset_index(drop=True), {
        "summary": summary,
        "dates": date_audit,
        "variables": variable_audit,
        "binaries": binary_audit,
    }


def _normalize_max_date(max_allowed_date: str | pd.Timestamp | None) -> pd.Timestamp:
    """Define the maximum allowed date reproducibly when it is provided."""
    if max_allowed_date is None:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(max_allowed_date, errors="raise").normalize()


def _convert_dates_and_audit(
    df: pd.DataFrame,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Convert known date columns and summarize invalid/future values."""
    rows = []
    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue

        original_no_null = df[col].notna()
        converted = pd.to_datetime(df[col], errors="coerce")
        invalid = original_no_null & converted.isna()
        future = converted.gt(max_date)
        df[col] = converted

        rows.append(
            {
                "variable": col,
                "n_original_non_null": int(original_no_null.sum()),
                "n_invalid_conversions": int(invalid.sum()),
                "n_future_dates": int(future.sum()),
                "min": converted.min(),
                "max": converted.max(),
            }
        )

    return pd.DataFrame(rows)


def _invalidate_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set values outside broad physiologic ranges to NaN and audit the action."""
    rows = []
    for col, (allowed_min, allowed_max) in PRE_SOFA_PHYSIOLOGIC_RANGES.items():
        if col not in df.columns:
            continue

        original_no_null = df[col].notna()
        numeric = pd.to_numeric(df[col], errors="coerce")

        if col == "FIO2":
            fraction = numeric.gt(0) & numeric.le(1)
            numeric.loc[fraction] = numeric.loc[fraction] * 100
            converted_fraction_count = int(fraction.sum())
        else:
            converted_fraction_count = 0

        no_numeric = original_no_null & numeric.isna()
        out_of_range = numeric.notna() & (
            numeric.lt(allowed_min) | numeric.gt(allowed_max)
        )
        invalid = no_numeric | out_of_range
        df[col] = numeric
        df.loc[invalid, col] = pd.NA

        observed_count = int(original_no_null.sum())
        invalid_count = int(invalid.sum())
        rows.append(
            {
                "variable": col,
                "allowed_min": allowed_min,
                "allowed_max": allowed_max,
                "n_observed_before": observed_count,
                "n_fio2_fraction_converted_to_percent": converted_fraction_count,
                "n_no_numeric": int(no_numeric.sum()),
                "n_out_of_range": int(out_of_range.sum()),
                "n_invalid_values_total": invalid_count,
                "pct_invalid_among_observed": round(
                    100 * invalid_count / observed_count,
                    3,
                ) if observed_count else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _normalize_binary_flags_and_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pre-SOFA binary flags are 0/1/NaN."""
    rows = []
    for col in PRE_SOFA_BINARY_COLUMNS:
        if col not in df.columns:
            continue

        original_no_null = df[col].notna()
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = original_no_null & ~numeric.isin([0, 1])
        df[col] = numeric
        df.loc[invalid, col] = pd.NA

        observed_count = int(original_no_null.sum())
        invalid_count = int(invalid.sum())
        rows.append(
            {
                "variable": col,
                "n_observed_before": observed_count,
                "n_invalid_values": invalid_count,
                "pct_invalid_among_observed": round(
                    100 * invalid_count / observed_count,
                    3,
                ) if observed_count else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _audit_summary_to_dict(summary: pd.DataFrame) -> dict[str, str | int | float]:
    """Convert the pre-SOFA summary table into a dictionary for JSON output."""
    if summary.empty or not {"metric", "value"}.issubset(summary.columns):
        return {}
    result = {}
    for metric, value in zip(summary["metric"], summary["value"]):
        if hasattr(value, "item"):
            value = value.item()
        result[str(metric)] = value
    return result


def _save_excel_csv(df: pd.DataFrame, path, index: bool = False) -> None:
    """Save a CSV compatible with European Excel defaults."""
    df.to_csv(path, index=index, sep=";", decimal=",", encoding="utf-8-sig")


def _clear_sofa_pre_outputs() -> None:
    """Remove stale generated pre-SOFA outputs before rewriting them."""
    for pattern in ("*.csv", "*.json", "*.txt"):
        for path in SOFA_PRE_DIR.glob(pattern):
            path.unlink()
    figures_dir = SOFA_PRE_DIR / "figures"
    if figures_dir.exists():
        for path in figures_dir.glob("*.png"):
            path.unlink()


def _save_temporal_gap_decision_outputs(
    exclusion_summary: pd.DataFrame,
    episode_gap_exclusion_threshold_days: int | None,
) -> None:
    """Save tables and figures that justify the temporal-gap exclusion rule."""
    if exclusion_summary.empty or "max_day_gap" not in exclusion_summary.columns:
        return

    figures_dir = SOFA_PRE_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    decision_table = _temporal_gap_decision_table(
        exclusion_summary,
        episode_gap_exclusion_threshold_days,
    )
    _save_excel_csv(decision_table, SOFA_PRE_DIR / "temporal_gap_decision_summary.csv")
    _write_temporal_gap_decision_report(
        decision_table,
        episode_gap_exclusion_threshold_days,
        SOFA_PRE_DIR / "temporal_gap_decision_rationale.txt",
    )
    _plot_temporal_gap_distribution(
        exclusion_summary,
        episode_gap_exclusion_threshold_days,
        figures_dir / "temporal_gap_distribution.png",
    )
    _plot_temporal_gap_decision_counts(
        decision_table,
        figures_dir / "temporal_gap_decision_counts.png",
    )
    sensitivity_table = _temporal_gap_threshold_sensitivity_table(exclusion_summary)
    _save_excel_csv(
        sensitivity_table,
        SOFA_PRE_DIR / "temporal_gap_threshold_sensitivity.csv",
    )
    _plot_temporal_gap_threshold_sensitivity(
        sensitivity_table,
        episode_gap_exclusion_threshold_days,
        figures_dir / "temporal_gap_threshold_sensitivity.png",
    )


def _temporal_gap_decision_table(
    exclusion_summary: pd.DataFrame,
    episode_gap_exclusion_threshold_days: int | None,
) -> pd.DataFrame:
    """Count episodes by temporal-gap decision group."""
    data = exclusion_summary.copy()
    threshold = episode_gap_exclusion_threshold_days
    data["max_day_gap"] = pd.to_numeric(data["max_day_gap"], errors="coerce").fillna(0)

    if threshold is None:
        large_gap = pd.Series(False, index=data.index)
    else:
        large_gap = data["max_day_gap"] > threshold

    has_critical_record = (
        data["has_critical_care_record"].fillna(False).astype(bool)
        if "has_critical_care_record" in data.columns
        else pd.Series(False, index=data.index)
    )

    data["temporal_gap_decision_group"] = "No large temporal gap"
    data.loc[large_gap & has_critical_record, "temporal_gap_decision_group"] = (
        "Large gap with critical-care record"
    )
    data.loc[large_gap & ~has_critical_record, "temporal_gap_decision_group"] = (
        "Large unexplained gap"
    )

    order = [
        "No large temporal gap",
        "Large gap with critical-care record",
        "Large unexplained gap",
    ]
    rows = []
    total_episodes = len(data)
    for group in order:
        group_data = data.loc[data["temporal_gap_decision_group"] == group]
        if "exclude_for_temporal_gap" in group_data.columns:
            excluded_by_gap = int((group_data["exclude_for_temporal_gap"] == 1).sum())
        else:
            excluded_by_gap = 0
        rows.append(
            {
                "decision_group": group,
                "n_episodes": int(len(group_data)),
                "pct_episodes": round(100 * len(group_data) / total_episodes, 2)
                if total_episodes
                else 0.0,
                "median_max_gap_days": round(float(group_data["max_day_gap"].median()), 2)
                if len(group_data)
                else 0.0,
                "max_gap_days": round(float(group_data["max_day_gap"].max()), 2)
                if len(group_data)
                else 0.0,
                "excluded_by_temporal_gap": excluded_by_gap,
            }
        )

    return pd.DataFrame(rows)


def _temporal_gap_threshold_sensitivity_table(
    exclusion_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Show how many episodes would be excluded with alternative gap thresholds."""
    data = exclusion_summary.copy()
    data["max_day_gap"] = pd.to_numeric(data["max_day_gap"], errors="coerce").fillna(0)
    has_critical_record = (
        data["has_critical_care_record"].fillna(False).astype(bool)
        if "has_critical_care_record" in data.columns
        else pd.Series(False, index=data.index)
    )

    rows = []
    total_episodes = len(data)
    for threshold in [7, 14, 30, 60, 90]:
        large_gap = data["max_day_gap"] > threshold
        unexplained_gap = large_gap & ~has_critical_record
        rows.append(
            {
                "threshold_days": threshold,
                "episodes_with_large_gap": int(large_gap.sum()),
                "episodes_with_large_gap_pct": round(
                    100 * large_gap.sum() / total_episodes,
                    2,
                )
                if total_episodes
                else 0.0,
                "episodes_excluded_if_used": int(unexplained_gap.sum()),
                "episodes_excluded_if_used_pct": round(
                    100 * unexplained_gap.sum() / total_episodes,
                    2,
                )
                if total_episodes
                else 0.0,
                "episodes_kept_due_to_critical_record": int(
                    (large_gap & has_critical_record).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def _write_temporal_gap_decision_report(
    decision_table: pd.DataFrame,
    episode_gap_exclusion_threshold_days: int | None,
    path,
) -> None:
    """Write a short plain-English rationale for the temporal-gap criterion."""
    if episode_gap_exclusion_threshold_days is None:
        threshold_text = "disabled"
    else:
        threshold_text = f">{episode_gap_exclusion_threshold_days} days"

    table_lines = []
    for row in decision_table.to_dict("records"):
        table_lines.append(
            (
                "- {decision_group}: {n_episodes} episodes ({pct_episodes:.2f}%), "
                "median max gap {median_max_gap_days:.2f} days, "
                "maximum {max_gap_days:.2f} days, excluded {excluded_by_temporal_gap}."
            ).format(**row)
        )

    text = "\n".join(
        [
            "Temporal-gap exclusion rationale",
            "",
            f"Rule used: exclude episodes with maximum temporal gap {threshold_text} "
            "only when the episode has no evidence of a critical-care stay.",
            "",
            "Reasoning: a very large gap without a critical-care record is treated as "
            "an unexplained interruption of the daily episode timeline. A large gap "
            "with a critical-care record is kept because the missing days may be "
            "clinically explained by the critical-care stay and should not be removed "
            "automatically.",
            "",
            "Decision summary:",
            *table_lines,
            "",
            "Generated files:",
            "- temporal_gap_decision_summary.csv",
            "- temporal_gap_threshold_sensitivity.csv",
            "- figures/temporal_gap_distribution.png",
            "- figures/temporal_gap_decision_counts.png",
            "- figures/temporal_gap_threshold_sensitivity.png",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _plot_temporal_gap_distribution(
    exclusion_summary: pd.DataFrame,
    episode_gap_exclusion_threshold_days: int | None,
    path,
) -> None:
    """Plot the distribution of maximum temporal gaps per episode."""
    data = exclusion_summary.copy()
    data["max_day_gap"] = pd.to_numeric(data["max_day_gap"], errors="coerce").fillna(0)
    if data.empty:
        return

    has_critical_record = (
        data["has_critical_care_record"].fillna(False).astype(bool)
        if "has_critical_care_record" in data.columns
        else pd.Series(False, index=data.index)
    )
    data["critical_group"] = has_critical_record.map(
        {
            True: "With critical-care record",
            False: "Without critical-care record",
        }
    )

    plot_data = data.loc[data["max_day_gap"] > 0].copy()
    if plot_data.empty:
        plot_data = data.copy()

    cap = max(30, float(plot_data["max_day_gap"].quantile(0.99)))
    plot_data["max_gap_days_capped"] = plot_data["max_day_gap"].clip(upper=cap)

    _set_sofa_figure_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [
        ("Without critical-care record", PALETTE["rose"]),
        ("With critical-care record", PALETTE["blue"]),
    ]:
        values = plot_data.loc[
            plot_data["critical_group"] == label,
            "max_gap_days_capped",
        ]
        if values.empty:
            continue
        ax.hist(
            values,
            bins=30,
            alpha=0.65,
            label=label,
            color=color,
            edgecolor="white",
        )

    if episode_gap_exclusion_threshold_days is not None:
        ax.axvline(
            episode_gap_exclusion_threshold_days,
            color=PALETTE["ink"],
            linestyle="--",
            linewidth=1.5,
            label=f"Exclusion threshold ({episode_gap_exclusion_threshold_days} days)",
        )

    ax.set_title("Maximum temporal gap per episode")
    ax.set_xlabel("Maximum gap between consecutive daily records (days)")
    ax.set_ylabel("Episodes")
    ax.legend(frameon=False)
    save_report_figure(fig, path)


def _plot_temporal_gap_decision_counts(decision_table: pd.DataFrame, path) -> None:
    """Plot episode counts by temporal-gap decision group."""
    if decision_table.empty:
        return

    labels = decision_table["decision_group"].tolist()
    counts = decision_table["n_episodes"].astype(int).tolist()
    colors = [PALETTE["green"], PALETTE["blue"], PALETTE["rose"]]

    _set_sofa_figure_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels, counts, color=colors[: len(labels)])
    ax.set_title("Temporal-gap decision groups")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("")
    ax.invert_yaxis()

    max_count = max(counts) if counts else 0
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(max_count * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va="center",
            fontsize=9,
        )

    save_report_figure(fig, path)


def _plot_temporal_gap_threshold_sensitivity(
    sensitivity_table: pd.DataFrame,
    selected_threshold_days: int | None,
    path,
) -> None:
    """Plot how the exclusion count changes across candidate thresholds."""
    if sensitivity_table.empty:
        return

    _set_sofa_figure_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = sensitivity_table["threshold_days"]
    excluded = sensitivity_table["episodes_excluded_if_used"]
    large_gap = sensitivity_table["episodes_with_large_gap"]

    ax.plot(
        x,
        large_gap,
        marker="o",
        linewidth=2,
        color=PALETTE["blue"],
        label="Episodes with large gap",
    )
    ax.plot(
        x,
        excluded,
        marker="o",
        linewidth=2,
        color=PALETTE["rose"],
        label="Excluded if no critical-care record",
    )

    if selected_threshold_days is not None:
        ax.axvline(
            selected_threshold_days,
            color=PALETTE["ink"],
            linestyle="--",
            linewidth=1.5,
            label=f"Selected threshold ({selected_threshold_days} days)",
        )

    ax.set_title("Sensitivity of temporal-gap exclusion threshold")
    ax.set_xlabel("Candidate threshold (days)")
    ax.set_ylabel("Episodes")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    save_report_figure(fig, path)


def _set_sofa_figure_style() -> None:
    """Apply a small consistent style for SOFA cleaning figures."""
    apply_report_style()


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate the minimum columns required to order episodes and days."""
    required_columns = {"Episodi", "data_index"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns required for SOFA cleaning: {', '.join(missing)}"
        )


def _prepare_sofa_availability(
    df: pd.DataFrame,
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> None:
    """Prepare an operational copy to estimate whether each SOFA component is calculable."""
    if "data_index" in df.columns:
        df["data_index"] = pd.to_datetime(df["data_index"], errors="coerce")

    for col, normal_value in SOFA_FALLBACK_NORMAL.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(normal_value)

    present_columns = [col for col in SOFA_FORWARD_FILL_COLUMNS if col in df.columns]
    if not present_columns:
        return

    critical_care_return = _critical_care_return_row(df)
    for col in present_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        pre_return_col = _critical_care_return_lab_column(df, col)
        if col in LAB_FORWARD_FILL_COLUMNS and pre_return_col in df.columns:
            pre_return = pd.to_numeric(df[pre_return_col], errors="coerce")
            use_pre_return = df[col].isna() & pre_return.notna() & critical_care_return
            df[col] = df[col].where(~use_pre_return, pre_return)
        original_series = df[col].copy()
        df[col] = df.groupby("Episodi")[col].ffill()
        ffill_limit_days = (
            lab_ffill_limit_days
            if col in LAB_FORWARD_FILL_COLUMNS
            else vitals_ffill_limit_days
        )

        if ffill_limit_days is not None:
            days_since_last = _days_since_last_observation(
                original_series,
                df["data_index"],
                df["Episodi"],
            )
            df.loc[days_since_last > ffill_limit_days, col] = pd.NA


def _critical_care_return_lab_column(df: pd.DataFrame, col: str) -> str:
    """Return the original CSV column for pre-return laboratory values."""
    return f"{col}_pre_retorn_critics_3d"


def _add_time_gaps(df: pd.DataFrame) -> None:
    """Add the distance in days from the previous row of the episode."""
    df["previous_day_gap"] = (
        df.groupby("Episodi")["data_index"].diff().dt.total_seconds() / 86400
    )


def _critical_care_return_row(df: pd.DataFrame) -> pd.Series:
    """Identify the day row when a patient leaves prolonged critical care."""
    if {"data_hora_alta_critics", "data_index"}.issubset(df.columns):
        critical_care_discharge = pd.to_datetime(df["data_hora_alta_critics"], errors="coerce")
        data_index = pd.to_datetime(df["data_index"], errors="coerce")
        return_row = (
            critical_care_discharge.notna()
            & data_index.notna()
            & critical_care_discharge.ge(data_index)
            & critical_care_discharge.lt(data_index + pd.Timedelta(days=1))
        )
        if "temps_critics" in df.columns:
            critical_care_hours = pd.to_numeric(df["temps_critics"], errors="coerce")
            if "temps_cirurgia" in df.columns:
                surgery_hours = pd.to_numeric(df["temps_cirurgia"], errors="coerce").fillna(0)
                critical_care_hours = critical_care_hours - surgery_hours
            return_row = return_row & critical_care_hours.gt(
                SOFA_CRITICAL_CARE_RETURN_MIN_HOURS
            ).fillna(False)
        return return_row

    return pd.Series(True, index=df.index)


def _critical_care_record_by_episode(df: pd.DataFrame) -> pd.Series:
    """Return True for episodes with any evidence of critical-care stay."""
    row_signals = []

    for col in ("en_critics_dia", "temps_critics_dia", "temps_critics"):
        if col in df.columns:
            row_signals.append(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0))

    if "data_hora_alta_critics" in df.columns:
        critical_care_discharge = pd.to_datetime(
            df["data_hora_alta_critics"],
            errors="coerce",
        )
        row_signals.append(critical_care_discharge.notna())

    if not row_signals:
        critical_care_record = pd.Series(False, index=df.index)
    else:
        critical_care_record = row_signals[0]
        for signal in row_signals[1:]:
            critical_care_record = critical_care_record | signal

    return critical_care_record.groupby(df["Episodi"]).any()


def _summarize_gaps_by_episode(
    df: pd.DataFrame,
    episode_gap_exclusion_threshold_days: int | None,
) -> pd.DataFrame:
    """Summarize temporal gaps and whether they are explained by critical care."""
    if "previous_day_gap" not in df.columns:
        _add_time_gaps(df)

    summary = (
        df.groupby("Episodi", as_index=False)
        .agg(
            max_day_gap=("previous_day_gap", "max"),
            n_gaps_over_7d=("previous_day_gap", lambda s: int((s > 7).sum())),
            n_gaps_over_30d=("previous_day_gap", lambda s: int((s > 30).sum())),
        )
    )
    summary["max_day_gap"] = summary["max_day_gap"].fillna(0).round(2)
    critical_care_record = _critical_care_record_by_episode(df)
    summary["has_critical_care_record"] = (
        summary["Episodi"].map(critical_care_record).fillna(False).astype(bool)
    )

    if episode_gap_exclusion_threshold_days is not None:
        threshold_col = f"n_gaps_over_{episode_gap_exclusion_threshold_days}d"
        summary[threshold_col] = (
            df.groupby("Episodi")["previous_day_gap"]
            .apply(lambda s: int((s > episode_gap_exclusion_threshold_days).sum()))
            .reset_index(drop=True)
        )
        summary["n_unexplained_gaps_over_threshold"] = (
            summary[threshold_col].gt(0) & ~summary["has_critical_care_record"]
        ).astype(int)
    else:
        summary["n_unexplained_gaps_over_threshold"] = 0

    return summary


def _days_since_last_observation(
    series: pd.Series,
    dates: pd.Series,
    episodes: pd.Series,
) -> pd.Series:
    """Calculate days since the last observed value in a series."""
    dates = pd.to_datetime(dates, errors="coerce")
    non_null_mask = series.notna()
    last_dates = dates.where(non_null_mask).groupby(episodes).ffill()
    difference = (dates - last_dates).dt.days
    return difference


def _calculate_sofa_component_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Build availability flags for the six SOFA components."""
    respiratory = _respiratory_component_available(df)
    coagulation = _simple_component_available(df, "plaquetes")
    hepatic = _simple_component_available(df, "bilirubina_total")
    neurologic = _simple_component_available(df, "GLASGOW")
    renal = _renal_component_available(df)
    cardiovascular = _cardiovascular_component_available(df)

    availability = pd.DataFrame(
        {
            "sofa_respiratory_available": respiratory.astype(int),
            "sofa_coagulation_available": coagulation.astype(int),
            "sofa_hepatic_available": hepatic.astype(int),
            "sofa_neurologic_available": neurologic.astype(int),
            "sofa_renal_available": renal.astype(int),
            "sofa_cardiovascular_available": cardiovascular.astype(int),
        }
    )

    availability_columns = list(availability.columns)
    availability["n_missing_sofa_components"] = (
        len(availability_columns) - availability[availability_columns].sum(axis=1)
    )
    availability["pct_missing_sofa_components_row"] = (
        availability["n_missing_sofa_components"] / len(availability_columns) * 100
    )
    return availability


def _simple_component_available(df: pd.DataFrame, column: str) -> pd.Series:
    """Return whether a simple column has a numeric value available on each row."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").notna()


def _respiratory_component_available(df: pd.DataFrame) -> pd.Series:
    """Return whether the respiratory component can be calculated with FiO2 and PaO2/O2SAT."""
    if "FIO2" not in df.columns:
        return pd.Series(False, index=df.index)

    fio2 = pd.to_numeric(df["FIO2"], errors="coerce")
    pao2 = (
        pd.to_numeric(df["pao2_arterial"], errors="coerce")
        if "pao2_arterial" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )
    o2sat = (
        pd.to_numeric(df["O2SAT"], errors="coerce")
        if "O2SAT" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )

    return fio2.notna() & (pao2.notna() | o2sat.notna())


def _renal_component_available(df: pd.DataFrame) -> pd.Series:
    """Return whether the renal component is available with creatinine."""
    return (
        pd.to_numeric(df["creatinina"], errors="coerce").notna()
        if "creatinina" in df.columns
        else pd.Series(False, index=df.index)
    )


def _cardiovascular_component_available(df: pd.DataFrame) -> pd.Series:
    """Return whether cardiovascular information exists through MAP or vasopressors."""
    map_available = (
        pd.to_numeric(df["TAM"], errors="coerce").notna()
        if "TAM" in df.columns
        else pd.Series(False, index=df.index)
    )

    vasopressor = pd.Series(False, index=df.index)
    for col in [
        "vasopressor_qualsevol",
        "vasopressor_multiple",
        "vasopressor_dobutamina",
        "vasopressor_dopamina",
        "vasopressor_noradrenalina",
        "vasopressor_adrenalina",
    ]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            vasopressor = vasopressor | series.notna()

    return map_available | vasopressor


def _summarize_missingness_by_episode(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the mean percentage of absent SOFA components per episode."""
    return (
        df.groupby("Episodi", as_index=False)
        .agg(
            n_episode_rows=("Episodi", "size"),
            pct_missing_sofa_components=("pct_missing_sofa_components_row", "mean"),
        )
        .assign(
            pct_missing_sofa_components=lambda x: x["pct_missing_sofa_components"].round(2)
        )
    )







