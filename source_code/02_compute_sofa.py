"""Prepare the modelling dataset and compute daily SOFA labels."""

from __future__ import annotations

import _bootstrap

assert _bootstrap.PROJECT_ROOT

import pandas as pd

from src.sofa_calculation import compute_sofa, save_sofa_outputs
from src.data_loading import load_sepsis_model
from src.config import (
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    SOFA_OUTPUTS_DIR,
    SOFA_REPORTS_DIR,
)
from src.general_eda import run_general_eda
from src.sofa_cleaning import clean_data_for_sofa, save_sofa_cleaning_outputs
from src.progress import log_end, log_start, step


EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
TOTAL_STEPS = 5
POST_SOFA_EDA_SUBFOLDER = f"{SOFA_OUTPUTS_DIR.name}/eda_post_sofa"
VARIABLE_DECISION_CRITERIA_PATH = SOFA_REPORTS_DIR / "variable_decision_criteria.txt"


VARIABLE_DECISION_CRITERIA = """CRITERIA FOR DECIDING WHEN TO REMOVE VARIABLES

1. Before computing SOFA
   No predictive variable selection is performed at this stage. Cleaning is
   limited to the minimum needed to compute SOFA reliably: invalid or future
   dates, exact duplicate rows, physiologically implausible values, and
   inconsistent binary flags. Episodes with excessive missingness in SOFA
   components are also excluded, as are episodes with very large unexplained
   temporal gaps and no critical-care record.

2. During SOFA computation
   Original variables are preserved whenever possible. Operational decisions
   are recorded in derived columns: ambient FiO2 assumptions, normal Glasgow
   assumptions, normal bilirubin assumptions, time-limited forward fills, and
   respiratory ratio type. Creatinine and platelets are not imputed as normal
   when no recent value is available.

3. After computing SOFA
   The post-SOFA EDA reviews SOFA prevalence, score distribution, component
   coverage, and the signal in `next_day_sepsis`. This step helps identify
   candidate variables for later exclusion or review, but it does not modify
   the base dataset.

4. Before/inside the models
   Final predictor selection is performed inside the model preprocessing.
   Missingness, category handling, and exclusion rules are learned only from
   the training split to prevent leakage. Recommended criteria:
   - remove variables with >80% missingness in train;
   - remove identifiers, absolute dates, and future-information variables;
   - remove SOFA-derived variables when they would leak outcome information;
   - keep clinically meaningful variables with moderate missingness, adding a
     missingness indicator when needed.
"""


def _count_episodes(df: pd.DataFrame) -> int:
    """Return the number of episodes when the column exists."""
    return int(df["Episodi"].nunique()) if "Episodi" in df.columns else 0


def _count_flag_rows(df: pd.DataFrame, column: str) -> int:
    """Count rows where a binary column equals 1."""
    if column not in df.columns:
        return 0
    return int((pd.to_numeric(df[column], errors="coerce") == 1).sum())


def _count_episodes_with_flag(df: pd.DataFrame, column: str) -> int:
    """Count episodes with at least one positive row in a binary column."""
    if "Episodi" not in df.columns or column not in df.columns:
        return 0
    mask = pd.to_numeric(df[column], errors="coerce") == 1
    return int(df.loc[mask, "Episodi"].nunique())


def _pre_sofa_audit_as_dict(
    pre_sofa_audit: dict[str, pd.DataFrame] | None,
) -> dict[str, object]:
    """Convert the pre-SOFA audit table into an easy lookup dictionary."""
    if not pre_sofa_audit or "summary" not in pre_sofa_audit:
        return {}
    summary = pre_sofa_audit["summary"]
    return dict(zip(summary["metric"], summary["value"]))


def print_sofa_summary(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_sofa: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    pre_sofa_audit: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Print a final sanity-check summary for the SOFA pipeline."""
    excluded_episodes = int((exclusion_summary["exclude_episode"] == 1).sum())
    pre_sofa_summary = _pre_sofa_audit_as_dict(pre_sofa_audit)

    print("\nFINAL SOFA SUMMARY")
    print("Original rows:", len(df_original))
    print("Original episodes:", _count_episodes(df_original))

    print("\nAfter cleaning:")
    print(f"Rows after cleaning: {len(df_clean)}")
    print(f"Episodes after cleaning: {_count_episodes(df_clean)}")
    print(f"Episodes excluded during cleaning: {excluded_episodes}")
    if "exclude_for_temporal_gap" in exclusion_summary.columns:
        gap_exclusions = int((exclusion_summary["exclude_for_temporal_gap"] == 1).sum())
        print(f"Episodes excluded for unexplained temporal gaps: {gap_exclusions}")
    if pre_sofa_summary:
        print(
            "Rows removed because data_index was in the future:",
            pre_sofa_summary.get("n_rows_removed_future_data_index", 0),
        )
        print(
            "Rows removed because data_index was invalid:",
            pre_sofa_summary.get("n_rows_removed_invalid_data_index", 0),
        )
        print(
            "Exact duplicate rows removed:",
            pre_sofa_summary.get("n_exact_duplicate_rows_removed", 0),
        )
        print(
            "Numeric values converted to NaN:",
            pre_sofa_summary.get("n_numeric_values_set_to_nan", 0),
        )

    print("\nAfter SOFA computation:")
    print(f"Rows with SOFA: {len(df_sofa)}")
    print(f"Episodes with SOFA: {_count_episodes(df_sofa)}")
    print("Criterion: sepsis = delta SOFA >= 2 relative to the operational baseline")
    print(f"Rows with total SOFA >= 2: {_count_flag_rows(df_sofa, 'sofa_total_ge_2')}")
    print(
        "Episodes with at least one day with total SOFA >= 2:",
        _count_episodes_with_flag(df_sofa, "sofa_total_ge_2"),
    )
    print(f"Rows classified as sepsis: {_count_flag_rows(df_sofa, 'sepsis')}")
    print(f"Episodes classified as sepsis: {_count_episodes_with_flag(df_sofa, 'sepsis')}")

    print("\nPredictive labels:")
    if {"eligible_next_day_model_row", "next_day_sepsis"}.issubset(df_sofa.columns):
        print(
            "Rows eligible for next-day prediction:",
            _count_flag_rows(df_sofa, "eligible_next_day_model_row"),
        )
        print(
            "Rows positive for next-day sepsis:",
            _count_flag_rows(df_sofa, "next_day_sepsis"),
        )
        print(
            "Episodes with at least one positive next-day row:",
            _count_episodes_with_flag(df_sofa, "next_day_sepsis"),
        )


def write_variable_decision_criteria() -> None:
    """Write a short guide for deciding when to remove episodes or variables."""
    SOFA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VARIABLE_DECISION_CRITERIA_PATH.write_text(
        VARIABLE_DECISION_CRITERIA,
        encoding="utf-8",
    )


def main() -> None:
    """Run the full workflow: load, clean, compute SOFA, and persist outputs."""
    title = "SOFA computation"
    log_start(title)

    with step("Load the daily_sepsis_model dataset", number=1, total=TOTAL_STEPS):
        df_original = load_sepsis_model()

    with step("Clean and prepare data for SOFA", number=2, total=TOTAL_STEPS):
        df_clean, exclusion_summary, pre_sofa_audit = clean_data_for_sofa(
            df_original,
            episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
            episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
            max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
        )
        save_sofa_cleaning_outputs(
            df_clean,
            exclusion_summary,
            pre_sofa_audit,
            episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        )

    with step("Compute SOFA and sepsis labels", number=3, total=TOTAL_STEPS):
        df_sofa = compute_sofa(
            df_clean,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        )

    with step("Save SOFA outputs", number=4, total=TOTAL_STEPS):
        save_sofa_outputs(
            df_sofa,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        )

    with step("Generate post-SOFA EDA and decision criteria", number=5, total=TOTAL_STEPS):
        run_general_eda(
            df_sofa,
            output_subfolder=POST_SOFA_EDA_SUBFOLDER,
            title_label="post-sofa",
        )
        write_variable_decision_criteria()

    print_sofa_summary(df_original, df_clean, df_sofa, exclusion_summary, pre_sofa_audit)
    print("SOFA outputs:", SOFA_OUTPUTS_DIR)
    print("EDA post-SOFA:", SOFA_OUTPUTS_DIR / "eda_post_sofa")
    print("Variable decision criteria:", VARIABLE_DECISION_CRITERIA_PATH)
    log_end(title)


if __name__ == "__main__":
    main()




