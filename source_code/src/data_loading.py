"""Load the original CSV and manage clean SOFA caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    DATE_COLUMNS,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_CRITICAL_CARE_RETURN_MIN_HOURS,
    SOFA_DATASETS_DIR,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    SOFA_PRE_DIR,
    SOFA_REPORTS_DIR,
)


PINNED_CSV: str | None = None

# Cached datasets are reused only after their schema and processing settings pass validation.
CLEAN_SOFA_CSV = SOFA_DATASETS_DIR / "daily_sepsis_model_clean_sofa.csv"
SOFA_CSV = SOFA_DATASETS_DIR / "daily_sepsis_model_with_sofa.csv"

REQUIRED_COLUMNS_WITH_SOFA = {
    "next_day_sepsis",
    "eligible_next_day_model_row",
    "plaquetes_pre_critical_return_used",
    "creatinina_pre_critical_return_used",
    "bilirubina_total_pre_critical_return_used",
    "operational_baseline_segment",
}

CRITICAL_CARE_RETURN_COLUMNS = {
    "en_critics_dia",
    "temps_critics_dia",
    "plaquetes_pre_retorn_critics_3d",
    "data_plaquetes_pre_retorn_critics_3d",
    "creatinina_pre_retorn_critics_3d",
    "data_creatinina_pre_retorn_critics_3d",
    "bilirubina_total_pre_retorn_critics_3d",
    "data_bilirubina_total_pre_retorn_critics_3d",
}

DATE_COLUMNS_CACHE = DATE_COLUMNS + [
    "operational_baseline_date",
    "first_sepsis_date",
]


def _resolve_csv_path() -> Path:
    """Return the pinned CSV path, or the latest extraction by default."""
    if PINNED_CSV:
        # A pinned file makes an analysis reproducible across later extractions.
        csv_path = DATA_DIR / PINNED_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Pinned CSV was not found: {csv_path}")
        return csv_path

    # The sorted final candidate is treated as the most recent extraction.
    candidates = sorted(DATA_DIR.glob("daily_sepsis_model_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No daily_sepsis_model_*.csv file was found in: {DATA_DIR}"
        )
    return candidates[-1]


def load_sepsis_model() -> pd.DataFrame:
    """Load the daily sepsis dataset with basic types prepared.

    Responsibilities:
    - locate the project CSV;
    - read it with pandas;
    - parse the date columns defined in `config.DATE_COLUMNS`;
    - convert `dia_relatiu` to numeric when present.

    Cleaning, imputation, and output creation are handled by the specific
    pipeline modules.
    """
    return _read_typed_csv(_resolve_csv_path(), "Loading data from")


def load_clean_sepsis_model_for_sofa(
    episode_missingness_threshold: float = 0.50,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
    episode_gap_exclusion_threshold_days: int | None = SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    max_allowed_date: str | pd.Timestamp | None = PRE_SOFA_MAX_ANALYSIS_DATE,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load the clean pre-SOFA dataset and regenerate the cache only if needed."""
    source_path = _resolve_csv_path()

    # Reuse the clean cache only when it matches the source and current policies.
    if _clean_sofa_cache_is_valid(
        source_path,
        lab_ffill_limit_days,
        vitals_ffill_limit_days,
        episode_gap_exclusion_threshold_days,
        regenerate,
    ):
        return _read_typed_csv(CLEAN_SOFA_CSV, "Loading clean pre-SOFA dataset from")

    if CLEAN_SOFA_CSV.exists():
        print(
            "\nThe pre-SOFA cache is outdated or does not preserve the "
            f"critical-care return columns; it will be regenerated: {CLEAN_SOFA_CSV}"
        )
    else:
        print(f"\nNo pre-SOFA cache was found; it will be generated: {CLEAN_SOFA_CSV}")

    from .sofa_cleaning import clean_data_for_sofa, save_sofa_cleaning_outputs

    df = load_sepsis_model()
    df_clean, exclusion_summary, pre_sofa_audit = clean_data_for_sofa(
        df,
        episode_missingness_threshold=episode_missingness_threshold,
        lab_ffill_limit_days=lab_ffill_limit_days,
        vitals_ffill_limit_days=vitals_ffill_limit_days,
        episode_gap_exclusion_threshold_days=episode_gap_exclusion_threshold_days,
        max_allowed_date=max_allowed_date,
    )
    save_sofa_cleaning_outputs(
        df_clean,
        exclusion_summary,
        pre_sofa_audit,
        episode_gap_exclusion_threshold_days=episode_gap_exclusion_threshold_days,
        lab_ffill_limit_days=lab_ffill_limit_days,
        vitals_ffill_limit_days=vitals_ffill_limit_days,
    )
    return df_clean


def load_sepsis_model_with_sofa(
    episode_missingness_threshold: float = 0.50,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
    episode_gap_exclusion_threshold_days: int | None = SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    max_allowed_date: str | pd.Timestamp | None = PRE_SOFA_MAX_ANALYSIS_DATE,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load the dataset with SOFA scores and labels, reusing a valid cache."""
    source_path = _resolve_csv_path()

    # Reuse the SOFA cache only when its labels, source columns and parameters match.
    if _sofa_cache_is_valid(
        source_path,
        lab_ffill_limit_days,
        vitals_ffill_limit_days,
        episode_gap_exclusion_threshold_days,
        regenerate,
    ):
        return _read_typed_csv(SOFA_CSV, "Loading dataset with SOFA from")

    if SOFA_CSV.exists():
        print(
            "\nThe SOFA cache is outdated, does not contain the predictive "
            "label, does not preserve the critical-care return columns, "
            "or does not match the temporal-gap policy; it will be regenerated: "
            f"{SOFA_CSV}"
        )
    else:
        print(f"\nNo SOFA cache was found; it will be generated: {SOFA_CSV}")

    from .sofa_calculation import compute_sofa, save_sofa_outputs

    df_clean = load_clean_sepsis_model_for_sofa(
        episode_missingness_threshold=episode_missingness_threshold,
        lab_ffill_limit_days=lab_ffill_limit_days,
        vitals_ffill_limit_days=vitals_ffill_limit_days,
        episode_gap_exclusion_threshold_days=episode_gap_exclusion_threshold_days,
        max_allowed_date=max_allowed_date,
        regenerate=regenerate,
    )
    df_sofa = compute_sofa(
        df_clean,
        lab_ffill_limit_days=lab_ffill_limit_days,
        vitals_ffill_limit_days=vitals_ffill_limit_days,
    )
    save_sofa_outputs(
        df_sofa,
        lab_ffill_limit_days=lab_ffill_limit_days,
        vitals_ffill_limit_days=vitals_ffill_limit_days,
    )
    return df_sofa


def _clean_sofa_cache_is_valid(
    source_path: Path,
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
    episode_gap_exclusion_threshold_days: int | None,
    regenerate: bool,
) -> bool:
    """Check whether the pre-SOFA cache can be reused."""
    return (
        not regenerate
        and _cache_is_fresh(CLEAN_SOFA_CSV, source_path)
        and _cache_preserves_source_columns(
            CLEAN_SOFA_CSV,
            source_path,
            CRITICAL_CARE_RETURN_COLUMNS,
        )
        and _sofa_cleaning_cache_parameters_ok(episode_gap_exclusion_threshold_days)
        and _sofa_cleaning_cache_limits_ok(
            lab_ffill_limit_days,
            vitals_ffill_limit_days,
        )
        and _sofa_cleaning_critical_care_return_policy_ok()
    )


def _sofa_cache_is_valid(
    source_path: Path,
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
    episode_gap_exclusion_threshold_days: int | None,
    regenerate: bool,
) -> bool:
    """Check whether the SOFA cache with labels can be reused."""
    return (
        not regenerate
        and _cache_is_fresh(SOFA_CSV, source_path)
        and _cache_has_columns(SOFA_CSV, REQUIRED_COLUMNS_WITH_SOFA)
        and _cache_preserves_source_columns(
            SOFA_CSV,
            source_path,
            CRITICAL_CARE_RETURN_COLUMNS,
        )
        and _sofa_cleaning_cache_parameters_ok(episode_gap_exclusion_threshold_days)
        and _sofa_cleaning_cache_limits_ok(
            lab_ffill_limit_days,
            vitals_ffill_limit_days,
        )
        and _sofa_cache_limits_ok(
            lab_ffill_limit_days,
            vitals_ffill_limit_days,
        )
        and _sofa_cache_critical_care_return_policy_ok()
    )


def _read_typed_csv(csv_path: Path, message: str) -> pd.DataFrame:
    """Read a CSV and normalize the column types shared by all pipelines."""
    print(f"\n{message}: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Coercion keeps malformed dates as missing values for downstream cleaning.
    for col in DATE_COLUMNS_CACHE:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "dia_relatiu" in df.columns:
        df["dia_relatiu"] = pd.to_numeric(df["dia_relatiu"], errors="coerce")

    return df


def _cache_is_fresh(cache_path: Path, source_path: Path) -> bool:
    """Return True when the cache is newer than the source extraction."""
    # A cache older than the source may not reflect the current extraction.
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _cache_has_columns(cache_path: Path, required_columns: set[str]) -> bool:
    """Return True when all required columns are present in the cached CSV."""
    if not cache_path.exists():
        return False
    try:
        columns = set(pd.read_csv(cache_path, nrows=0).columns)
    except Exception:
        return False
    return required_columns.issubset(columns)


def _cache_preserves_source_columns(
    cache_path: Path,
    source_path: Path,
    candidate_columns: set[str],
) -> bool:
    """Check that the cache has not lost candidate columns from the source CSV."""
    # Critical-care return variables must survive cleaning for later SOFA audits.
    if not cache_path.exists():
        return False
    try:
        source_columns = set(pd.read_csv(source_path, nrows=0).columns)
        cache_columns = set(pd.read_csv(cache_path, nrows=0).columns)
    except Exception:
        return False

    expected_columns = source_columns & candidate_columns
    return expected_columns.issubset(cache_columns)


def _sofa_cleaning_cache_parameters_ok(
    episode_gap_exclusion_threshold_days: int | None,
) -> bool:
    """Avoid reusing pre-SOFA caches created with a different gap policy."""
    summary_path = SOFA_PRE_DIR / "sofa_cleaning_summary.json"
    summary = _read_json_safely(summary_path)

    if "episode_gap_exclusion_threshold_days" not in summary:
        return False
    if summary.get("temporal_gap_criterion") != "large_gap_without_critical_care_record":
        return False
    return (
        summary.get("episode_gap_exclusion_threshold_days")
        == episode_gap_exclusion_threshold_days
    )


def _sofa_cleaning_cache_limits_ok(
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> bool:
    """Avoid reusing the pre-SOFA cache with old forward-fill limits."""
    summary_path = SOFA_PRE_DIR / "sofa_cleaning_summary.json"
    summary = _read_json_safely(summary_path)
    if not summary:
        return False

    return (
        summary.get("lab_ffill_limit_days") == lab_ffill_limit_days
        and summary.get("vitals_ffill_limit_days") == vitals_ffill_limit_days
    )


def _sofa_cleaning_critical_care_return_policy_ok() -> bool:
    """Avoid reusing pre-SOFA caches with a different critical-care threshold."""
    summary_path = SOFA_PRE_DIR / "sofa_cleaning_summary.json"
    summary = _read_json_safely(summary_path)
    return summary.get("critical_care_return_min_hours") == SOFA_CRITICAL_CARE_RETURN_MIN_HOURS


def _sofa_cache_limits_ok(
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> bool:
    """Avoid reusing the SOFA dataset if it was computed with other limits."""
    summary_path = SOFA_REPORTS_DIR / "sofa_summary.json"
    summary = _read_json_safely(summary_path)
    parameters = summary.get("parameters") if summary else None
    if not isinstance(parameters, dict):
        return False

    return (
        parameters.get("lab_ffill_limit_days") == lab_ffill_limit_days
        and parameters.get("vitals_ffill_limit_days") == vitals_ffill_limit_days
    )


def _sofa_cache_critical_care_return_policy_ok() -> bool:
    """Avoid reusing SOFA caches with a different critical-care threshold."""
    summary_path = SOFA_REPORTS_DIR / "sofa_summary.json"
    summary = _read_json_safely(summary_path)
    parameters = summary.get("parameters") if summary else None
    if not isinstance(parameters, dict):
        return False
    return (
        parameters.get("critical_care_return_min_hours")
        == SOFA_CRITICAL_CARE_RETURN_MIN_HOURS
    )


def _read_json_safely(path: Path) -> dict:
    """Read a cache JSON and return an empty dict if it cannot be used."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}






