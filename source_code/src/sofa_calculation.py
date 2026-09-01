"""Compute daily SOFA scores and sepsis labels used by the models."""

from __future__ import annotations

import json

import pandas as pd

from .config import (
    SOFA_CRITICAL_CARE_RETURN_MIN_HOURS,
    SOFA_DATASETS_DIR,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_REPORTS_DIR,
)


COMPONENTS_SOFA = [
    "sofa_respiratory",
    "sofa_coagulation",
    "sofa_hepatic",
    "sofa_cardiovascular",
    "sofa_neurologic",
    "sofa_renal",
]

# The operational baseline starts only once these labs are truly observed or
# explicitly recovered from the pre-critical-care return window.
REQUIRED_BASELINE_LABS = ["creatinina", "plaquetes"]


def compute_sofa(
    df: pd.DataFrame,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
) -> pd.DataFrame:
    """Compute daily SOFA scores and the project sepsis labels.

    The function applies the operational rules defined for this thesis:
    explicit imputation flags, time-limited forward filling, an operational
    baseline by segment, and sepsis defined as delta SOFA >= 2.
    """
    df_sofa = df.copy()

    _coerce_numeric_columns(df_sofa)
    _prepare_sofa_variables(
        df_sofa,
        lab_ffill_limit_days,
        vitals_ffill_limit_days,
    )
    _calculate_respiratory_ratios(df_sofa)

    df_sofa["sofa_respiratory"] = _score_respiratory(df_sofa)
    df_sofa["sofa_coagulation"] = _score_coagulation(df_sofa)
    df_sofa["sofa_hepatic"] = _score_hepatic(df_sofa)
    df_sofa["sofa_cardiovascular"] = _score_cardiovascular(df_sofa)
    df_sofa["sofa_neurologic"] = _score_neurologic(df_sofa)
    df_sofa["sofa_renal"] = _score_renal(df_sofa)

    df_sofa["sofa_available_components"] = df_sofa[COMPONENTS_SOFA].notna().sum(axis=1)
    df_sofa[COMPONENTS_SOFA] = df_sofa[COMPONENTS_SOFA].fillna(0)
    df_sofa["sofa_total"] = df_sofa[COMPONENTS_SOFA].sum(axis=1)
    df_sofa["delta_sofa_assumed_zero"] = df_sofa["sofa_total"]
    df_sofa["sofa_total_ge_2"] = (df_sofa["sofa_total"] >= 2).astype(int)

    if "Episodi" in df_sofa.columns:
        _add_operational_baseline(df_sofa)
        _add_baseline_sofa_and_delta(df_sofa)
        _add_sepsis_labels(df_sofa)
    else:
        _add_labels_without_episode(df_sofa)

    return df_sofa


def _add_sepsis_labels(df: pd.DataFrame) -> None:
    """Create sepsis labels once the baseline and delta SOFA already exist."""
    df["organ_dysfunction_sofa_ge_2"] = (
        df["first_day_delta_sofa"].ge(2).fillna(False).astype(int)
    )
    df["sepsis"] = df["organ_dysfunction_sofa_ge_2"]
    df["episode_sepsis"] = df.groupby("Episodi")["sepsis"].transform("max").astype(int)

    if "data_index" not in df.columns:
        df["first_sepsis_date"] = pd.NaT
        df["next_day_sepsis"] = 0
        df["eligible_next_day_model_row"] = 0
        return

    first_sepsis_date = (
        df.loc[df["sepsis"] == 1]
        .groupby("Episodi")["data_index"]
        .min()
    )
    df["first_sepsis_date"] = pd.to_datetime(
        df["Episodi"].map(first_sepsis_date.to_dict()),
        errors="coerce",
    )
    _add_next_day_label(df)


def _add_labels_without_episode(df: pd.DataFrame) -> None:
    """Create neutral labels when `Episodi` is missing, keeping the pipeline alive."""
    df["first_day_baseline_sofa"] = df["sofa_total"]
    df["first_day_delta_sofa"] = 0.0
    df["organ_dysfunction_sofa_ge_2"] = 0
    df["sepsis"] = 0
    df["episode_sepsis"] = df["sepsis"]
    df["first_sepsis_date"] = pd.NaT
    df["next_day_sepsis"] = 0
    df["eligible_next_day_model_row"] = 0


def save_sofa_outputs(
    df_sofa: pd.DataFrame,
    lab_ffill_limit_days: int | None = SOFA_LAB_FFILL_LIMIT_DAYS,
    vitals_ffill_limit_days: int | None = SOFA_VITALS_FFILL_LIMIT_DAYS,
) -> None:
    """Save the SOFA-enriched dataset and audit summaries as CSV/JSON files."""
    SOFA_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SOFA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_sofa_report_outputs()

    summary = _build_sofa_summary(
        df_sofa,
        lab_ffill_limit_days,
        vitals_ffill_limit_days,
    )

    with open(SOFA_REPORTS_DIR / "sofa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _save_sofa_audits(df_sofa)

    df_sofa.to_csv(SOFA_DATASETS_DIR / "daily_sepsis_model_with_sofa.csv", index=False)

    if "Episodi" in df_sofa.columns:
        summary_aggregations = {
            "episode_sepsis": ("episode_sepsis", "max"),
            "max_sofa_total": ("sofa_total", "max"),
            "first_day_baseline_sofa": ("first_day_baseline_sofa", "first"),
            "max_delta_sofa": ("first_day_delta_sofa", "max"),
            "first_sepsis_date": ("first_sepsis_date", "min"),
            "has_next_day_sepsis": ("next_day_sepsis", "max"),
            "n_model_days": ("Episodi", "size"),
        }
        if "previous_day_gap" in df_sofa.columns:
            summary_aggregations["max_day_gap"] = ("previous_day_gap", "max")

        episode_summary = (
            df_sofa.groupby("Episodi", as_index=False)
            .agg(**summary_aggregations)
        )
        episode_summary.to_csv(SOFA_REPORTS_DIR / "episode_sepsis_summary.csv", index=False)


def _clear_sofa_report_outputs() -> None:
    """Remove stale generated SOFA report tables before rewriting them."""
    for pattern in ("*.csv", "*.json"):
        for path in SOFA_REPORTS_DIR.glob(pattern):
            path.unlink()


def _save_sofa_audits(df_sofa: pd.DataFrame) -> None:
    """Save compact tables for reviewing whether SOFA depends heavily on imputations."""
    _audit_sofa_variables(df_sofa).to_csv(
        SOFA_REPORTS_DIR / "sofa_variable_audit.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    _audit_sofa_components(df_sofa).to_csv(
        SOFA_REPORTS_DIR / "sofa_component_audit.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def _audit_sofa_variables(df_sofa: pd.DataFrame) -> pd.DataFrame:
    """Summarize imputations, forward fills, and values without recent data."""
    rows = []
    for column, description in [
        ("FIO2_imputed_ambient_air", "Missing FiO2 assumed as ambient air"),
        ("GLASGOW_imputed_normal", "Missing Glasgow assumed as normal"),
        ("bilirubina_total_imputed_normal", "Missing bilirubin assumed as normal"),
        ("creatinina_forward_filled", "Creatinine forward-filled within episode"),
        ("plaquetes_forward_filled", "Platelets forward-filled within episode"),
        ("bilirubina_total_forward_filled", "Bilirubin forward-filled within episode"),
        ("TAM_forward_filled", "Mean arterial pressure forward-filled within episode"),
        ("O2SAT_forward_filled", "O2 saturation forward-filled within episode"),
        ("creatinina_without_recent_value", "Creatinine without a recent value"),
        ("plaquetes_without_recent_value", "Platelets without a recent value"),
    ]:
        if column not in df_sofa.columns:
            continue
        row_count = _count_flag(df_sofa, column) or 0
        rows.append(
            {
                "audit_variable": column,
                "description": description,
                "n_rows": row_count,
                "pct_rows": round(100 * row_count / len(df_sofa), 2) if len(df_sofa) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _audit_sofa_components(df_sofa: pd.DataFrame) -> pd.DataFrame:
    """Summarize component availability, baseline availability, and final labels."""
    rows = []
    if "sofa_available_components" in df_sofa.columns:
        counts = df_sofa["sofa_available_components"].value_counts(dropna=False).sort_index()
        for components, row_count in counts.items():
            rows.append(
                {
                    "block": "available_components",
                    "metric": f"{components}_components",
                    "n_rows": int(row_count),
                    "pct_rows": round(100 * row_count / len(df_sofa), 2) if len(df_sofa) else 0.0,
                }
            )

    for column, label in [
        ("operational_baseline_sofa_available", "operational_baseline_available"),
        ("pre_operational_baseline_row", "pre_operational_baseline_row"),
        ("sofa_total_ge_2", "sofa_total_ge_2"),
        ("sepsis", "sepsis"),
        ("eligible_next_day_model_row", "eligible_next_day_model_row"),
        ("next_day_sepsis", "next_day_sepsis"),
    ]:
        if column not in df_sofa.columns:
            continue
        row_count = _count_flag(df_sofa, column) or 0
        rows.append(
                {
                    "block": "labels_and_baseline",
                    "metric": label,
                    "n_rows": row_count,
                    "pct_rows": round(100 * row_count / len(df_sofa), 2) if len(df_sofa) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_sofa_summary(
    df_sofa: pd.DataFrame,
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> dict:
    """Build the SOFA audit JSON."""
    # Keep this JSON lightweight and stable because data_loading uses it to
    # decide whether an existing SOFA cache matches the current parameters.
    return {
        "n_rows": int(len(df_sofa)),
        "n_episodes": int(df_sofa["Episodi"].nunique()) if "Episodi" in df_sofa.columns else None,
        "parameters": {
            "lab_ffill_limit_days": lab_ffill_limit_days,
            "vitals_ffill_limit_days": vitals_ffill_limit_days,
            "critical_care_return_min_hours": SOFA_CRITICAL_CARE_RETURN_MIN_HOURS,
        },
        "imputation_criteria": _sofa_imputation_criteria(
            lab_ffill_limit_days,
            vitals_ffill_limit_days,
        ),
        "n_rows_sofa_total_ge_2": _count_flag(df_sofa, "sofa_total_ge_2"),
        "pct_rows_sofa_total_ge_2": _flag_percentage(df_sofa, "sofa_total_ge_2"),
        "n_episodes_sofa_total_ge_2": _episodes_with_flag(df_sofa, "sofa_total_ge_2"),
        "n_rows_delta_sofa_ge_2": _count_flag(df_sofa, "organ_dysfunction_sofa_ge_2"),
        "pct_rows_delta_sofa_ge_2": _flag_percentage(df_sofa, "organ_dysfunction_sofa_ge_2"),
        "n_episodes_delta_sofa_ge_2": _episodes_with_flag(df_sofa, "organ_dysfunction_sofa_ge_2"),
        "n_rows_sepsis": _count_flag(df_sofa, "sepsis"),
        "pct_rows_sepsis": _flag_percentage(df_sofa, "sepsis"),
        "n_episodes_sepsis": _episodes_with_flag(df_sofa, "sepsis"),
        "n_rows_next_day_sepsis": _count_flag(df_sofa, "next_day_sepsis"),
        "n_rows_eligible_for_next_day_model": _count_flag(df_sofa, "eligible_next_day_model_row"),
        "n_episodes_with_next_day_sepsis": _episodes_with_flag(df_sofa, "next_day_sepsis"),
    }


def _sofa_imputation_criteria(
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> dict[str, str | None]:
    """Describe the operational decisions applied during SOFA calculation."""
    return {
        "FIO2_missing": "21% ambient air",
        "GLASGOW_missing": "15 normal",
        "PaO2_FIO2_respiratory": "PaO2/FiO2 when PaO2 is available",
        "O2SAT_FIO2_respiratory": "O2SAT/FiO2 when PaO2 is missing",
        "bilirubina_missing": "hepatic normality if no recent value is available",
        "creatinina_plaquetes_missing": "not imputed as normal; requires observed, pre-return, or forward-filled value",
        "renal": "renal component calculated from creatinine; DIURESIS excluded due to missingness",
        "initial_missing_sofa_labs": "no longitudinal delta until first day with creatinine and platelets",
        "lab_forward_fill": _text_limit_ffill(lab_ffill_limit_days),
        "vitals_forward_fill": _text_limit_ffill(vitals_ffill_limit_days),
        "critical_care_return": (
            f"critical-care return >{SOFA_CRITICAL_CARE_RETURN_MIN_HOURS}h "
            "starts a new baseline segment"
        ),
        "sepsis": "delta SOFA >= 2 from the current operational baseline",
        "vasopressors": "without dose, dobutamine/dopamine raise minimum score to 2 and noradrenaline/adrenaline to 3",
    }


def _text_limit_ffill(limit_days: int | None) -> str:
    """Format the forward-fill time limit."""
    if limit_days is None:
        return "no time limit"
    return f"maximum {limit_days} day(s)"


def _count_flag(df: pd.DataFrame, column: str) -> int | None:
    """Count positive rows in a binary column, when present."""
    if column not in df.columns:
        return None
    return int((pd.to_numeric(df[column], errors="coerce") == 1).sum())


def _flag_percentage(df: pd.DataFrame, column: str) -> float | None:
    """Calculate the percentage of positive rows in a binary column."""
    if column not in df.columns:
        return None
    if not len(df):
        return 0.0
    return round(100 * (pd.to_numeric(df[column], errors="coerce") == 1).mean(), 2)


def _episodes_with_flag(df: pd.DataFrame, column: str) -> int | None:
    """Count episodes with at least one positive row."""
    if "Episodi" not in df.columns or column not in df.columns:
        return None
    mask = pd.to_numeric(df[column], errors="coerce") == 1
    return int(df.loc[mask, "Episodi"].nunique())


def _add_baseline_sofa_and_delta(df: pd.DataFrame) -> None:
    """Add the current baseline SOFA and the delta from that baseline."""
    if "operational_baseline_sofa" in df.columns:
        df["first_day_baseline_sofa"] = df["operational_baseline_sofa"]
    elif "data_index" in df.columns:
        order = pd.to_datetime(df["data_index"], errors="coerce")
        baseline = (
            df.assign(_baseline_sofa_order=order)
            .sort_values(["Episodi", "_baseline_sofa_order"], kind="stable")
            .groupby("Episodi")["sofa_total"]
            .first()
        )
        df["first_day_baseline_sofa"] = df["Episodi"].map(baseline)
    else:
        baseline = df.groupby("Episodi")["sofa_total"].first()
        df["first_day_baseline_sofa"] = df["Episodi"].map(baseline)

    df["first_day_delta_sofa"] = df["sofa_total"] - df["first_day_baseline_sofa"]

    if "pre_operational_baseline_row" in df.columns:
        pre_basal = pd.to_numeric(df["pre_operational_baseline_row"], errors="coerce").fillna(0).astype(bool)
        df.loc[pre_basal, "first_day_delta_sofa"] = pd.NA


def _add_operational_baseline(df: pd.DataFrame) -> None:
    """Determine the current longitudinal baseline by episode segment."""
    # The operational baseline starts once creatinine and platelets are observed,
    # including the pre-critical-care-return value when SQL recovered it explicitly.
    # A merely forward-filled value cannot define a new baseline.
    sufficient = pd.Series(True, index=df.index)
    for col in REQUIRED_BASELINE_LABS:
        original_col = f"{col}_original"
        base_col = original_col if original_col in df.columns else col
        if base_col not in df.columns:
            sufficient = pd.Series(False, index=df.index)
            break
        observed_value = pd.to_numeric(df[base_col], errors="coerce").notna()
        pre_return_flag = f"{col}_pre_critical_return_used"
        if pre_return_flag in df.columns:
            observed_value = observed_value | pd.to_numeric(
                df[pre_return_flag], errors="coerce"
            ).fillna(0).astype(bool)
        sufficient = sufficient & observed_value

    if "dia_relatiu" in df.columns:
        relative_day = pd.to_numeric(df["dia_relatiu"], errors="coerce")
        sufficient = sufficient & relative_day.ge(0).fillna(False)

    df["operational_baseline_sofa_sufficient"] = sufficient.astype(int)

    candidates = df.loc[sufficient].copy()
    if candidates.empty:
        df["operational_baseline_segment"] = _operational_baseline_segment(df)
        df["operational_baseline_sofa_index"] = pd.NA
        df["operational_baseline_sofa"] = pd.NA
        df["operational_baseline_date"] = pd.NaT
        df["operational_baseline_relative_day"] = pd.NA
        df["operational_baseline_sofa_available"] = 0
        df["pre_operational_baseline_row"] = 0
        return

    df["operational_baseline_segment"] = _operational_baseline_segment(df)

    candidates["_operational_baseline_idx"] = candidates.index
    candidates["operational_baseline_segment"] = df.loc[candidates.index, "operational_baseline_segment"]
    if "data_index" in candidates.columns:
        candidates["_operational_baseline_order"] = pd.to_datetime(candidates["data_index"], errors="coerce")
        sort_cols = ["Episodi", "_operational_baseline_order", "_operational_baseline_idx"]
    else:
        sort_cols = ["Episodi", "_operational_baseline_idx"]

    baseline_idx = (
        candidates.sort_values(sort_cols, kind="stable")
        .groupby(["Episodi", "operational_baseline_segment"])["_operational_baseline_idx"]
        .first()
    )
    baseline_idx_map = baseline_idx.to_dict()
    segment_keys = list(zip(df["Episodi"], df["operational_baseline_segment"]))
    df["operational_baseline_sofa_index"] = pd.Series(
        [baseline_idx_map.get(key, pd.NA) for key in segment_keys],
        index=df.index,
    )

    idx_to_sofa = df["sofa_total"].to_dict()
    df["operational_baseline_sofa"] = df["operational_baseline_sofa_index"].map(idx_to_sofa)

    if "data_index" in df.columns:
        idx_to_data = pd.to_datetime(df["data_index"], errors="coerce").to_dict()
        df["operational_baseline_date"] = pd.to_datetime(
            df["operational_baseline_sofa_index"].map(idx_to_data),
            errors="coerce",
        )
        current_date = pd.to_datetime(df["data_index"], errors="coerce")
        df["pre_operational_baseline_row"] = (
            df["operational_baseline_date"].notna() & current_date.lt(df["operational_baseline_date"])
        ).astype(int)
    else:
        df["operational_baseline_date"] = pd.NaT
        idx_num = pd.Series(df.index, index=df.index)
        baseline_idx_num = pd.to_numeric(df["operational_baseline_sofa_index"], errors="coerce")
        df["pre_operational_baseline_row"] = idx_num.lt(baseline_idx_num).fillna(False).astype(int)

    if "dia_relatiu" in df.columns:
        idx_to_relative_day = pd.to_numeric(df["dia_relatiu"], errors="coerce").to_dict()
        df["operational_baseline_relative_day"] = df["operational_baseline_sofa_index"].map(idx_to_relative_day)
    else:
        df["operational_baseline_relative_day"] = pd.NA

    df["operational_baseline_sofa_available"] = df["operational_baseline_sofa_index"].notna().astype(int)


def _operational_baseline_segment(df: pd.DataFrame) -> pd.Series:
    """Number baseline segments, restarting after prolonged critical-care stays."""
    if "Episodi" not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    if not {"data_hora_alta_critics", "data_index", "temps_critics"}.issubset(df.columns):
        return pd.Series(0, index=df.index, dtype="int64")

    critical_care_return = _critical_care_return_row(df)
    segment = pd.Series(0, index=df.index, dtype="int64")
    for _, idx in df.groupby("Episodi", sort=False).groups.items():
        idx_list = list(idx)
        if "data_index" in df.columns:
            idx_list = (
                df.loc[idx_list, ["data_index"]]
                .assign(_idx_original=idx_list)
                .sort_values(["data_index", "_idx_original"], kind="stable")
                .index
                .tolist()
            )
        segment.loc[idx_list] = critical_care_return.loc[idx_list].astype(int).cumsum().to_numpy()
    return segment


def _add_next_day_label(df: pd.DataFrame) -> None:
    """Add the main model label: next-day sepsis.

    For each episode, row D is used to predict whether the next observed day,
    D+1, has `sepsis = 1`.
    """
    df["data_index"] = pd.to_datetime(df["data_index"], errors="coerce")
    if "first_sepsis_date" not in df.columns:
        df["first_sepsis_date"] = pd.NaT
    df["first_sepsis_date"] = pd.to_datetime(df["first_sepsis_date"], errors="coerce")

    df["hours_to_next_day"] = pd.NA
    df["next_day_sepsis"] = 0

    day_sepsis = pd.to_numeric(df["sepsis"], errors="coerce").fillna(0).astype(bool)
    for _, idx in df.groupby("Episodi", sort=False).groups.items():
        idx_list = list(idx)
        if "data_index" in df.columns:
            idx_list = (
                df.loc[idx_list, ["data_index"]]
                .sort_values("data_index", kind="stable")
                .index
                .tolist()
            )
        dates = df.loc[idx_list, "data_index"]

        next_date = dates.shift(-1)
        next_sepsis = day_sepsis.loc[idx_list].shift(-1, fill_value=False)
        hours_to_next_day = (next_date - dates).dt.total_seconds() / 3600
        has_next_day = hours_to_next_day.gt(0) & hours_to_next_day.le(24)
        next_day_idx = pd.Index(idx_list)[has_next_day.fillna(False).to_numpy()]
        if len(next_day_idx):
            df.loc[next_day_idx, "hours_to_next_day"] = hours_to_next_day.loc[next_day_idx]
            positive_idx = next_day_idx[next_sepsis.loc[next_day_idx].astype(bool).to_numpy()]
            df.loc[positive_idx, "next_day_sepsis"] = 1

    if "pre_operational_baseline_row" in df.columns:
        row_from_baseline = ~pd.to_numeric(
            df["pre_operational_baseline_row"],
            errors="coerce",
        ).fillna(0).astype(bool)
    else:
        row_from_baseline = pd.Series(True, index=df.index)
    if "operational_baseline_sofa_available" in df.columns:
        row_from_baseline = row_from_baseline & pd.to_numeric(
            df["operational_baseline_sofa_available"],
            errors="coerce",
        ).fillna(0).astype(bool)

    has_next_day = pd.to_numeric(
        df["hours_to_next_day"],
        errors="coerce",
    ).gt(0)
    next_day_model_row = row_from_baseline & has_next_day
    df.loc[~next_day_model_row, "next_day_sepsis"] = 0
    df["eligible_next_day_model_row"] = next_day_model_row.astype(int)


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    """Convert SOFA input variables to numeric dtype."""
    sofa_columns = [
        "pao2_arterial",
        "O2SAT",
        "FIO2",
        "plaquetes",
        "bilirubina_total",
        "TAM",
        "vasopressor_qualsevol",
        "vasopressor_multiple",
        "vasopressor_dobutamina",
        "vasopressor_dopamina",
        "vasopressor_noradrenalina",
        "vasopressor_adrenalina",
        "GLASGOW",
        "creatinina",
        "porta_o2",
    ]

    for col in sofa_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _prepare_sofa_variables(
    df: pd.DataFrame,
    lab_ffill_limit_days: int | None,
    vitals_ffill_limit_days: int | None,
) -> None:
    """Sort rows and prepare variables before component scoring."""
    if {"Episodi", "data_index"}.issubset(df.columns):
        df["data_index"] = pd.to_datetime(df["data_index"], errors="coerce")
        df.sort_values(["Episodi", "data_index"], kind="stable", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if "previous_day_gap" not in df.columns:
            df["previous_day_gap"] = (
                df.groupby("Episodi")["data_index"].diff().dt.total_seconds() / 86400
            )

    _prepare_respiratory_sofa(df, vitals_ffill_limit_days)
    _prepare_cardiovascular_sofa(df, vitals_ffill_limit_days)
    _prepare_glasgow_sofa(df)
    _prepare_laboratory_sofa(df, lab_ffill_limit_days)


def _prepare_respiratory_sofa(
    df: pd.DataFrame,
    ffill_limit_days: int | None,
) -> None:
    """Prepare PaO2, O2SAT, and FiO2 for the respiratory component."""
    # PaO2 is not forward-filled; O2SAT is; missing FiO2 is treated as ambient air.
    _prepare_pao2_without_forward_fill(df)
    _prepare_sofa_measure(
        df,
        "O2SAT",
        ffill_limit_days=ffill_limit_days,
    )
    _prepare_ambient_fio2(df)


def _prepare_cardiovascular_sofa(
    df: pd.DataFrame,
    ffill_limit_days: int | None,
) -> None:
    """Prepare mean arterial pressure for the cardiovascular SOFA component."""
    _prepare_sofa_measure(
        df,
        "TAM",
        ffill_limit_days=ffill_limit_days,
    )


def _prepare_pao2_without_forward_fill(df: pd.DataFrame) -> None:
    """Keep PaO2 only when observed on the same day."""
    if "pao2_arterial" not in df.columns:
        return

    original = pd.to_numeric(df["pao2_arterial"], errors="coerce")
    df["pao2_arterial_original"] = original
    df["pao2_arterial"] = original


def _prepare_ambient_fio2(df: pd.DataFrame) -> None:
    """Impute missing FiO2 as ambient air and store the imputation flag."""
    if "FIO2" not in df.columns:
        df["FIO2"] = 21.0
        df["FIO2_original"] = pd.NA
        df["FIO2_imputed_ambient_air"] = 1
        return

    original = pd.to_numeric(df["FIO2"], errors="coerce")
    df["FIO2_original"] = original
    df["FIO2_imputed_ambient_air"] = original.isna().astype(int)
    df["FIO2"] = original.fillna(21)


def _prepare_glasgow_sofa(df: pd.DataFrame) -> None:
    """Impute missing Glasgow as normal and preserve the original value."""
    if "GLASGOW" not in df.columns:
        df["GLASGOW"] = 15.0
        df["GLASGOW_original"] = pd.NA
        df["GLASGOW_imputed_normal"] = 1
        return

    original = pd.to_numeric(df["GLASGOW"], errors="coerce")
    df["GLASGOW_original"] = original
    df["GLASGOW_imputed_normal"] = original.isna().astype(int)
    df["GLASGOW"] = original.fillna(15)


def _prepare_laboratory_sofa(
    df: pd.DataFrame,
    lab_ffill_limit_days: int | None,
) -> None:
    """Prepare platelets, bilirubin, and creatinine with the operational SOFA rules."""
    # Missing bilirubin is treated as hepatic normality if there is no observed
    # or recently recovered value. Creatinine and platelets, however, must be
    # observed or forward-filled within the time limit.
    normal_values = {"bilirubina_total": 0.8}
    lab_cols = [
        col
        for col in ["plaquetes", "bilirubina_total", "creatinina"]
        if col in df.columns
    ]
    critical_care_return = _critical_care_return_row(df)
    for col in lab_cols:
        original = pd.to_numeric(df[col], errors="coerce")
        base_series = original
        df[f"{col}_original"] = original

        pre_return_col = _critical_care_return_lab_column(df, col)
        if pre_return_col in df.columns:
            pre_return = pd.to_numeric(df[pre_return_col], errors="coerce")
            # Critical-care return priority:
            # 1) observed same-day value; 2) pre-return value from SQL;
            # 3) later forward-fill within the time limit.
            use_pre_return = original.isna() & pre_return.notna() & critical_care_return
            base_series = original.where(~use_pre_return, pre_return)
            df[f"{col}_pre_critical_return_used"] = use_pre_return.astype(int)
        else:
            df[f"{col}_pre_critical_return_used"] = 0

        prepared = _forward_fill_by_episode(
            df,
            base_series,
            ffill_limit_days=lab_ffill_limit_days,
        )
        df[f"{col}_forward_filled"] = (base_series.isna() & prepared.notna()).astype(int)
        df[f"{col}_without_recent_value"] = prepared.isna().astype(int)
        if col == "bilirubina_total":
            df[f"{col}_imputed_normal"] = prepared.isna().astype(int)
            df[col] = prepared.fillna(normal_values[col])
        else:
            df[f"{col}_imputed_normal"] = 0
            df[col] = prepared


def _critical_care_return_lab_column(df: pd.DataFrame, col: str) -> str:
    """Return the original CSV column for pre-return laboratory values."""
    return f"{col}_pre_retorn_critics_3d"


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

    # Compatibility with old extracts: if SQL only fills pre-return columns on
    # the return row, do not block their use.
    return pd.Series(True, index=df.index)


def _prepare_sofa_measure(
    df: pd.DataFrame,
    col: str,
    ffill_limit_days: int | None = 1,
) -> None:
    """Prepare a generic same-episode measure with time-limited forward fill."""
    if col not in df.columns:
        return

    original = pd.to_numeric(df[col], errors="coerce")
    prepared = _forward_fill_by_episode(
        df,
        original,
        ffill_limit_days=ffill_limit_days,
    )
    df[f"{col}_original"] = original
    df[f"{col}_forward_filled"] = (original.isna() & prepared.notna()).astype(int)
    df[col] = prepared


def _forward_fill_by_episode(
    df: pd.DataFrame,
    series: pd.Series,
    ffill_limit_days: int | None = 1,
) -> pd.Series:
    """Propagate the last known value within each episode."""
    if "Episodi" in df.columns:
        prepared = series.groupby(df["Episodi"]).ffill()
        if ffill_limit_days is None or "data_index" not in df.columns:
            return prepared

        dates = pd.to_datetime(df["data_index"], errors="coerce")
        last_date = dates.where(series.notna()).groupby(df["Episodi"]).ffill()
        days_since_last = (dates - last_date).dt.total_seconds() / 86400
        exceeds_limit = days_since_last > ffill_limit_days
        return prepared.mask(exceeds_limit)

    prepared = series.ffill()
    if ffill_limit_days is None or "data_index" not in df.columns:
        return prepared

    dates = pd.to_datetime(df["data_index"], errors="coerce")
    last_date = dates.where(series.notna()).ffill()
    days_since_last = (dates - last_date).dt.total_seconds() / 86400
    exceeds_limit = days_since_last > ffill_limit_days
    return prepared.mask(exceeds_limit)


def _calculate_respiratory_ratios(df: pd.DataFrame) -> None:
    """Calculate PaFi, SaFi, and select the available respiratory ratio."""
    if "FIO2" not in df.columns:
        df["PaFi"] = pd.NA
        df["SaFi"] = pd.NA
        df["respiratory_ratio"] = pd.NA
        df["respiratory_ratio_type"] = pd.NA
        return

    fio2_decimal = df["FIO2"] / 100
    fio2_decimal = fio2_decimal.where(fio2_decimal > 0)

    if "pao2_arterial" in df.columns:
        df["PaFi"] = df["pao2_arterial"] / fio2_decimal
    else:
        df["PaFi"] = pd.NA

    if "O2SAT" in df.columns:
        df["SaFi"] = df["O2SAT"] / fio2_decimal
    else:
        df["SaFi"] = pd.NA

    df["respiratory_ratio"] = df["PaFi"].where(df["PaFi"].notna(), df["SaFi"])
    df["respiratory_ratio_type"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["SaFi"].notna(), "respiratory_ratio_type"] = "SaFi_O2SAT_FIO2"
    df.loc[df["PaFi"].notna(), "respiratory_ratio_type"] = "PaFi_PaO2_FIO2"


def _score_respiratory(df: pd.DataFrame) -> pd.Series:
    """Score the respiratory SOFA component with PaFi or SaFi, depending on availability."""
    if "respiratory_ratio" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    ratio = pd.to_numeric(df["respiratory_ratio"], errors="coerce")
    ratio_type = (
        df["respiratory_ratio_type"]
        if "respiratory_ratio_type" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    oxygen_support = (
        pd.to_numeric(df["porta_o2"], errors="coerce").fillna(0)
        if "porta_o2" in df.columns
        else pd.Series(0, index=df.index)
    )

    score = pd.Series(pd.NA, index=df.index, dtype="Float64")

    pafi_mask = ratio_type == "PaFi_PaO2_FIO2"
    score = score.mask(pafi_mask & (ratio > 400), 0)
    score = score.mask(pafi_mask & (ratio <= 400) & (ratio > 300), 1)
    score = score.mask(pafi_mask & (ratio <= 300) & (ratio > 200), 2)
    score = score.mask(pafi_mask & (ratio <= 200) & (ratio > 100) & (oxygen_support == 1), 3)
    score = score.mask(pafi_mask & (ratio <= 100) & (oxygen_support == 1), 4)
    score = score.mask(pafi_mask & (ratio <= 200) & (oxygen_support != 1), 2)

    # SaFi: approximate equivalence for scoring the respiratory component with O2SAT/FiO2.
    safi_mask = ratio_type == "SaFi_O2SAT_FIO2"
    score = score.mask(safi_mask & (ratio > 512), 0)
    score = score.mask(safi_mask & (ratio <= 512) & (ratio > 357), 1)
    score = score.mask(safi_mask & (ratio <= 357) & (ratio > 214), 2)
    score = score.mask(safi_mask & (ratio <= 214) & (ratio > 89) & (oxygen_support == 1), 3)
    score = score.mask(safi_mask & (ratio <= 89) & (oxygen_support == 1), 4)
    score = score.mask(safi_mask & (ratio <= 214) & (oxygen_support != 1), 2)

    return score


def _score_coagulation(df: pd.DataFrame) -> pd.Series:
    """Score the coagulation component from platelet count."""
    if "plaquetes" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    platelets = pd.to_numeric(df["plaquetes"], errors="coerce")
    score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    score = score.mask(platelets > 150, 0)
    score = score.mask((platelets <= 150) & (platelets > 100), 1)
    score = score.mask((platelets <= 100) & (platelets > 50), 2)
    score = score.mask((platelets <= 50) & (platelets > 20), 3)
    score = score.mask(platelets <= 20, 4)
    return score


def _score_hepatic(df: pd.DataFrame) -> pd.Series:
    """Score the hepatic component from total bilirubin."""
    if "bilirubina_total" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    bilirubin = pd.to_numeric(df["bilirubina_total"], errors="coerce")
    score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    score = score.mask(bilirubin < 1.2, 0)
    score = score.mask((bilirubin >= 1.2) & (bilirubin < 2.0), 1)
    score = score.mask((bilirubin >= 2.0) & (bilirubin < 6.0), 2)
    score = score.mask((bilirubin >= 6.0) & (bilirubin < 12.0), 3)
    score = score.mask(bilirubin >= 12.0, 4)
    return score


def _score_cardiovascular(df: pd.DataFrame) -> pd.Series:
    """Score the cardiovascular component from MAP and vasopressor presence."""
    map_pressure = (
        pd.to_numeric(df["TAM"], errors="coerce")
        if "TAM" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )
    dobutamine = (
        pd.to_numeric(df["vasopressor_dobutamina"], errors="coerce").fillna(0)
        if "vasopressor_dobutamina" in df.columns
        else pd.Series(0, index=df.index)
    )
    dopamine = (
        pd.to_numeric(df["vasopressor_dopamina"], errors="coerce").fillna(0)
        if "vasopressor_dopamina" in df.columns
        else pd.Series(0, index=df.index)
    )
    noradrenaline = (
        pd.to_numeric(df["vasopressor_noradrenalina"], errors="coerce").fillna(0)
        if "vasopressor_noradrenalina" in df.columns
        else pd.Series(0, index=df.index)
    )
    adrenaline = (
        pd.to_numeric(df["vasopressor_adrenalina"], errors="coerce").fillna(0)
        if "vasopressor_adrenalina" in df.columns
        else pd.Series(0, index=df.index)
    )
    vasopressor_minimum = pd.Series(0, index=df.index, dtype="Float64")
    # Without vasopressor dose, raise the minimum score when the drug is present.
    vasopressor_minimum = vasopressor_minimum.mask((dobutamine == 1) | (dopamine == 1), 2)
    vasopressor_minimum = vasopressor_minimum.mask((noradrenaline == 1) | (adrenaline == 1), 3)

    score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    score = score.mask(map_pressure > 70, 0)
    score = score.mask((map_pressure <= 70) & (map_pressure >= 60), 1)
    score = score.mask((map_pressure < 60) & (map_pressure >= 50), 2)
    score = score.mask((map_pressure < 50) & (map_pressure >= 40), 3)
    score = score.mask(map_pressure < 40, 4)
    score = pd.concat([score, vasopressor_minimum], axis=1).max(axis=1)
    score = score.fillna(0)
    return score


def _score_neurologic(df: pd.DataFrame) -> pd.Series:
    """Score the neurologic component from the Glasgow coma scale."""
    if "GLASGOW" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    glasgow = pd.to_numeric(df["GLASGOW"], errors="coerce")
    score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    score = score.mask(glasgow == 15, 0)
    score = score.mask((glasgow >= 13) & (glasgow <= 14), 1)
    score = score.mask((glasgow >= 10) & (glasgow <= 12), 2)
    score = score.mask((glasgow >= 6) & (glasgow <= 9), 3)
    score = score.mask(glasgow < 6, 4)
    return score


def _score_renal(df: pd.DataFrame) -> pd.Series:
    """Score the renal component from creatinine."""
    creatinine = (
        pd.to_numeric(df["creatinina"], errors="coerce")
        if "creatinina" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )

    creatinine_score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    creatinine_score = creatinine_score.mask(creatinine < 1.2, 0)
    creatinine_score = creatinine_score.mask((creatinine >= 1.2) & (creatinine < 2.0), 1)
    creatinine_score = creatinine_score.mask((creatinine >= 2.0) & (creatinine < 3.5), 2)
    creatinine_score = creatinine_score.mask((creatinine >= 3.5) & (creatinine < 5.0), 3)
    creatinine_score = creatinine_score.mask(creatinine >= 5.0, 4)

    return creatinine_score








