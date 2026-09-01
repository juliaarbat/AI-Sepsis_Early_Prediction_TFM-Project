"""Shared split validation and metadata helpers."""

from __future__ import annotations

import pandas as pd

from .predictive_model_24h import ID_COL, PATIENT_COL, TARGET


def normalize_split_unit(split_unit: str) -> str:
    """Normalize patient/episode split-unit aliases."""
    aliases = {
        "patient": "patient",
        PATIENT_COL.lower(): "patient",
        "episode": "episode",
        ID_COL.lower(): "episode",
    }
    key = split_unit.lower().strip()
    if key not in aliases:
        raise ValueError("split_unit must be 'patient'/'Nhc' or 'episode'/'Episodi'.")
    return aliases[key]


def split_unit(
    df_model: pd.DataFrame,
    split_unit_name: str,
    *,
    missing_patient_label: str = "episode",
) -> pd.Series:
    """Return the patient or episode unit that must stay in one split."""
    if split_unit_name == "patient" and PATIENT_COL in df_model.columns:
        return pd.Series(
            [
                ("patient", patient) if pd.notna(patient) else (missing_patient_label, episode)
                for patient, episode in zip(df_model[PATIENT_COL], df_model[ID_COL])
            ],
            index=df_model.index,
            dtype="object",
        )
    return df_model[ID_COL].map(lambda episode: ("episode", episode))


def split_date(
    df_model: pd.DataFrame,
    date_columns: tuple[str, ...] = ("data_index", "DataIngres", "DataIniciUrgencies"),
) -> pd.Series:
    """Return the date used to order rows chronologically."""
    for col in date_columns:
        if col in df_model.columns:
            return pd.to_datetime(df_model[col], errors="coerce")
    return pd.Series(pd.NaT, index=df_model.index)


def validate_train_valid_test_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Fail early when train, validation, or test are unusable.

    The training split must contain both positive and negative labels.
    """
    if train.empty or valid.empty or test.empty:
        raise ValueError("The train, validation, and test splits must all contain data.")
    if train[TARGET].nunique() < 2:
        raise ValueError("The train split needs positives and negatives to train the models.")


def validate_exclusive_patients_between_splits(df_model: pd.DataFrame) -> None:
    """Prevent the same patient from appearing in more than one split."""
    if PATIENT_COL not in df_model.columns:
        return

    splits_per_patient = df_model.dropna(subset=[PATIENT_COL]).groupby(PATIENT_COL)["split"].nunique()
    n_overlapping_patients = int(splits_per_patient.gt(1).sum())
    if n_overlapping_patients:
        raise RuntimeError(
            "Invalid split: some Nhc patients are present in more than one subset "
            f"({n_overlapping_patients} patients)."
        )


def summarize_split_audit(
    df_model: pd.DataFrame,
    split_order: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize size, dates, and prevalence for each split.

    One audit row is returned for each requested split.
    """
    data_col = split_date(df_model)
    df_tmp = df_model.assign(_data_split=data_col)
    rows: list[dict[str, object]] = []
    for split in split_order:
        subset = df_tmp.loc[df_tmp["split"] == split]
        if subset.empty:
            rows.append(_empty_split_audit_row(split))
            continue
        rows.append(
            {
                "split": split,
                "n_rows": int(len(subset)),
                "n_episodes": int(subset[ID_COL].nunique()),
                "n_patients": int(subset[PATIENT_COL].nunique()) if PATIENT_COL in subset else 0,
                "date_min": _format_date(subset["_data_split"].min()),
                "date_max": _format_date(subset["_data_split"].max()),
                "n_positives": int(subset[TARGET].sum()),
                "prevalence": _safe_div(float(subset[TARGET].sum()), len(subset)),
            }
        )
    return pd.DataFrame(rows)


def _empty_split_audit_row(split: str) -> dict[str, object]:
    """Build an empty split-audit row."""
    return {
        "split": split,
        "n_rows": 0,
        "n_episodes": 0,
        "n_patients": 0,
        "date_min": None,
        "date_max": None,
        "n_positives": 0,
        "prevalence": 0.0,
    }


def _format_date(value: object) -> str | None:
    """Format dates for CSV/JSON outputs."""
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")

