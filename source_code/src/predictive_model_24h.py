"""Shared utilities for next-day sepsis prediction models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .output_contracts import TOP_RISK_LEVEL_DAY, TOP_RISK_LEVEL_EPISODE

TARGET = "next_day_sepsis"
ELIGIBILITY_COL = "eligible_next_day_model_row"
ID_COL = "Episodi"
PATIENT_COL = "Nhc"

EXCLUDED_FEATURES = {
    "Episodi",
    "Nhc",
    "data_index",
    "DataIngres",
    "DataIniciUrgencies",
    "DataAlta",
    "data_hora_alta_critics",
    "hemocultiu_positiu_data_extraccio",
    "sepsis",
    "episode_sepsis",
    "first_sepsis_date",
    "hours_to_next_day",
    "next_day_sepsis",
    "eligible_next_day_model_row",
    "organ_dysfunction_sofa_ge_2",
    "sofa_total_ge_2",
}

FUTURE_OR_SUMMARY_COLUMNS = {
    "passa_per_critics",
    "temps_critics",
}

SOFA_DERIVED_SUFFIXES = (
    "_original",
    "_forward_filled",
    "_without_recent_value",
    "_imputed_normal",
    "_imputed_ambient_air",
    "_pre_retorn_critics_3d",
    "_pre_critical_return_used",
    "_disponible",
)

SOFA_DERIVED_COLUMNS = {
    "FIO2_original",
    "FIO2_imputed_ambient_air",
    "O2SAT_original",
    "O2SAT_forward_filled",
    "TAM_original",
    "TAM_forward_filled",
    "GLASGOW_original",
    "GLASGOW_imputed_normal",
    "plaquetes_original",
    "plaquetes_forward_filled",
    "plaquetes_without_recent_value",
    "plaquetes_pre_retorn_critics_3d",
    "data_plaquetes_pre_retorn_critics_3d",
    "plaquetes_pre_critical_return_used",
    "plaquetes_imputed_normal",
    "bilirubina_total_original",
    "bilirubina_total_forward_filled",
    "bilirubina_total_without_recent_value",
    "bilirubina_total_pre_retorn_critics_3d",
    "data_bilirubina_total_pre_retorn_critics_3d",
    "bilirubina_total_pre_critical_return_used",
    "bilirubina_total_imputed_normal",
    "creatinina_original",
    "creatinina_forward_filled",
    "creatinina_without_recent_value",
    "creatinina_pre_retorn_critics_3d",
    "data_creatinina_pre_retorn_critics_3d",
    "creatinina_pre_critical_return_used",
    "creatinina_imputed_normal",
    "PaFi",
    "SaFi",
    "respiratory_ratio",
    "respiratory_ratio_type",
    "sofa_respiratory",
    "sofa_coagulation",
    "sofa_hepatic",
    "sofa_cardiovascular",
    "sofa_neurologic",
    "sofa_renal",
    "sofa_available_components",
    "sofa_total",
    "delta_sofa_assumed_zero",
    "operational_baseline_sofa_sufficient",
    "operational_baseline_segment",
    "operational_baseline_sofa_index",
    "operational_baseline_sofa",
    "operational_baseline_date",
    "pre_operational_baseline_row",
    "operational_baseline_relative_day",
    "operational_baseline_sofa_available",
    "first_day_baseline_sofa",
    "first_day_delta_sofa",
}

@dataclass
class Preprocessor:
    numeric_columns: list[str]
    categorical_levels: dict[str, list[str]]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    max_missing_ratio: float
    excluded_high_missing_columns: list[str]
    missing_ratios: dict[str, float]
    excluded_leakage_columns: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        """Return the expanded numeric, missingness, and one-hot feature names."""
        names: list[str] = []
        for col in self.numeric_columns:
            names.append(col)
            names.append(f"{col}__missing")
        for col, levels in self.categorical_levels.items():
            names.extend(f"{col}__{level}" for level in levels)
        return names


def prepare_model_dataset_24h(df_sofa: pd.DataFrame) -> pd.DataFrame:
    """Filter eligible rows for next-day sepsis modeling."""
    # Check that the required columns are available: episode, target, and eligibility.
    required_columns = {ID_COL, TARGET, ELIGIBILITY_COL}
    missing_columns = sorted(required_columns - set(df_sofa.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for the 24h model: {', '.join(missing_columns)}")

    # Keep only rows with an observable next day and a known episode.
    df_model = df_sofa.loc[df_sofa[ELIGIBILITY_COL] == 1].copy()
    df_model = df_model.loc[df_model[ID_COL].notna()].copy()
    df_model[TARGET] = pd.to_numeric(df_model[TARGET], errors="coerce").fillna(0).astype(int)

    # If days since admission is missing, derive it from the relative day.
    if "dia_relatiu" in df_model.columns and "dies_des_ingres" not in df_model.columns:
        days = pd.to_numeric(df_model["dia_relatiu"], errors="coerce")
        df_model["dies_des_ingres"] = days.clip(lower=0)

    return df_model.reset_index(drop=True)


def fit_preprocessor(
    train: pd.DataFrame,
    max_missing_ratio: float = 0.80,
    force_categorical_columns: set[str] | None = None,
) -> Preprocessor:
    """Learn imputations and encodings from train only to avoid leakage."""
    force_categorical_columns = force_categorical_columns or set()

    # First select candidate columns, removing IDs, labels, and SOFA-derived variables.
    candidates = []
    excluded_leakage_columns = []
    for col in train.columns:
        if col in EXCLUDED_FEATURES:
            excluded_leakage_columns.append(col)
            continue
        if col in FUTURE_OR_SUMMARY_COLUMNS:
            excluded_leakage_columns.append(col)
            continue
        if _is_sofa_or_availability_derived_column(col):
            excluded_leakage_columns.append(col)
            continue
        if col == "split":
            continue
        candidates.append(col)

    # Compute missingness from train only. Columns with too many missing values are excluded.
    missing_ratios = {}
    excluded_high_missing_columns = []
    for col in candidates:
        missing_ratio = float(train[col].isna().mean())
        missing_ratios[col] = missing_ratio
        if missing_ratio > max_missing_ratio:
            excluded_high_missing_columns.append(col)

    candidates = [col for col in candidates if col not in excluded_high_missing_columns]

    # Split numeric and categorical variables.
    # Free-text columns are discarded to avoid thousands of rare categories.
    numeric_columns = []
    categorical_candidates = []
    for col in candidates:
        is_numeric = pd.api.types.is_numeric_dtype(train[col]) or _numeric_like_series(train[col])
        if is_numeric:
            numeric_columns.append(col)
        elif col in force_categorical_columns or not _looks_like_free_text(train[col]):
            categorical_candidates.append(col)

    # For each categorical variable, keep at most the 20 most frequent categories.
    categorical_levels = {}
    for col in categorical_candidates:
        vc = train[col].astype("string").fillna("__MISSING__").value_counts(dropna=False)
        levels = [str(level) for level in vc.head(20).index.tolist()]
        if train[col].isna().any() and "__MISSING__" not in levels:
            levels.append("__MISSING__")
        if len(levels) >= 2:
            categorical_levels[col] = levels

    # For numeric variables, store the median for imputation and mean/std for scaling.
    medians = {}
    means = {}
    stds = {}
    for col in numeric_columns:
        series = pd.to_numeric(train[col], errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        medians[col] = median
        means[col] = mean
        stds[col] = std if std > 1e-8 else 1.0

    return Preprocessor(
        numeric_columns=numeric_columns,
        categorical_levels=categorical_levels,
        medians=medians,
        means=means,
        stds=stds,
        max_missing_ratio=max_missing_ratio,
        excluded_high_missing_columns=excluded_high_missing_columns,
        missing_ratios=missing_ratios,
        excluded_leakage_columns=sorted(set(excluded_leakage_columns)),
    )


def transform_features(df: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    """Apply the trained preprocessing recipe to any split."""
    blocks = []

    # Numeric variables: impute with the train median, scale, and add a missingness flag.
    for col in preprocessor.numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        missing = series.isna().astype(float).to_numpy().reshape(-1, 1)
        filled = series.fillna(preprocessor.medians[col])
        scaled = (filled - preprocessor.means[col]) / preprocessor.stds[col]

        blocks.append(scaled.to_numpy().reshape(-1, 1))
        blocks.append(missing)

    # Categorical variables: one-hot encode with the levels saved during training.
    for col, levels in preprocessor.categorical_levels.items():
        series = df[col].astype("string").fillna("__MISSING__")
        for level in levels:
            blocks.append((series == level).astype(float).to_numpy().reshape(-1, 1))

    if not blocks:
        raise ValueError("Could not build predictive features.")
    return np.hstack(blocks).astype(float)

def _is_sofa_or_availability_derived_column(col: str) -> bool:
    """Detect helper variables from SOFA calculation or data availability."""
    return (
        col in SOFA_DERIVED_COLUMNS
        or col.endswith(SOFA_DERIVED_SUFFIXES)
        or "_pre_retorn_critics_" in col
        or "_pre_critical_return_" in col
        or col.startswith("sofa_")
        or col.startswith("delta_sofa_")
    )


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute binary-classification metrics at a fixed threshold."""
    # Convert probabilities to 0/1 with the selected threshold.
    y_pred = (y_score >= threshold).astype(int)

    # Basic confusion matrix.
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)

    prevalence = _safe_div(y_true.sum(), n)
    auprc_value = auprc(y_true, y_score)
    auprc_lift = (
        float(auprc_value / prevalence)
        if prevalence > 0 and math.isfinite(float(auprc_value))
        else float("nan")
    )

    return {
        "n": int(n),
        "positives": int(y_true.sum()),
        "prevalence": prevalence,
        "auroc": auroc(y_true, y_score),
        "auprc": auprc_value,
        "auprc_lift": auprc_lift,
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, n),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calculate_episode_metrics(
    df_split: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Aggregate prediction by episode: any true positive day versus any alerted day."""
    tmp = df_split[[ID_COL, TARGET]].copy()
    tmp["score"] = y_score
    episode = tmp.groupby(ID_COL).agg(
        y_true=(TARGET, "max"),
        y_score=("score", "max"),
    )
    return calculate_metrics(
        episode["y_true"].astype(int).to_numpy(),
        episode["y_score"].astype(float).to_numpy(),
        threshold,
    )


def calculate_top_risk_capture(
    df_split: pd.DataFrame,
    y_score: np.ndarray,
    level: str,
    percentages: tuple[float, ...] = (1, 2, 5, 10, 20),
) -> list[dict[str, float | int | str]]:
    """Summarize how many positives are captured by reviewing the highest-risk percent."""
    tmp = df_split[[ID_COL, TARGET]].copy()
    tmp["score"] = y_score

    # Top risk can be computed by day or after aggregating by episode.
    if level == TOP_RISK_LEVEL_EPISODE:
        tmp = (
            tmp.groupby(ID_COL, as_index=False)
            .agg(
                y_true=(TARGET, "max"),
                score=("score", "max"),
            )
        )
    elif level == TOP_RISK_LEVEL_DAY:
        tmp = tmp.rename(columns={TARGET: "y_true"})
    else:
        raise ValueError(f"level must be '{TOP_RISK_LEVEL_DAY}' or '{TOP_RISK_LEVEL_EPISODE}'")

    tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)
    n_total = len(tmp)
    n_positives = int(tmp["y_true"].sum())
    rows: list[dict[str, float | int | str]] = []

    # For each percentage, count positives captured by reviewing the riskiest rows.
    for pct in percentages:
        n_reviewed = max(1, int(math.ceil(n_total * pct / 100)))
        selected = tmp.head(n_reviewed)
        captured_positives = int(selected["y_true"].sum())
        rows.append(
            {
                "level": level,
                "top_percent": pct,
                "n_reviewed": n_reviewed,
                "captured_positives": captured_positives,
                "total_positives": n_positives,
                "sensitivity": _safe_div(captured_positives, n_positives),
                "ppv": _safe_div(captured_positives, n_reviewed),
            }
        )

    return rows


def select_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Select the threshold that maximizes sensitivity + specificity - 1."""
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 199)))
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_score, float(threshold))
        score = metrics["sensitivity"] + metrics["specificity"] - 1
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def select_minimum_sensitivity_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitivity: float,
) -> float:
    """Select the most specific threshold that reaches the requested sensitivity."""
    # Find the threshold with the requested sensitivity and the best possible specificity.
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 199)))
    best_threshold = None
    best_specificity = -1.0

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_score, float(threshold))
        if metrics["sensitivity"] >= sensitivity and metrics["specificity"] > best_specificity:
            best_threshold = float(threshold)
            best_specificity = metrics["specificity"]

    if best_threshold is None:
        return select_youden_threshold(y_true, y_score)

    return best_threshold


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC from ranks without requiring scikit-learn."""
    # AUROC computed from ranks: 1.0 perfect, 0.5 random.
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg_rank = ranks[order[start:end]].mean()
            ranks[order[start:end]] = avg_rank
        start = end

    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve."""
    # AUPRC summarizes precision-recall and is useful when positives are rare.
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def summarize_cohort(df_model: pd.DataFrame) -> dict[str, int | float]:
    """Count rows, episodes, patients, and positives for a modeling cohort."""
    summary = {
        "n_rows": int(len(df_model)),
        "n_episodes": int(df_model[ID_COL].nunique()),
        "n_positive_rows": int(df_model[TARGET].sum()),
        "pct_positive_rows": round(100 * float(df_model[TARGET].mean()), 3),
        "n_positive_episodes": int(df_model.loc[df_model[TARGET] == 1, ID_COL].nunique()),
    }
    if PATIENT_COL in df_model.columns:
        summary["n_patients"] = int(df_model[PATIENT_COL].nunique())
        summary["n_positive_patients"] = int(
            df_model.loc[df_model[TARGET] == 1, PATIENT_COL].nunique()
        )
    return summary


def summarize_splits(df_model: pd.DataFrame) -> dict[str, dict[str, int | float]]:
    """Summarize each named split in a modeling dataset."""
    summary: dict[str, dict[str, int | float]] = {}
    for split, group in df_model.groupby("split"):
        summary[split] = summarize_cohort(group)
    return summary


def _numeric_like_series(series: pd.Series) -> bool:
    """Return True when a text-like column is almost always numeric."""
    if series.dtype == "object" or pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(500)
        if sample.empty:
            return False
        converted = pd.to_numeric(sample, errors="coerce")
        return converted.notna().mean() > 0.95
    return False


def _looks_like_free_text(series: pd.Series) -> bool:
    # If it has many categories or long text values, one-hot encoding is likely not useful.
    sample = series.dropna().astype(str).head(1000)
    if sample.empty:
        return False
    too_many_categories = sample.nunique() > 50
    long_text = sample.str.len().median() > 40
    return bool(too_many_categories or long_text)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0







