from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.classic_models_24h import (
    CLASSIC_MODEL_FILES,
    _compute_and_save_classic_model_shap,
    create_chronological_episode_split,
    filter_real_from_start_date,
    generate_classic_model_figures_from_outputs,
    prepare_classic_model_data,
)
from src.config import (
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    OUTPUTS_DIR,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SHAP_SAMPLE_N_DEFAULT,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
)
from src.data_loading import load_sepsis_model_with_sofa
from src.deep_learning_figures import generate_temporal_model_figures
from src.deep_learning_shap_24h import calculate_deep_learning_shap
from src.figure_style import PALETTE, apply_report_style, save_report_figure
from src.output_contracts import (
    LEVEL_EPISODE,
    LEVEL_NEXT_DAY,
    deep_metrics_filename,
    deep_output_paths,
)
from src.predictive_model_24h import ID_COL, PATIENT_COL, TARGET, calculate_metrics, transform_features
from src.real_policies import REAL_ALL_2026, REAL_START_DATE_DEFAULT


OUTPUT_BASE = OUTPUTS_DIR / "previous_day_only_ablation"
TABLES_DIR = OUTPUT_BASE / "tables"
FIGURES_DIR = OUTPUT_BASE / "figures"
FIGURES_COMPARISON_DIR = FIGURES_DIR / "comparison"
INDEX_FILE = OUTPUT_BASE / "previous_day_only_figures_index.json"

FULL_COMPARISON_PATH = OUTPUTS_DIR / "post_optuna_final" / "tables" / "01_final_model_comparison.csv"
CLASSIC_PREVIOUS_DAY_PATH = (
    OUTPUT_BASE / "classic_lightgbm_optuna" / CLASSIC_MODEL_FILES["results"]
)
CLASSIC_PREVIOUS_DAY_OUTPUT_DIR = OUTPUT_BASE / "classic_lightgbm_optuna"
DEEP_OUTPUT_PREFIX = "transformer_previous_day_only_24h"
REAL_POLICY = REAL_ALL_2026
REAL_START_DATE = REAL_START_DATE_DEFAULT
EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
SHAP_SAMPLE_N = SHAP_SAMPLE_N_DEFAULT
DEEP_PREVIOUS_DAY_PATH = (
    OUTPUT_BASE / "deep_transformer_optuna" / deep_metrics_filename(DEEP_OUTPUT_PREFIX)
)
DEEP_PREVIOUS_DAY_OUTPUT_DIR = OUTPUT_BASE / "deep_transformer_optuna"
FULL_CLASSIC_SHAP_PATH = (
    OUTPUTS_DIR
    / "models_classics_24h"
    / "optuna_best"
    / "real_all_2026"
    / "lightgbm"
    / "shap"
    / "lightgbm_shap_variable_importance.csv"
)
PREVIOUS_CLASSIC_SHAP_PATH = (
    OUTPUT_BASE
    / "classic_lightgbm_optuna"
    / "shap"
    / "lightgbm_shap_variable_importance.csv"
)
FULL_DEEP_SHAP_PATH = (
    OUTPUTS_DIR
    / "deep_learning_24h"
    / "optuna_best"
    / "real_all_2026"
    / "transformer"
    / "shap"
    / "transformer_24h_shap_variable_importance.csv"
)
PREVIOUS_DEEP_SHAP_PATH = (
    OUTPUT_BASE
    / "deep_transformer_optuna"
    / "shap"
    / f"{DEEP_OUTPUT_PREFIX}_shap_variable_importance.csv"
)


def main() -> None:
    """Create tables and figures for the previous-day-only ablation."""
    _prepare_output_directories()

    # Script 12 is intentionally safe to rerun: it reads saved models/results and only
    # regenerates SHAP outputs or figures when needed.
    ensure_classic_predictions_include_all_splits()
    ensure_previous_day_only_interpretability_outputs()

    comparison = build_previous_day_only_comparison()
    importance = build_feature_importance_comparison()

    tables = save_comparison_tables(comparison, importance)
    figures = generate_ablation_figures(comparison, importance)
    index_path = write_figures_index(tables, figures)

    print("Previous-day-only comparison table:", tables["comparison"])
    print("Previous-day-only feature comparison table:", tables["feature_importance_comparison"])
    print("Previous-day-only figures index:", index_path)


def _prepare_output_directories() -> None:
    """Create the output folders used by this figure-generation step."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def save_comparison_tables(
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, str]:
    """Save performance and feature-importance comparison tables."""
    table_path = TABLES_DIR / "01_previous_day_only_comparison.csv"
    importance_path = TABLES_DIR / "02_previous_day_only_feature_importance_comparison.csv"
    comparison.to_csv(table_path, index=False)
    importance.to_csv(importance_path, index=False)
    return {
        "comparison": str(table_path),
        "feature_importance_comparison": str(importance_path),
    }


def generate_ablation_figures(
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, object]:
    """Generate all figures for the previous-day-only ablation analysis."""
    return {
        "classic_lightgbm_previous_day_only": generate_classic_model_figures_from_outputs(
            CLASSIC_PREVIOUS_DAY_OUTPUT_DIR,
        ),
        "classic_lightgbm_previous_day_only_diagnostic": generate_classic_diagnostic_figures(),
        "deep_transformer_previous_day_only": generate_temporal_model_figures(
            DEEP_OUTPUT_PREFIX,
            output_dir=DEEP_PREVIOUS_DAY_OUTPUT_DIR,
        ),
        "full_vs_previous_day_auprc": plot_full_vs_previous_day_auprc(comparison),
        "previous_day_only_metrics": plot_previous_day_only_metrics(comparison),
        "lightgbm_feature_importance_comparison": plot_feature_importance_comparison(
            importance,
            model="LightGBM",
            path=FIGURES_COMPARISON_DIR / "03_lightgbm_feature_importance_comparison.png",
        ),
        "transformer_feature_importance_comparison": plot_feature_importance_comparison(
            importance,
            model="Transformer",
            path=FIGURES_COMPARISON_DIR / "04_transformer_feature_importance_comparison.png",
        ),
    }


def generate_classic_diagnostic_figures() -> dict[str, str]:
    """Generate deep-style diagnostic figures for the classic LightGBM run."""
    predictions = pd.read_csv(CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / CLASSIC_MODEL_FILES["predictions"])
    results = pd.read_csv(CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / CLASSIC_MODEL_FILES["results"])
    predictions = predictions.loc[predictions["model"].eq("lightgbm")].copy()
    result_row = results.loc[results["model"].eq("lightgbm")].iloc[0]
    figures_dir = CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "next_day_metrics_comparison": plot_classic_metrics_by_split(
            result_row,
            path=figures_dir / "03_next_day_metrics_comparison.png",
        ),
        "auprc_lift": plot_classic_auprc_lift(
            result_row,
            path=figures_dir / "04_auprc_lift_and_prevalence.png",
        ),
        "roc_next_day": plot_classic_curve(
            predictions,
            curve="roc",
            path=figures_dir / "05_next_day_roc.png",
        ),
        "precision_recall_next_day": plot_classic_curve(
            predictions,
            curve="pr",
            path=figures_dir / "06_next_day_precision_recall.png",
        ),
        "real_next_day_risk_distribution": plot_classic_real_risk_distribution(
            predictions,
            threshold=float(result_row["threshold_valid"]),
            path=figures_dir / "07_real_next_day_risk_distribution.png",
        ),
    }


def plot_classic_metrics_by_split(row: pd.Series, path: Path) -> str:
    """Plot the main patient-day metrics for the classic test and real splits."""
    data = classic_metrics_table_from_predictions(float(row["threshold_valid"]))
    metrics = ["AUROC", "AUPRC", "Sensitivity", "PPV", "F1"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"], PALETTE["vermilion"]]
    x = np.arange(len(data))
    width = 0.15
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for index, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + (index - 2) * width, data[metric], width, label=metric, color=color)
        _annotate(ax, bars)
    ax.set_title("LightGBM: metrics by split (next-day level)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(data["split"])
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    return save_report_figure(fig, path)


def plot_classic_auprc_lift(row: pd.Series, path: Path) -> str:
    """Plot AUPRC against prevalence and AUPRC lift for the classic model."""
    data = classic_metrics_table_from_predictions(float(row["threshold_valid"]))
    labels = data["split"].tolist()
    x = np.arange(len(labels))
    width = 0.34
    apply_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    bars_auprc = axes[0].bar(x - width / 2, data["AUPRC"], width, color=PALETTE["orange"], label="AUPRC")
    bars_prev = axes[0].bar(x + width / 2, data["Prevalence"], width, color=PALETTE["teal"], label="Prevalence")
    _annotate(axes[0], bars_auprc)
    _annotate(axes[0], bars_prev)
    axes[0].set_title("AUPRC and prevalence")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(loc="upper left")

    bars_lift = axes[1].bar(x, data["AUPRC lift"], color=PALETTE["purple"])
    _annotate_lift(axes[1], bars_lift)
    axes[1].set_title("AUPRC lift")
    axes[1].set_ylim(0, max(float(data["AUPRC lift"].max()) * 1.18, 1.0))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    fig.suptitle("LightGBM: AUPRC compared with prevalence", fontsize=14, fontweight="bold")
    return save_report_figure(fig, path)


def plot_classic_curve(predictions: pd.DataFrame, curve: str, path: Path) -> str:
    """Plot ROC or precision-recall curves from saved classic predictions."""
    apply_report_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    split_labels = {
        "train": "Train",
        "valid": "Validation",
        "test": "Test",
        "real": "Real 2026",
    }
    split_colors = {
        "train": PALETTE["blue"],
        "valid": PALETTE["teal"],
        "test": PALETTE["orange"],
        "real": PALETTE["gold"],
    }
    prevalences = []
    for split, label in split_labels.items():
        df_split = predictions.loc[predictions["split"].eq(split)].copy()
        if df_split.empty or df_split["y_true"].nunique() < 2:
            continue
        y = df_split["y_true"].astype(int).to_numpy()
        score = df_split["score"].astype(float).to_numpy()
        prevalences.append(float(y.mean()))
        if curve == "roc":
            x_values, y_values, _ = roc_curve(y, score)
            metric = roc_auc_score(y, score)
            ax.plot(x_values, y_values, linewidth=2.2, color=split_colors[split], label=f"{label} ({metric:.3f})")
        elif curve == "pr":
            precision, recall, _ = precision_recall_curve(y, score)
            metric = average_precision_score(y, score)
            ax.plot(recall, precision, linewidth=2.2, color=split_colors[split], label=f"{label} ({metric:.3f})")
        else:
            raise ValueError("curve must be 'roc' or 'pr'.")

    if curve == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["muted"], linewidth=1)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title("LightGBM: ROC curves (next-day level)")
        ax.legend(loc="lower right")
    else:
        if prevalences:
            ax.axhline(float(np.mean(prevalences)), linestyle="--", color=PALETTE["muted"], linewidth=1, label="Mean prevalence")
        ax.set_xlabel("Sensitivity / recall")
        ax.set_ylabel("Precision / PPV")
        ax.set_title("LightGBM: precision-recall curves (next-day level)")
        ax.legend(loc="upper right")
    return save_report_figure(fig, path)


def plot_classic_real_risk_distribution(predictions: pd.DataFrame, threshold: float, path: Path) -> str:
    """Plot real-cohort risk distributions by next-day sepsis status."""
    real = predictions.loc[predictions["split"].eq("real")].copy()
    y = real["y_true"].astype(int).to_numpy()
    score = real["score"].astype(float).to_numpy()
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.hist(score[y == 0], bins=50, alpha=0.72, density=True, color=PALETTE["sky_blue"], label="No sepsis")
    ax.hist(score[y == 1], bins=30, alpha=0.72, density=True, color=PALETTE["yellow"], label="Sepsis")
    ax.axvline(threshold, color=PALETTE["ink"], linestyle="--", linewidth=1.5, label=f"threshold {threshold:.3f}")
    ax.set_title("Risk distribution: LightGBM - real - next-day level")
    ax.set_xlabel("Estimated risk")
    ax.set_ylabel("Density")
    ax.legend()
    return save_report_figure(fig, path)


def write_figures_index(
    tables: dict[str, str],
    figures: dict[str, object],
) -> Path:
    """Write an index linking every input, table, and figure produced here."""
    index = {
        "tables": tables,
        "figures": figures,
        "inputs": {
            "full_optuna": str(FULL_COMPARISON_PATH),
            "previous_day_lightgbm": str(CLASSIC_PREVIOUS_DAY_PATH),
            "previous_day_transformer": str(DEEP_PREVIOUS_DAY_PATH),
            "full_lightgbm_shap": str(FULL_CLASSIC_SHAP_PATH),
            "previous_day_lightgbm_shap": str(PREVIOUS_CLASSIC_SHAP_PATH),
            "full_transformer_shap": str(FULL_DEEP_SHAP_PATH),
            "previous_day_transformer_shap": str(PREVIOUS_DEEP_SHAP_PATH),
        },
    }
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_FILE


def ensure_classic_predictions_include_all_splits() -> None:
    """Backfill train/validation predictions for older classic ablation outputs."""
    predictions_path = CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / CLASSIC_MODEL_FILES["predictions"]
    _require_file(predictions_path, "Run scripts/11_previous_day_only_ablation.py first.")
    predictions = pd.read_csv(predictions_path)
    expected = {"train", "valid", "test", "real"}
    present = set(predictions.loc[predictions["model"].eq("lightgbm"), "split"].dropna().astype(str))
    if expected.issubset(present):
        return

    print("Completing classic LightGBM predictions with train and validation splits...")
    df_sofa = _load_modeling_dataset()
    full_predictions = _build_classic_predictions_from_saved_model(df_sofa)
    full_predictions.to_csv(predictions_path, index=False)


def _build_classic_predictions_from_saved_model(df_sofa: pd.DataFrame) -> pd.DataFrame:
    """Rebuild all-split predictions from the saved previous-day-only LightGBM model."""
    pickle_path = CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / CLASSIC_MODEL_FILES["model_pickle"]
    _require_file(pickle_path, "Run scripts/11_previous_day_only_ablation.py first.")
    with open(pickle_path, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    model = dict(bundle["models"])["lightgbm"]
    df_model = _prepare_previous_day_classic_frame(df_sofa)
    frames = []
    for split in ("train", "valid", "test", "real"):
        df_split = df_model.loc[df_model["split"].eq(split)].copy()
        if df_split.empty:
            continue
        x_split = transform_features(df_split, preprocessor)
        score = model.predict_proba(x_split)[:, 1]
        frames.append(_classic_prediction_frame(df_split, score, split))
    return pd.concat(frames, ignore_index=True)


def _prepare_previous_day_classic_frame(df_sofa: pd.DataFrame) -> pd.DataFrame:
    """Prepare the previous-day-only classic model frame with the saved split policy."""
    df_model, _ = prepare_classic_model_data(
        df_sofa,
        exclude_microbiology=False,
        include_temporal_features=False,
    )
    split_map, _ = create_chronological_episode_split(
        df_model,
        proportions=(0.70, 0.15, 0.15),
        split_unit="patient",
        real_start_date=REAL_START_DATE,
        real_overlap_policy=REAL_POLICY,
    )
    df_model["split"] = df_model[ID_COL].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()
    df_model, _ = filter_real_from_start_date(
        df_model,
        real_start_date=REAL_START_DATE,
        enabled=True,
    )
    return df_model


def _classic_prediction_frame(df_split: pd.DataFrame, score: np.ndarray, split: str) -> pd.DataFrame:
    """Build the standard classic prediction table for one split."""
    cols = [ID_COL, TARGET]
    if PATIENT_COL in df_split.columns:
        cols.insert(1, PATIENT_COL)
    if "data_index" in df_split.columns:
        cols.append("data_index")
    pred = df_split[cols].copy()
    pred.insert(0, "model", "lightgbm")
    pred.insert(1, "label", "LightGBM")
    pred["split"] = split
    pred["y_true"] = df_split[TARGET].astype(int).to_numpy()
    pred["score"] = score
    return pred


def classic_metrics_table_from_predictions(threshold: float) -> pd.DataFrame:
    """Compute classic metrics for every prediction split saved on disk."""
    predictions = pd.read_csv(CLASSIC_PREVIOUS_DAY_OUTPUT_DIR / CLASSIC_MODEL_FILES["predictions"])
    predictions = predictions.loc[predictions["model"].eq("lightgbm")].copy()
    rows = []
    for split in ("train", "valid", "test", "real"):
        df_split = predictions.loc[predictions["split"].eq(split)].copy()
        if df_split.empty:
            continue
        metrics = calculate_metrics(
            df_split["y_true"].astype(int).to_numpy(),
            df_split["score"].astype(float).to_numpy(),
            threshold,
        )
        rows.append(
            {
                "split": _split_label(split),
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "Sensitivity": metrics["sensitivity"],
                "PPV": metrics["ppv"],
                "F1": metrics["f1"],
                "Prevalence": metrics["prevalence"],
                "AUPRC lift": metrics["auprc_lift"],
            }
        )
    return pd.DataFrame(rows)


def ensure_previous_day_only_interpretability_outputs() -> None:
    """Create SHAP tables/figures from saved ablation models when missing."""
    if PREVIOUS_CLASSIC_SHAP_PATH.exists() and PREVIOUS_DEEP_SHAP_PATH.exists():
        return
    df_sofa = _load_modeling_dataset()
    if not PREVIOUS_CLASSIC_SHAP_PATH.exists():
        print("Calculating previous-day-only LightGBM SHAP outputs...")
        _calculate_previous_day_lightgbm_shap(df_sofa)
    if not PREVIOUS_DEEP_SHAP_PATH.exists():
        print("Calculating previous-day-only Transformer SHAP outputs...")
        _calculate_previous_day_transformer_shap(df_sofa)


def _load_modeling_dataset() -> pd.DataFrame:
    """Load the same SOFA-labelled dataset used by the ablation script."""
    return load_sepsis_model_with_sofa(
        episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
        lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
        vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
        max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
    )


def _calculate_previous_day_lightgbm_shap(df_sofa: pd.DataFrame) -> dict[str, str]:
    """Calculate SHAP for the saved previous-day-only LightGBM run."""
    pickle_path = OUTPUT_BASE / "classic_lightgbm_optuna" / CLASSIC_MODEL_FILES["model_pickle"]
    _require_file(pickle_path, "Run scripts/11_previous_day_only_ablation.py first.")
    with open(pickle_path, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    model = dict(bundle["models"])["lightgbm"]
    df_model, _ = prepare_classic_model_data(
        df_sofa,
        exclude_microbiology=False,
        include_temporal_features=False,
    )
    split_map, _ = create_chronological_episode_split(
        df_model,
        proportions=(0.70, 0.15, 0.15),
        split_unit="patient",
        real_start_date=REAL_START_DATE,
        real_overlap_policy=REAL_POLICY,
    )
    df_model["split"] = df_model[ID_COL].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()
    df_model, _ = filter_real_from_start_date(
        df_model,
        real_start_date=REAL_START_DATE,
        enabled=True,
    )
    train = df_model.loc[df_model["split"].eq("train")].copy()
    valid = df_model.loc[df_model["split"].eq("valid")].copy()
    test = df_model.loc[df_model["split"].eq("test")].copy()
    real = df_model.loc[df_model["split"].eq("real")].copy()

    x_train = transform_features(train, preprocessor)
    x_valid = transform_features(valid, preprocessor)
    x_test = transform_features(test, preprocessor)
    x_real = transform_features(real, preprocessor) if not real.empty else None
    return _compute_and_save_classic_model_shap(
        model=model,
        model_key="lightgbm",
        label="LightGBM",
        feature_names=list(preprocessor.feature_names),
        x_train_valid=np.vstack([x_train, x_valid]),
        x_test=x_test,
        test=test,
        x_real=x_real,
        real=real,
        output_dir=OUTPUT_BASE / "classic_lightgbm_optuna",
        sample_n=SHAP_SAMPLE_N,
        preferred_split="real",
        real_overlap_policy=REAL_POLICY,
        seed=42,
    )


def _calculate_previous_day_transformer_shap(df_sofa: pd.DataFrame) -> dict[str, str]:
    """Calculate SHAP for the saved previous-day-only Transformer run."""
    deep_paths = deep_output_paths(DEEP_PREVIOUS_DAY_OUTPUT_DIR, DEEP_OUTPUT_PREFIX)
    candidate = {
        "model_key": "transformer",
        "prefix": DEEP_OUTPUT_PREFIX,
        "policy": REAL_POLICY,
        "output_dir": DEEP_PREVIOUS_DAY_OUTPUT_DIR,
        "metrics_path": DEEP_PREVIOUS_DAY_PATH,
        "model_path": deep_paths["model"],
        "summary_path": deep_paths["summary"],
        "valid_auprc": float("nan"),
        "valid_auroc": float("nan"),
    }
    return calculate_deep_learning_shap(
        candidate=candidate,
        df_sofa=df_sofa,
        output_dir=DEEP_PREVIOUS_DAY_OUTPUT_DIR / "shap",
    )


def build_previous_day_only_comparison() -> pd.DataFrame:
    """Combine full-history Optuna results with previous-day-only results."""
    _require_file(FULL_COMPARISON_PATH, "Run scripts/09_post_optuna_final_figures.py first.")
    _require_file(CLASSIC_PREVIOUS_DAY_PATH, "Run scripts/11_previous_day_only_ablation.py first.")
    _require_file(DEEP_PREVIOUS_DAY_PATH, "Run scripts/11_previous_day_only_ablation.py first.")

    full = pd.read_csv(FULL_COMPARISON_PATH)
    classic_previous = pd.read_csv(CLASSIC_PREVIOUS_DAY_PATH)
    deep_previous = pd.read_csv(DEEP_PREVIOUS_DAY_PATH)

    rows: list[dict[str, object]] = []
    rows.extend(_full_optuna_rows(full))
    rows.append(_classic_previous_day_row(classic_previous))
    rows.append(_deep_previous_day_row(deep_previous))
    comparison = pd.DataFrame(rows)
    comparison["setting_order"] = comparison["setting"].map({"full_history": 0, "previous_day_only": 1})
    comparison["model_order"] = comparison["model"].map({"LightGBM": 0, "Transformer": 1})
    return comparison.sort_values(["model_order", "setting_order"]).drop(
        columns=["model_order", "setting_order"]
    )


def build_feature_importance_comparison() -> pd.DataFrame:
    """Combine full-history and previous-day-only SHAP importance tables."""
    for path in (
        FULL_CLASSIC_SHAP_PATH,
        PREVIOUS_CLASSIC_SHAP_PATH,
        FULL_DEEP_SHAP_PATH,
        PREVIOUS_DEEP_SHAP_PATH,
    ):
        _require_file(path, "Run scripts/11_previous_day_only_ablation.py and the main SHAP scripts first.")

    rows = []
    rows.extend(_importance_rows(FULL_CLASSIC_SHAP_PATH, "LightGBM", "full_history"))
    rows.extend(_importance_rows(PREVIOUS_CLASSIC_SHAP_PATH, "LightGBM", "previous_day_only"))
    rows.extend(_importance_rows(FULL_DEEP_SHAP_PATH, "Transformer", "full_history"))
    rows.extend(_importance_rows(PREVIOUS_DEEP_SHAP_PATH, "Transformer", "previous_day_only"))
    return pd.DataFrame(rows)


def _importance_rows(path: Path, model: str, setting: str) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    if "shap_importance_pct" not in table.columns:
        raise ValueError(f"Missing shap_importance_pct in: {path}")
    out = []
    for _, row in table.iterrows():
        out.append(
            {
                "model": model,
                "setting": setting,
                "variable": str(row["variable"]),
                "rank": int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else None,
                "mean_abs_shap": _float(row["mean_abs_shap"]),
                "shap_importance_pct": _float(row["shap_importance_pct"]),
                "source": str(path),
            }
        )
    return out


def _full_optuna_rows(full: pd.DataFrame) -> list[dict[str, object]]:
    selected = full.loc[
        (full["run"].eq("optuna"))
        & (
            ((full["family"].eq("classic")) & full["model"].eq("lightgbm"))
            | ((full["family"].eq("deep_learning")) & full["model"].eq("transformer"))
        )
    ].copy()
    if selected.empty:
        raise ValueError(f"Could not find full-history Optuna rows in: {FULL_COMPARISON_PATH}")

    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "family": row["family"],
                "model": str(row["label"]),
                "setting": "full_history",
                "real_auroc": _float(row["real_auroc"]),
                "real_auprc": _float(row["real_auprc"]),
                "real_sensitivity": _float(row["real_sensitivity"]),
                "real_ppv": _float(row["real_ppv"]),
                "real_episode_auroc": _float(row["real_episode_auroc"]),
                "real_episode_auprc": _float(row["real_episode_auprc"]),
                "real_episode_sensitivity": _float(row["real_episode_sensitivity"]),
                "real_episode_ppv": _float(row["real_episode_ppv"]),
                "source": row["source"],
            }
        )
    return rows


def _classic_previous_day_row(results: pd.DataFrame) -> dict[str, object]:
    row = results.loc[results["model"].eq("lightgbm")].iloc[0]
    return {
        "family": "classic",
        "model": "LightGBM",
        "setting": "previous_day_only",
        "real_auroc": _float(row["real_auroc"]),
        "real_auprc": _float(row["real_auprc"]),
        "real_sensitivity": _float(row["real_sensitivity"]),
        "real_ppv": _float(row["real_ppv"]),
        "real_episode_auroc": _float(row["real_episode_auroc"]),
        "real_episode_auprc": _float(row["real_episode_auprc"]),
        "real_episode_sensitivity": _float(row["real_episode_sensitivity"]),
        "real_episode_ppv": _float(row["real_episode_ppv"]),
        "source": str(CLASSIC_PREVIOUS_DAY_PATH),
    }


def _deep_previous_day_row(metrics: pd.DataFrame) -> dict[str, object]:
    day = metrics.loc[(metrics["split"].eq("real")) & (metrics["level"].eq(LEVEL_NEXT_DAY))].iloc[0]
    episode = metrics.loc[(metrics["split"].eq("real")) & (metrics["level"].eq(LEVEL_EPISODE))].iloc[0]
    return {
        "family": "deep_learning",
        "model": "Transformer",
        "setting": "previous_day_only",
        "real_auroc": _float(day["auroc"]),
        "real_auprc": _float(day["auprc"]),
        "real_sensitivity": _float(day["sensitivity"]),
        "real_ppv": _float(day["ppv"]),
        "real_episode_auroc": _float(episode["auroc"]),
        "real_episode_auprc": _float(episode["auprc"]),
        "real_episode_sensitivity": _float(episode["sensitivity"]),
        "real_episode_ppv": _float(episode["ppv"]),
        "source": str(DEEP_PREVIOUS_DAY_PATH),
    }


def plot_full_vs_previous_day_auprc(comparison: pd.DataFrame) -> str:
    """Plot patient-day and episode AUPRC for full-history and one-day runs."""
    apply_report_style()
    models = ["LightGBM", "Transformer"]
    settings = [
        ("full_history", "Full history", PALETTE["blue"]),
        ("previous_day_only", "Previous day only", PALETTE["orange"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(models))
    width = 0.34
    for ax, metric, title in zip(
        axes,
        ["real_auprc", "real_episode_auprc"],
        ["Patient-day AUPRC", "Episode-level AUPRC"],
    ):
        for offset, (setting_key, setting_label, color) in zip([-width / 2, width / 2], settings):
            values = [
                float(
                    comparison.loc[
                        comparison["model"].eq(model) & comparison["setting"].eq(setting_key),
                        metric,
                    ].iloc[0]
                )
                for model in models
            ]
            bars = ax.bar(x + offset, values, width, label=setting_label, color=color)
            _annotate(ax, bars)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 0.62)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUPRC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.08))
    return save_report_figure(fig, FIGURES_DIR / "01_full_vs_previous_day_auprc.png")


def plot_previous_day_only_metrics(comparison: pd.DataFrame) -> str:
    """Plot the main patient-day metrics for the previous-day-only runs."""
    previous = comparison.loc[comparison["setting"].eq("previous_day_only")].copy()
    metrics = ["real_auroc", "real_auprc", "real_sensitivity", "real_ppv"]
    metric_labels = ["AUROC", "AUPRC", "Sensitivity", "PPV"]
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    x = np.arange(len(metric_labels))
    width = 0.34
    for offset, (_, row) in zip([-width / 2, width / 2], previous.iterrows()):
        bars = ax.bar(
            x + offset,
            [row[metric] for metric in metrics],
            width,
            label=row["model"],
            color=PALETTE["blue"] if row["model"] == "LightGBM" else PALETTE["teal"],
        )
        _annotate(ax, bars)
    ax.set_title("Previous-day-only real-world performance", pad=14)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.10))
    return save_report_figure(fig, FIGURES_DIR / "02_previous_day_only_metrics.png")


def plot_feature_importance_comparison(
    importance: pd.DataFrame,
    model: str,
    path: Path,
    top_n: int = 12,
) -> str:
    """Plot full-history vs previous-day-only SHAP variable importance."""
    model_data = importance.loc[importance["model"].eq(model)].copy()
    if model_data.empty:
        raise ValueError(f"No importance rows found for model: {model}")
    top_variables = (
        model_data.sort_values("shap_importance_pct", ascending=False)
        .groupby("variable", as_index=False)["shap_importance_pct"]
        .max()
        .sort_values("shap_importance_pct", ascending=False)
        .head(top_n)["variable"]
        .tolist()
    )
    wide = (
        model_data.loc[model_data["variable"].isin(top_variables)]
        .pivot_table(
            index="variable",
            columns="setting",
            values="shap_importance_pct",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(top_variables)
    )
    for col in ("full_history", "previous_day_only"):
        if col not in wide.columns:
            wide[col] = 0.0

    y = np.arange(len(wide.index))
    height = 0.36
    apply_report_style()
    fig, ax = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)
    bars_full = ax.barh(
        y - height / 2,
        wide["full_history"].to_numpy(),
        height,
        label="Full history",
        color=PALETTE["blue"],
    )
    bars_previous = ax.barh(
        y + height / 2,
        wide["previous_day_only"].to_numpy(),
        height,
        label="Previous day only",
        color=PALETTE["orange"],
    )
    ax.set_title(f"{model} SHAP variable importance comparison")
    ax.set_xlabel("SHAP importance (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    _annotate_h(ax, bars_full)
    _annotate_h(ax, bars_previous)
    return save_report_figure(fig, path)


def _annotate(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.012,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=PALETTE["ink"],
        )


def _annotate_h(ax: plt.Axes, bars) -> None:
    max_width = max([float(bar.get_width()) for bar in bars] + [1.0])
    for bar in bars:
        width = float(bar.get_width())
        if not np.isfinite(width) or width <= 0:
            continue
        ax.text(
            width + max_width * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            ha="left",
            va="center",
            fontsize=8,
            color=PALETTE["ink"],
        )


def _annotate_lift(ax: plt.Axes, bars) -> None:
    """Annotate lift bars with an x suffix."""
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(height * 0.02, 0.05),
            f"{height:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=PALETTE["ink"],
        )


def _split_label(split: str) -> str:
    """Return display labels consistent with the deep-learning figures."""
    return {
        "train": "Train",
        "valid": "Validation",
        "test": "Test",
        "real": "Real 2026",
    }.get(str(split), str(split))


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}. {hint}")


def _float(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))
