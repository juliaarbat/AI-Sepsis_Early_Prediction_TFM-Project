from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve

from src.config import DEEP_LEARNING_OUTPUTS_DIR, MODELS_CLASSICS_OUTPUTS_DIR, OUTPUTS_DIR
from src.classic_models_24h import CLASSIC_MODEL_FILES
from src.figure_style import PALETTE, save_report_figure
from src.output_contracts import (
    COL_REAL_AUPRC,
    COL_REAL_AUPRC_LIFT,
    COL_REAL_EPISODE_AUPRC,
    COL_REAL_EPISODE_AUPRC_LIFT,
    COL_REAL_EPISODE_AUROC,
    COL_REAL_PPV,
    COL_REAL_SENSITIVITY,
    EXECUTION_BASE,
    EXECUTION_OPTUNA,
    LEVEL_EPISODE,
    LEVEL_NEXT_DAY,
    METRIC_AUPRC,
    MODEL_FAMILY_CLASSIC,
    MODEL_FAMILY_DEEP_LEARNING,
    POST_OPTUNA_COMPARISON_FILE,
    POST_OPTUNA_CV_FILE,
    POST_OPTUNA_DECISION_FILE,
    POST_OPTUNA_INDEX_FILE,
    POST_OPTUNA_README_FILE,
    SPLIT_REAL,
    SPLIT_TEST,
    STANDARD_METRICS,
    TOP_RISK_LEVEL_DAY,
    deep_metrics_filename,
)
from src.output_paths import (
    deep_metrics_path,
    deep_optuna_dirs,
)
from src.plot_utils import annotate_bars, clear_pngs, metric_label, padded_ymax, setup_report_style
from src.real_policies import REAL_ALL_2026, real_policy_labels
from src.progress import log_end, log_start, step


OPTUNA_REAL_POLICY = REAL_ALL_2026
_POLICY_LABELS = real_policy_labels(short=True)
POLICIES = {OPTUNA_REAL_POLICY: _POLICY_LABELS[OPTUNA_REAL_POLICY]}

DEEP_MODELS = {
    "transformer": {"prefix": "transformer_24h", "label": "Transformer"},
    "lstm": {"prefix": "lstm_24h", "label": "LSTM"},
    "rnn": {"prefix": "rnn_24h", "label": "RNN"},
}

OUTPUT_DIR = OUTPUTS_DIR / "post_optuna_final"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
# Remove these folders when regenerating the curated report.
LEGACY_FIGURE_DIRS = (
    FIGURES_DIR / "01_base_optuna_comparison",
    FIGURES_DIR / "01_optuna_effect",
    FIGURES_DIR / "02_test_real_generalization",
    FIGURES_DIR / "03_row_episode_level",
    FIGURES_DIR / "03_final_model_decision",
    FIGURES_DIR / "04_model_decision",
    FIGURES_DIR / "05_robustness_cv",
    FIGURES_DIR / "06_result_diagnostics",
)
COMPARISON_PATH = TABLES_DIR / POST_OPTUNA_COMPARISON_FILE
DECISION_PATH = TABLES_DIR / POST_OPTUNA_DECISION_FILE
CV_PATH = TABLES_DIR / POST_OPTUNA_CV_FILE
CLINICAL_CLASSIFICATION_PATH = TABLES_DIR / "04_final_lightgbm_threshold_classification.csv"
CLINICAL_THRESHOLD_CURVE_PATH = TABLES_DIR / "05_final_lightgbm_threshold_tradeoff.csv"
CLINICAL_TOP_RISK_PATH = TABLES_DIR / "06_final_lightgbm_top_risk_capture.csv"
LEGACY_TABLE_PATHS = (
    TABLES_DIR / "01_base_vs_optuna_comparison.csv",
    TABLES_DIR / "02_model_decision.csv",
)
INDEX_PATH = OUTPUT_DIR / POST_OPTUNA_INDEX_FILE
README_PATH = OUTPUT_DIR / POST_OPTUNA_README_FILE

COLORS = {
    EXECUTION_BASE: PALETTE["blue"],
    EXECUTION_OPTUNA: PALETTE["orange"],
    MODEL_FAMILY_CLASSIC: PALETTE["teal"],
    MODEL_FAMILY_DEEP_LEARNING: PALETTE["purple"],
    SPLIT_TEST: PALETTE["blue"],
    SPLIT_REAL: PALETTE["orange"],
    "neutral": PALETTE["muted"],
}

def main() -> None:
    """Generate final baseline-vs-Optuna tables and figures for the thesis."""
    log_start("final post-Optuna figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (
        TABLES_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_pngs(FIGURES_DIR)
    for directory in (
        *LEGACY_FIGURE_DIRS,
    ):
        _clear_pngs(directory)
        _remove_empty_dir(directory)
    for table_path in LEGACY_TABLE_PATHS:
        if table_path.exists():
            table_path.unlink()

    with step("Load baseline and post-Optuna results", number=1, total=3):
        # Build comparable tables from saved outputs; no model is retrained here.
        comparison = _create_final_comparison()
        if comparison.empty:
            raise FileNotFoundError("No baseline/post-Optuna results were found to summarize.")
        _write_english_csv(comparison, COMPARISON_PATH)

        decision = _create_decision_table(comparison)
        _write_english_csv(decision, DECISION_PATH)

        cv = _load_robustness_cv()
        if not cv.empty:
            _write_english_csv(cv, CV_PATH)

    figures: dict[str, str] = {}
    with step("Generate final figures", number=2, total=3):
        # Keep the main report focused on tuning effect and real-cohort transfer.
        figures["01_optuna_effect_real_auprc"] = _plot_base_vs_optuna(
            comparison,
            metric=METRIC_AUPRC,
            level=TOP_RISK_LEVEL_DAY,
            path=FIGURES_DIR / "01_optuna_effect_real_auprc.png",
        )
        figures["02_post_optuna_test_vs_real_auprc"] = _plot_test_vs_real_optuna(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "02_post_optuna_test_vs_real_auprc.png",
        )
        figures["03_final_model_decision"] = _plot_decision_table(
            decision,
            path=FIGURES_DIR / "03_final_model_decision.png",
        )
        clinical_tables, clinical_figures = _create_clinical_usefulness_outputs()
        figures.update(clinical_figures)

    with step("Write index", number=3, total=3):
        payload = {
            "objective": "Three final, thesis-ready post-Optuna figures plus supporting tables.",
            "comparison_csv": str(COMPARISON_PATH),
            "decision_csv": str(DECISION_PATH),
            "cv_csv": str(CV_PATH) if not cv.empty else None,
            "clinical_usefulness_tables": clinical_tables,
            "tables_dir": str(TABLES_DIR),
            "figures_dir": str(FIGURES_DIR),
            "readme": str(README_PATH),
            "figures": figures,
            "notes": [
                "Figures are intentionally limited to the main story: Optuna effect, test-real generalization, and final model decision.",
                "Robustness CV remains available as a supporting table when present, but is not plotted in the curated figure set.",
                "Clinical-usefulness outputs use saved final LightGBM predictions and do not retrain models.",
            ],
        }
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _write_readme(cv_exists=not cv.empty)

    print("Comparison:", COMPARISON_PATH)
    print("Decision table:", DECISION_PATH)
    if not cv.empty:
        print("Robustness CV:", CV_PATH)
    print("Figures:", FIGURES_DIR)
    print("Index:", INDEX_PATH)
    log_end("final post-Optuna figures")


def _write_readme(cv_exists: bool) -> None:
    lines = [
        "# Final post-Optuna results",
        "",
        f"This folder gathers final {REAL_ALL_2026} outputs for writing and defending the thesis results.",
        "The script that generates this folder is `post_optuna_final_figures.py`; it does not retrain models.",
        "",
        "## Structure",
        "",
        f"- `tables/01_final_model_comparison.csv`: baseline vs Optuna comparison for classic and deep-learning models on {REAL_ALL_2026}.",
        "- `tables/02_final_model_decision.csv`: summary table for choosing the final model.",
        "- `tables/03_post_optuna_robustness_cv.csv`: post-Optuna robustness CV folds, when available.",
        "- `tables/04_final_lightgbm_threshold_classification.csv`: patient-day and episode classification counts at the selected threshold.",
        "- `tables/05_final_lightgbm_threshold_tradeoff.csv`: sensitivity, PPV, specificity and alert burden across thresholds.",
        "- `tables/06_final_lightgbm_top_risk_capture.csv`: positives captured by reviewing the highest-risk observations.",
        "- `figures/01_optuna_effect_real_auprc.png`: whether Optuna changed real-cohort AUPRC.",
        "- `figures/02_post_optuna_test_vs_real_auprc.png`: whether post-Optuna performance transfers from test to real data.",
        "- `figures/03_final_model_decision.png`: compact visual table for the final model choice.",
        "- `figures/04_final_lightgbm_threshold_tradeoff.png`: clinical threshold trade-off.",
        "- `figures/05_final_lightgbm_real_risk_distribution.png`: risk separation in the real cohort.",
        "- `figures/06_final_lightgbm_top_risk_capture.png`: top-risk capture analysis.",
        "- `figures/07_final_lightgbm_confusion_matrix.png`: confusion matrices at the selected threshold.",
        "- `figures/08_final_lightgbm_real_roc_pr.png`: real-cohort ROC and precision-recall curves.",
        "- `post_optuna_final_index.json`: index with all main paths.",
        "",
        "## Current Status",
        "",
        f"- Robustness CV available: {'yes' if cv_exists else 'no'}",
        "",
        "If CV is still missing, run `models_classics_24h_optuna_best_main.py` first "
        "with `CV_FOLDS = 5` and then run this script again.",
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def _clear_pngs(directory: Path) -> None:
    """Remove stale generated PNGs from one figure directory."""
    clear_pngs(directory)


def _remove_empty_dir(directory: Path) -> None:
    """Remove an old generated figure directory when it is empty."""
    if not directory.exists() or any(directory.iterdir()):
        return
    directory.rmdir()


def _write_english_csv(df: pd.DataFrame, path: Path) -> None:
    """Write output tables with English column names."""
    df.to_csv(path, index=False)


def _create_final_comparison() -> pd.DataFrame:
    # Combine classic and deep-learning results under one output schema.
    rows: list[dict[str, object]] = []
    classic_model = _model_classic_optuna(OPTUNA_REAL_POLICY)
    rows.extend(_rows_classic_policy(OPTUNA_REAL_POLICY, classic_model))

    deep_model = _model_deep_optuna(OPTUNA_REAL_POLICY)
    if deep_model:
        rows.extend(_rows_deep_policy(OPTUNA_REAL_POLICY, deep_model))
    return pd.DataFrame(rows)


def _rows_classic_policy(policy: str, model_key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_path = _classic_base_path(policy, "results")
    optuna_path = _classic_optuna_path(policy, model_key, "results")
    for run, path in ((EXECUTION_BASE, base_path), (EXECUTION_OPTUNA, optuna_path)):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        subset = df.loc[df["model"] == model_key].copy()
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "family": MODEL_FAMILY_CLASSIC,
                "run": run,
                "real_policy": policy,
                "policy_label": POLICIES[policy],
                "model": model_key,
                "label": str(row.get("label", model_key)),
                "source": str(path),
                "test_auroc": _num(row, "test_auroc"),
                "test_auprc": _num(row, "test_auprc"),
                "test_auprc_lift": _num(row, "test_auprc_lift"),
                "test_sensitivity": _num(row, "test_sensitivity"),
                "test_ppv": _num(row, "test_ppv"),
                "real_auroc": _num(row, "real_auroc"),
                "real_auprc": _num(row, "real_auprc"),
                "real_auprc_lift": _num(row, "real_auprc_lift"),
                "real_sensitivity": _num(row, "real_sensitivity"),
                "real_ppv": _num(row, "real_ppv"),
                "test_episode_auroc": _num(row, "test_episode_auroc"),
                "test_episode_auprc": _num(row, "test_episode_auprc"),
                "test_episode_auprc_lift": _num(row, "test_episode_auprc_lift"),
                "test_episode_sensitivity": _num(row, "test_episode_sensitivity"),
                "test_episode_ppv": _num(row, "test_episode_ppv"),
                "real_episode_auroc": _num(row, "real_episode_auroc"),
                "real_episode_auprc": _num(row, "real_episode_auprc"),
                "real_episode_auprc_lift": _num(row, "real_episode_auprc_lift"),
                "real_episode_sensitivity": _num(row, "real_episode_sensitivity"),
                "real_episode_ppv": _num(row, "real_episode_ppv"),
            }
        )
    return rows


def _rows_deep_policy(policy: str, model_key: str) -> list[dict[str, object]]:
    model_info = DEEP_MODELS[model_key]
    prefix = str(model_info["prefix"])
    label = str(model_info["label"])
    rows: list[dict[str, object]] = []
    base_path = _deep_base_path(policy, model_key, deep_metrics_filename(prefix))
    optuna_dir = _deep_optuna_dir(policy, model_key, prefix)
    optuna_path = optuna_dir / deep_metrics_filename(prefix) if optuna_dir else None
    for run, path in ((EXECUTION_BASE, base_path), (EXECUTION_OPTUNA, optuna_path)):
        if path is None or not path.exists():
            continue
        metrics = pd.read_csv(path)
        rows.append(
            {
                "family": MODEL_FAMILY_DEEP_LEARNING,
                "run": run,
                "real_policy": policy,
                "policy_label": POLICIES[policy],
                "model": model_key,
                "label": label,
                "source": str(path),
                **_metrics_deep(metrics),
            }
        )
    return rows


def _metrics_deep(metrics: pd.DataFrame) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for split in (SPLIT_TEST, SPLIT_REAL):
        day = _metric_row(metrics, LEVEL_NEXT_DAY, split)
        episode = _metric_row(metrics, LEVEL_EPISODE, split)
        for metric in STANDARD_METRICS:
            values[f"{split}_{metric}"] = _metric_value(day, metric)
            values[f"{split}_episode_{metric}"] = _metric_value(episode, metric)
    return values


def _create_decision_table(comparison: pd.DataFrame) -> pd.DataFrame:
    optuna = comparison.loc[comparison["run"] == EXECUTION_OPTUNA].copy()
    if optuna.empty:
        return pd.DataFrame()
    cols = [
        "family",
        "real_policy",
        "policy_label",
        "model",
        "label",
        COL_REAL_AUPRC,
        COL_REAL_AUPRC_LIFT,
        "real_auroc",
        COL_REAL_SENSITIVITY,
        COL_REAL_PPV,
        COL_REAL_EPISODE_AUPRC,
        COL_REAL_EPISODE_AUPRC_LIFT,
        COL_REAL_EPISODE_AUROC,
        "source",
    ]
    decision = optuna[[col for col in cols if col in optuna.columns]].copy()
    # Rank candidates using real-cohort performance at day and episode levels.
    decision["decision_score"] = (
        pd.to_numeric(decision[COL_REAL_AUPRC], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(decision[COL_REAL_EPISODE_AUPRC], errors="coerce").fillna(0)
    )
    decision["recommendation"] = np.where(
        decision["family"] == MODEL_FAMILY_CLASSIC,
        "Primary candidate: strong performance and interpretability",
        "Comparative candidate: review only if it improves on the baseline enough to justify compute cost",
    )
    return decision.sort_values("decision_score", ascending=False).reset_index(drop=True)


def _load_robustness_cv() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model_key = _model_classic_optuna(OPTUNA_REAL_POLICY)
    path = _classic_optuna_path(OPTUNA_REAL_POLICY, model_key, "cv_folds")
    df = _read_csv_if_has_columns(path)
    if not df.empty and "auprc" in df.columns:
        df = df.copy()
        df.insert(0, "real_policy", OPTUNA_REAL_POLICY)
        df.insert(1, "policy_label", POLICIES[OPTUNA_REAL_POLICY])
        df.insert(2, "source", str(path))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _model_classic_optuna(policy: str) -> str:
    summary_path = _classic_optuna_index_path()
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        policy_payload = _policy_payload(payload, policy)
        model = policy_payload.get("selected_model")
        if model:
            return str(model)
    return "lightgbm"


def _model_deep_optuna(policy: str) -> str | None:
    summary_path = DEEP_LEARNING_OUTPUTS_DIR / "optuna_best" / "deep_learning_24h_optuna_best_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        policy_payload = _policy_payload(payload, policy)
        model = policy_payload.get("selected_model")
        if model:
            return str(model)
    for model_key, info in DEEP_MODELS.items():
        prefix = str(info["prefix"])
        if _deep_optuna_dir(policy, model_key, prefix) is not None:
            return model_key
    return None


def _policy_payload(payload: dict[str, object], policy: str) -> dict[str, object]:
    """Read a policy entry from the current policy key."""
    value = payload.get(policy)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _deep_optuna_dir(policy: str, model_key: str, prefix: str) -> Path | None:
    for path in deep_optuna_dirs(DEEP_LEARNING_OUTPUTS_DIR / "optuna_best", policy, model_key):
        if (path / deep_metrics_filename(prefix)).exists():
            return path
    return None


def _classic_base_path(policy: str, file_key: str) -> Path:
    return MODELS_CLASSICS_OUTPUTS_DIR / policy / CLASSIC_MODEL_FILES[file_key]


def _classic_optuna_path(policy: str, model_key: str, file_key: str) -> Path:
    return MODELS_CLASSICS_OUTPUTS_DIR / "optuna_best" / policy / model_key / CLASSIC_MODEL_FILES[file_key]


def _classic_optuna_index_path() -> Path:
    """Return the new classic Optuna index path, falling back to the legacy one."""
    new_path = MODELS_CLASSICS_OUTPUTS_DIR / "optuna_best" / "classic_models_24h_optuna_best_summary.json"
    if new_path.exists():
        return new_path
    legacy_path = MODELS_CLASSICS_OUTPUTS_DIR / "optuna_best" / "models_classics_24h_optuna_best_summary.json"
    return legacy_path if legacy_path.exists() else new_path


def _deep_base_path(policy: str, model_key: str, filename: str) -> Path:
    return deep_metrics_path(DEEP_LEARNING_OUTPUTS_DIR, OUTPUTS_DIR, model_key, policy, filename)


def _metric_row(metrics: pd.DataFrame, level: str, split: str) -> pd.Series | None:
    subset = metrics.loc[(metrics["level"] == level) & (metrics["split"] == split)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _metric_value(row: pd.Series | None, col: str) -> float | None:
    if row is None or col not in row.index:
        return None
    value = pd.to_numeric(row[col], errors="coerce")
    return None if pd.isna(value) else float(value)


def _read_csv_if_has_columns(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _style() -> None:
    setup_report_style()


def _plot_base_vs_optuna(df: pd.DataFrame, metric: str, level: str, path: Path) -> str:
    _style()
    col = f"real_{metric}" if level == TOP_RISK_LEVEL_DAY else f"real_episode_{metric}"
    plot_df = df.copy()
    plot_df["group"] = plot_df.apply(_group_label, axis=1)
    groups = plot_df["group"].drop_duplicates().tolist()
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for idx, run in enumerate([EXECUTION_BASE, EXECUTION_OPTUNA]):
        subset = plot_df.loc[plot_df["run"] == run].set_index("group").reindex(groups)
        values = pd.to_numeric(subset[col], errors="coerce").to_numpy()
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=run.capitalize(),
            color=COLORS[run],
        )
        _annotate_bars(ax, bars, "{:.2f}", 0.01)
    ax.set_xticks(x, groups, rotation=0, ha="center")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Real cohort {_metric_label(metric)}: base vs Optuna")
    ax.set_ylim(0, _ymax(plot_df[col]))
    ax.legend(loc="upper center", ncol=2)
    ax.text(
        0.5,
        -0.16,
        "Shows whether hyperparameter tuning improved the final real-cohort result.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["muted"],
    )
    return save_report_figure(fig, path)


def _plot_test_vs_real_optuna(df: pd.DataFrame, metric: str, path: Path) -> str:
    _style()
    optuna = df.loc[df["run"] == EXECUTION_OPTUNA].copy()
    optuna["group"] = optuna.apply(_group_label, axis=1)
    groups = optuna["group"].tolist()
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for idx, split in enumerate([SPLIT_TEST, SPLIT_REAL]):
        values = pd.to_numeric(optuna[f"{split}_{metric}"], errors="coerce").to_numpy()
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=split.capitalize(),
            color=COLORS[split],
        )
        _annotate_bars(ax, bars, "{:.2f}", 0.01)
    ax.set_xticks(x, groups, rotation=0, ha="center")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Post-Optuna {_metric_label(metric)}: test vs real")
    ax.set_ylim(0, _ymax(optuna[[f"test_{metric}", f"real_{metric}"]].to_numpy()))
    ax.legend(loc="upper center", ncol=2)
    ax.text(
        0.5,
        -0.16,
        "Checks whether post-Optuna performance remains stable on the 2026 real cohort.",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["muted"],
    )
    return save_report_figure(fig, path)


def _plot_decision_table(decision: pd.DataFrame, path: Path) -> str:
    _style()
    if decision.empty:
        return str(path)
    cols = [
        "family",
        "label",
        "real_auprc",
        "real_episode_auprc",
        "real_ppv",
        "real_sensitivity",
        "real_auprc_lift",
    ]
    table = decision[[col for col in cols if col in decision.columns]].copy()
    for col in table.columns:
        if col.startswith("real_"):
            table[col] = pd.to_numeric(table[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table["family"] = table["family"].map(_family_label)
    table = table.rename(
        columns={
            "family": "Family",
            "label": "Model",
            "real_auprc": "Real AUPRC",
            "real_episode_auprc": "Episode AUPRC",
            "real_ppv": "Real PPV",
            "real_sensitivity": "Real sensitivity",
            "real_auprc_lift": "AUPRC lift",
        }
    )
    fig_height = max(2.2, 0.42 * len(table) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.0, 1.0, 0.70],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.5)
    mpl_table.scale(1, 1.45)
    for (row, _col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E6EEF5")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#D0D7DE")
    ax.set_title("Final model decision", pad=8)
    ax.text(
        0.5,
        0.82,
        "Main real-cohort metrics after Optuna. Higher values are better.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["muted"],
    )
    return save_report_figure(fig, path)


def _create_clinical_usefulness_outputs() -> tuple[dict[str, str], dict[str, str]]:
    """Create final LightGBM threshold tables and clinical-usefulness figures."""
    model_key = _model_classic_optuna(OPTUNA_REAL_POLICY)
    results_path = _classic_optuna_path(OPTUNA_REAL_POLICY, model_key, "results")
    predictions_path = _classic_optuna_path(OPTUNA_REAL_POLICY, model_key, "predictions")
    if not results_path.exists() or not predictions_path.exists():
        return {}, {}

    results = pd.read_csv(results_path)
    model_results = results.loc[results["model"] == model_key]
    if model_results.empty:
        return {}, {}
    # Reuse the threshold selected during validation for real-cohort reporting.
    threshold = float(model_results.iloc[0]["threshold_valid"])

    predictions = pd.read_csv(predictions_path)
    real = predictions.loc[predictions["split"] == SPLIT_REAL].copy()
    if real.empty:
        return {}, {}

    classification = _classification_table(real, threshold)
    threshold_curve = _threshold_tradeoff_table(real)
    top_risk = _top_risk_capture_table(real)

    _write_english_csv(classification, CLINICAL_CLASSIFICATION_PATH)
    _write_english_csv(threshold_curve, CLINICAL_THRESHOLD_CURVE_PATH)
    _write_english_csv(top_risk, CLINICAL_TOP_RISK_PATH)

    figures = {
        "04_final_lightgbm_threshold_tradeoff": _plot_threshold_tradeoff(
            threshold_curve,
            threshold,
            path=FIGURES_DIR / "04_final_lightgbm_threshold_tradeoff.png",
        ),
        "05_final_lightgbm_real_risk_distribution": _plot_real_risk_distribution(
            real,
            threshold,
            path=FIGURES_DIR / "05_final_lightgbm_real_risk_distribution.png",
        ),
        "06_final_lightgbm_top_risk_capture": _plot_top_risk_capture(
            top_risk,
            path=FIGURES_DIR / "06_final_lightgbm_top_risk_capture.png",
        ),
        "07_final_lightgbm_confusion_matrix": _plot_confusion_matrices(
            classification,
            path=FIGURES_DIR / "07_final_lightgbm_confusion_matrix.png",
        ),
        "08_final_lightgbm_real_roc_pr": _plot_roc_pr_curves(
            real,
            classification,
            path=FIGURES_DIR / "08_final_lightgbm_real_roc_pr.png",
        ),
    }
    tables = {
        "classification": str(CLINICAL_CLASSIFICATION_PATH),
        "threshold_tradeoff": str(CLINICAL_THRESHOLD_CURVE_PATH),
        "top_risk_capture": str(CLINICAL_TOP_RISK_PATH),
    }
    return tables, figures


def _classification_table(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build patient-day and episode classification counts at one threshold."""
    # Evaluate the same predictions at patient-day and episode levels.
    rows = []
    for level, data in (
        ("patient_day", predictions[["y_true", "score"]].copy()),
        (
            "episode",
            predictions.groupby("Episodi", as_index=False)
            .agg(y_true=("y_true", "max"), score=("score", "max"))[["y_true", "score"]],
        ),
    ):
        y_true = data["y_true"].astype(int)
        y_pred = (data["score"].astype(float) >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "level": level,
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "total_observations": int(len(data)),
                "positive_observations": int(y_true.sum()),
                "alerts": int(y_pred.sum()),
                "prevalence": _safe_div(float(y_true.sum()), float(len(data))),
                "sensitivity": _safe_div(tp, tp + fn),
                "specificity": _safe_div(tn, tn + fp),
                "ppv": _safe_div(tp, tp + fp),
                "npv": _safe_div(tn, tn + fn),
                "alert_rate": _safe_div(int(y_pred.sum()), int(len(data))),
            }
        )
    return pd.DataFrame(rows)


def _safe_div(numerator: float, denominator: float) -> float:
    """Return a numeric ratio while avoiding zero-division errors."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _threshold_tradeoff_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute threshold metrics for the real patient-day cohort."""
    # Sweep empirical score quantiles to show the alert-performance trade-off.
    y_true = predictions["y_true"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    thresholds = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
    rows = []
    for threshold in thresholds:
        y_pred = (score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": _safe_div(tp, tp + fn),
                "specificity": _safe_div(tn, tn + fp),
                "ppv": _safe_div(tp, tp + fp),
                "npv": _safe_div(tn, tn + fn),
                "alert_rate": _safe_div(tp + fp, len(y_true)),
                "alerts_per_100_patient_days": 100 * _safe_div(tp + fp, len(y_true)),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def _top_risk_capture_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize positives captured within the highest-risk patient-days and episodes."""
    # Measure how many positives are found when only the highest-risk observations are reviewed.
    rows = []
    for level, data in (
        ("patient_day", predictions[["y_true", "score"]].copy()),
        (
            "episode",
            predictions.groupby("Episodi", as_index=False)
            .agg(y_true=("y_true", "max"), score=("score", "max"))[["y_true", "score"]],
        ),
    ):
        ranked = data.sort_values("score", ascending=False).reset_index(drop=True)
        n_total = len(ranked)
        n_positive = int(ranked["y_true"].sum())
        for pct in (1, 2, 5, 10, 20):
            n_reviewed = max(1, int(np.ceil(n_total * pct / 100)))
            selected = ranked.head(n_reviewed)
            captured = int(selected["y_true"].sum())
            rows.append(
                {
                    "level": level,
                    "top_percent": pct,
                    "n_reviewed": n_reviewed,
                    "captured_positives": captured,
                    "total_positives": n_positive,
                    "sensitivity": _safe_div(captured, n_positive),
                    "ppv": _safe_div(captured, n_reviewed),
                }
            )
    return pd.DataFrame(rows)


def _plot_threshold_tradeoff(data: pd.DataFrame, selected_threshold: float, path: Path) -> str:
    """Plot sensitivity, PPV and alert burden across thresholds."""
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    ax.plot(data["threshold"], data["sensitivity"], color=PALETTE["blue"], linewidth=2.4, label="Sensitivity")
    ax.plot(data["threshold"], data["ppv"], color=PALETTE["orange"], linewidth=2.4, label="PPV")
    ax.plot(data["threshold"], data["alert_rate"], color=PALETTE["teal"], linewidth=2.0, label="Alert rate")
    ax.axvline(selected_threshold, color=PALETTE["ink"], linestyle="--", linewidth=1.3, label=f"Selected threshold ({selected_threshold:.3f})")
    ax.set_title("Threshold trade-off for the final LightGBM model", pad=12)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", alpha=0.12)
    return save_report_figure(fig, path)


def _plot_real_risk_distribution(predictions: pd.DataFrame, threshold: float, path: Path) -> str:
    """Plot predicted risk distributions in the real patient-day cohort."""
    _style()
    score = predictions["score"].astype(float)
    y_true = predictions["y_true"].astype(int)
    fig, ax = plt.subplots(figsize=(9.2, 5.3), constrained_layout=True)
    bins = np.linspace(0, 1, 45)
    ax.hist(
        score[y_true == 0],
        bins=bins,
        density=True,
        alpha=0.72,
        color=PALETTE["sky_blue"],
        label="No next-day sepsis",
    )
    ax.hist(
        score[y_true == 1],
        bins=bins,
        density=True,
        alpha=0.70,
        color=PALETTE["orange"],
        label="Next-day sepsis",
    )
    ax.axvline(threshold, color=PALETTE["ink"], linestyle="--", linewidth=1.4, label=f"Selected threshold ({threshold:.3f})")
    ax.set_title("Predicted risk distribution in the all-2026 real-world cohort", pad=12)
    ax.set_xlabel("Estimated risk")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.grid(axis="x", alpha=0.08)
    return save_report_figure(fig, path)


def _plot_top_risk_capture(data: pd.DataFrame, path: Path) -> str:
    """Plot sensitivity from reviewing the highest-risk observations."""
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for level, color, label in (
        ("patient_day", PALETTE["blue"], "Patient-day"),
        ("episode", PALETTE["orange"], "Episode"),
    ):
        subset = data.loc[data["level"] == level].copy()
        ax.plot(
            subset["top_percent"],
            subset["sensitivity"],
            marker="o",
            linewidth=2.4,
            color=color,
            label=label,
        )
        for _, row in subset.iterrows():
            ax.text(
                row["top_percent"],
                row["sensitivity"] + 0.025,
                f"{row['sensitivity']:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=PALETTE["ink"],
            )
    ax.set_title("Top-risk capture by review fraction", pad=12)
    ax.set_xlabel("Highest-risk observations reviewed (%)")
    ax.set_ylabel("Share of positives captured")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", alpha=0.12)
    return save_report_figure(fig, path)


def _plot_confusion_matrices(data: pd.DataFrame, path: Path) -> str:
    """Plot patient-day and episode confusion matrices at the selected threshold."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    image = None
    for ax, level, title in (
        (axes[0], "patient_day", "Patient-day level"),
        (axes[1], "episode", "Episode level"),
    ):
        row = data.loc[data["level"] == level].iloc[0]
        counts = np.array(
            [
                [int(row["true_negatives"]), int(row["false_positives"])],
                [int(row["false_negatives"]), int(row["true_positives"])],
            ],
            dtype=float,
        )
        row_totals = counts.sum(axis=1, keepdims=True)
        row_percent = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)
        image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, pad=8)
        ax.set_xticks([0, 1], labels=["No alert", "Alert"])
        ax.set_yticks([0, 1], labels=["No sepsis", "Sepsis"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.tick_params(axis="both", length=0)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                color = "white" if row_percent[i, j] > 0.55 else PALETTE["ink"]
                ax.text(
                    j,
                    i,
                    f"{int(counts[i, j]):,}\n{row_percent[i, j]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    weight="bold",
                )
        ax.text(
            0.5,
            -0.24,
            f"Sensitivity {row['sensitivity']:.1%}   PPV {row['ppv']:.1%}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=PALETTE["muted"],
        )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_label("Row percentage")
    fig.suptitle("Confusion matrices at the selected LightGBM threshold", y=1.04)
    return save_report_figure(fig, path)


def _plot_roc_pr_curves(predictions: pd.DataFrame, classification: pd.DataFrame, path: Path) -> str:
    """Plot real-cohort ROC and precision-recall curves for the final model."""
    # Mark the validation-selected operating point on both discrimination curves.
    _style()
    y_true = predictions["y_true"].astype(int).to_numpy()
    score = predictions["score"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y_true, score)
    prevalence = float(np.mean(y_true))

    selected = classification.loc[classification["level"] == "patient_day"].iloc[0]
    selected_fpr = 1.0 - float(selected["specificity"])
    selected_tpr = float(selected["sensitivity"])
    selected_precision = float(selected["ppv"])
    selected_recall = float(selected["sensitivity"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=2.4, label=f"AUROC {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1.2, label="No skill")
    ax.scatter(
        selected_fpr,
        selected_tpr,
        color=PALETTE["orange"],
        s=58,
        zorder=4,
        label="Selected threshold",
    )
    ax.set_title("ROC curve", pad=8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.16)

    ax = axes[1]
    ax.plot(recall, precision, color=PALETTE["orange"], linewidth=2.4, label=f"AUPRC {auprc:.3f}")
    ax.axhline(prevalence, color=PALETTE["muted"], linestyle="--", linewidth=1.2, label=f"Prevalence {prevalence:.1%}")
    ax.scatter(
        selected_recall,
        selected_precision,
        color=PALETTE["blue"],
        s=58,
        zorder=4,
        label="Selected threshold",
    )
    ax.set_title("Precision-recall curve", pad=8)
    ax.set_xlabel("Sensitivity / recall")
    ax.set_ylabel("Precision / PPV")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.16)

    fig.suptitle("Real-world discrimination of the final LightGBM model", y=1.03)
    return save_report_figure(fig, path)


def _annotate_bars(ax, bars, fmt: str, offset: float) -> None:
    annotate_bars(ax, bars, fmt, offset, color="#111111")


def _num(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    value = pd.to_numeric(row[col], errors="coerce")
    return None if pd.isna(value) else float(value)


def _family_label(value: str) -> str:
    return {
        MODEL_FAMILY_CLASSIC: "Classic",
        MODEL_FAMILY_DEEP_LEARNING: "Deep learning",
    }.get(str(value), str(value))


def _group_label(row: pd.Series) -> str:
    """Use short plot labels; omit the policy when only the all-patient policy is shown."""
    family = _family_label(str(row.get("family", "")))
    policy = str(row.get("policy_label", "")).strip()
    return family if policy == "All" or not policy else f"{family} ({policy})"


def _metric_label(metric: str) -> str:
    return metric_label(metric)


def _ymax(values) -> float:
    return padded_ymax(values)


if __name__ == "__main__":
    main()


