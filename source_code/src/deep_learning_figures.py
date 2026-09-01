from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DEEP_LEARNING_OUTPUTS_DIR, FIGURES_DIR, OUTPUTS_DIR
from src.figure_style import PALETTE, save_report_figure
from src.output_contracts import (
    DEEP_SUMMARY_SUFFIX,
    LEVEL_EPISODE,
    LEVEL_NEXT_DAY,
    SPLIT_REAL,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    STANDARD_SPLITS,
    deep_predictions_filename,
    deep_summary_filename,
)
from src.plot_utils import (
    annotate_bars as _shared_annotate_bars,
    clear_pngs as _clear_pngs,
    setup_report_style as _setup_style,
)
ID_COL = "Episodi"
TARGET = "next_day_sepsis"
SPLITS = STANDARD_SPLITS
MODEL_PREFIXES = {
    "transformer": "transformer_24h",
    "lstm": "lstm_24h",
    "rnn": "rnn_24h",
}
SPLIT_LABELS = {
    SPLIT_TRAIN: "Train",
    SPLIT_VALID: "Validation",
    SPLIT_TEST: "Test",
    SPLIT_REAL: "Real 2026",
}
COLORS = {
    SPLIT_TRAIN: PALETTE["blue"],
    SPLIT_VALID: PALETTE["teal"],
    SPLIT_TEST: PALETTE["gold"],
    SPLIT_REAL: PALETTE["orange"],
    "sepsis": PALETTE["orange"],
    "no_sepsis": PALETTE["blue"],
    "neutral": PALETTE["muted"],
    "metric_1": PALETTE["blue"],
    "metric_2": PALETTE["orange"],
    "metric_3": PALETTE["teal"],
    "metric_4": PALETTE["purple"],
    "metric_5": PALETTE["vermilion"],
}


def main() -> None:
    """CLI entry point for regenerating temporal-model figures."""
    parser = argparse.ArgumentParser(
        description="Generate temporal-model figures from already saved predictions."
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix, for example transformer_24h_real_all_2026. "
            "If omitted, figures are generated for every available temporal model."
        ),
    )
    args = parser.parse_args()

    # Figures are regenerated from saved outputs; this command does not retrain models.
    if args.output_prefix:
        paths = generate_temporal_model_figures(args.output_prefix)
    else:
        paths = generate_available_temporal_model_figures()
    print("Temporal-model figures generated")
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def generate_temporal_model_figures(
    output_prefix: str = "transformer_24h",
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate report figures for one saved temporal-model run.

    The saved summary and predictions are used to create per-run metric,
    calibration, risk-distribution, and threshold figures.
    """
    paths_info = _resolve_temporal_outputs(output_prefix, output_dir=output_dir)
    summary_path = paths_info["summary"]
    predictions_path = paths_info["predictions"]
    file_prefix = str(paths_info["file_prefix"])
    if not summary_path.exists():
        raise FileNotFoundError(f"The summary does not exist: {summary_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"The predictions file does not exist: {predictions_path}")

    # The summary provides metrics and the validation threshold; predictions provide scores.
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    predictions = pd.read_csv(predictions_path)
    score_col = _prediction_column(predictions, file_prefix)
    model_label = _model_label(summary, output_prefix)

    figures_dir = paths_info["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(figures_dir)

    # Keep only recognized splits so auxiliary rows cannot enter the figures.
    datasets = {
        split: df_split.copy()
        for split, df_split in predictions.groupby("split", sort=False)
        if split in SPLITS
    }
    paths: dict[str, str] = {}
    metrics = summary["metrics"]
    paths["next_day_metrics_comparison"] = _plot_metrics_comparison(
        metrics,
        model_label,
        level="next_day",
        suffix="",
        path=figures_dir / "01_next_day_metrics_comparison.png",
    )
    paths["auprc_lift"] = _plot_auprc_lift(
        metrics,
        model_label,
        path=figures_dir / "02_auprc_lift_and_prevalence.png",
    )
    paths["roc_next_day"] = _plot_curve(
        datasets,
        score_col,
        model_label,
        level="next_day",
        curve="roc",
        path=figures_dir / "03_next_day_roc.png",
    )
    paths["precision_recall_next_day"] = _plot_curve(
        datasets,
        score_col,
        model_label,
        level="next_day",
        curve="pr",
        path=figures_dir / "04_next_day_precision_recall.png",
    )
    # Apply the threshold selected on validation when displaying real-cohort risk.
    threshold = float(summary["threshold_youden_valid"])
    if "real" in datasets:
        paths["real_next_day_risk_distribution"] = _plot_risk_distribution(
            datasets["real"][TARGET].astype(int).to_numpy(),
            datasets["real"][score_col].astype(float).to_numpy(),
            threshold,
            f"{model_label} - real - next-day level",
            figures_dir / "05_real_next_day_risk_distribution.png",
        )

    index_path = paths_info["figures_index"]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    return paths


def generate_available_temporal_model_figures() -> dict[str, dict[str, str]]:
    """Generate figures for every saved temporal-model run that can be found."""
    # Discover both the organized policy folders and the legacy flat output layout.
    prefixes = _detect_temporal_output_prefixes()
    if not prefixes:
        raise FileNotFoundError(
            "No temporal outputs with summary and prediction files were found at "
            f"{DEEP_LEARNING_OUTPUTS_DIR}. Run scripts/06_deep_learning.py first."
        )
    return {prefix: generate_temporal_model_figures(prefix) for prefix in prefixes}


def _detect_temporal_output_prefixes() -> list[str]:
    prefixes: list[str] = []
    for model_key, base_prefix in MODEL_PREFIXES.items():
        model_dir = DEEP_LEARNING_OUTPUTS_DIR / model_key
        if not model_dir.exists():
            continue
        for policy_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            summary_path = policy_dir / deep_summary_filename(base_prefix)
            predictions_path = policy_dir / deep_predictions_filename(base_prefix)
            if summary_path.exists() and predictions_path.exists():
                prefixes.append(f"{base_prefix}_{policy_dir.name}")

    for summary_path in sorted(OUTPUTS_DIR.glob(f"*{DEEP_SUMMARY_SUFFIX}")):
        prefix = summary_path.name[: -len(DEEP_SUMMARY_SUFFIX)]
        predictions_path = OUTPUTS_DIR / deep_predictions_filename(prefix)
        if not predictions_path.exists():
            continue
        if not _is_temporal_prefix(prefix):
            continue
        prefixes.append(prefix)
    return prefixes


def _resolve_temporal_outputs(
    output_prefix: str,
    output_dir: Path | None = None,
) -> dict[str, Path | str]:
    """Find temporal-model files in the organized output structure."""
    if output_dir is not None:
        run_dir = Path(output_dir)
        return {
            "summary": run_dir / deep_summary_filename(output_prefix),
            "predictions": run_dir / deep_predictions_filename(output_prefix),
            "figures_dir": run_dir / "figures",
            "figures_index": run_dir / f"{output_prefix}_figures_index.json",
            "file_prefix": output_prefix,
        }

    for model_key, base_prefix in MODEL_PREFIXES.items():
        if output_prefix == base_prefix:
            output_dir = DEEP_LEARNING_OUTPUTS_DIR / model_key / "simple_execution"
            summary = output_dir / deep_summary_filename(base_prefix)
            predictions = output_dir / deep_predictions_filename(base_prefix)
            if summary.exists() or predictions.exists():
                return {
                    "summary": summary,
                    "predictions": predictions,
                    "figures_dir": output_dir / "figures",
                    "figures_index": output_dir / f"{base_prefix}_figures_index.json",
                    "file_prefix": base_prefix,
                }

    parsed = _parse_output_prefix(output_prefix)
    if parsed is not None:
        model_key, base_prefix, policy_key = parsed
        output_dir = DEEP_LEARNING_OUTPUTS_DIR / model_key / policy_key
        summary = output_dir / deep_summary_filename(base_prefix)
        predictions = output_dir / deep_predictions_filename(base_prefix)
        if summary.exists() or predictions.exists():
            return {
                "summary": summary,
                "predictions": predictions,
                "figures_dir": output_dir / "figures",
                "figures_index": output_dir / f"{base_prefix}_figures_index.json",
                "file_prefix": base_prefix,
            }

    return {
        "summary": OUTPUTS_DIR / deep_summary_filename(output_prefix),
        "predictions": OUTPUTS_DIR / deep_predictions_filename(output_prefix),
        "figures_dir": FIGURES_DIR / output_prefix,
        "figures_index": OUTPUTS_DIR / f"{output_prefix}_figures_index.json",
        "file_prefix": output_prefix,
    }


def _parse_output_prefix(output_prefix: str) -> tuple[str, str, str] | None:
    for model_key, base_prefix in MODEL_PREFIXES.items():
        prefix_with_sep = f"{base_prefix}_"
        if output_prefix.startswith(prefix_with_sep):
            return model_key, base_prefix, output_prefix[len(prefix_with_sep) :]
    return None


def _is_temporal_prefix(prefix: str) -> bool:
    return any(prefix == base or prefix.startswith(f"{base}_") for base in MODEL_PREFIXES.values())


def _model_label(summary: dict[str, object], output_prefix: str) -> str:
    model_type = str(summary.get("model_type", "")).lower()
    if model_type == "lstm":
        return "LSTM"
    if model_type == "rnn":
        return "RNN"
    if model_type == "transformer":
        return "Transformer"
    label = str(summary.get("model", "")).strip()
    if label:
        return label
    return output_prefix


def _prediction_column(predictions: pd.DataFrame, output_prefix: str) -> str:
    # Prediction-column names vary by model and output layout, so resolve them safely.
    preferred = "sepsis_risk_24h_transformer"
    if preferred in predictions.columns:
        return preferred
    safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in output_prefix).strip("_")
    candidate = f"sepsis_risk_24h_{safe_prefix}"
    if candidate in predictions.columns:
        return candidate
    candidates = [col for col in predictions.columns if col.startswith("sepsis_risk_24h")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("The risk prediction column could not be identified.")


def _save_figure(fig: plt.Figure, path: Path) -> str:
    """Save one figure with report-friendly resolution and spacing."""
    return save_report_figure(fig, path)


def _split_label(split: str) -> str:
    return SPLIT_LABELS.get(split, split)


def _level_label(level: str) -> str:
    if level == LEVEL_NEXT_DAY:
        return "next-day level"
    if level == LEVEL_EPISODE:
        return "episode level"
    return level


def _plot_metrics_comparison(
    metrics: dict[str, dict[str, object]],
    model_label: str,
    level: str,
    suffix: str,
    path: Path,
) -> str:
    _setup_style()
    # Compare threshold-independent and threshold-dependent metrics by split.
    rows = []
    for split in SPLITS:
        key = f"{split}{suffix}"
        if key in metrics:
            values = metrics[key]
            rows.append(
                {
                    "split": split,
                    "AUROC": values["auroc"],
                    "AUPRC": values["auprc"],
                    "Sensitivity": values["sensitivity"],
                    "PPV": values["ppv"],
                    "F1": values["f1"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return ""

    labels = [_split_label(split) for split in df["split"].tolist()]
    metric_cols = ["AUROC", "AUPRC", "Sensitivity", "PPV", "F1"]
    colors = [COLORS[f"metric_{idx}"] for idx in range(1, 6)]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        bars = ax.bar(
            x + (i - 2) * width,
            pd.to_numeric(df[metric], errors="coerce"),
            width,
            label=metric,
            color=color,
        )
        _annotate_bars(ax, bars, "{:.2f}", 0.012)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_label}: metrics by split ({_level_label(level)})")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)
    return _save_figure(fig, path)


def _plot_auprc_lift(metrics: dict[str, dict[str, object]], model_label: str, path: Path) -> str:
    _setup_style()
    # AUPRC is interpreted together with prevalence; lift shows improvement over baseline.
    rows = []
    for split in SPLITS:
        values = metrics.get(split)
        if not values or "auprc_lift" not in values:
            continue
        rows.append(
            {
                "split": split,
                "AUPRC": values.get("auprc"),
                "Prevalence": values.get("prevalence"),
                "AUPRC lift": values.get("auprc_lift"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return ""

    labels = [_split_label(split) for split in df["split"].tolist()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    width = 0.34
    bars_auprc = axes[0].bar(
        x - width / 2,
        pd.to_numeric(df["AUPRC"], errors="coerce"),
        width,
        color=COLORS["metric_2"],
        label="AUPRC",
    )
    bars_prev = axes[0].bar(
        x + width / 2,
        pd.to_numeric(df["Prevalence"], errors="coerce"),
        width,
        color=COLORS["metric_3"],
        label="Prevalence",
    )
    _annotate_bars(axes[0], bars_auprc, "{:.2f}", 0.012)
    _annotate_bars(axes[0], bars_prev, "{:.2f}", 0.012)
    axes[0].set_xticks(x, labels=labels, rotation=0, ha="center")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("AUPRC and prevalence")
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].grid(axis="x", visible=False)

    lift = pd.to_numeric(df["AUPRC lift"], errors="coerce")
    max_lift = _nanmax_or_default(lift, default=1.0)
    bars_lift = axes[1].bar(x, lift, color=COLORS["metric_4"])
    _annotate_bars(axes[1], bars_lift, "{:.1f}x", max(max_lift * 0.02, 0.05))
    axes[1].set_xticks(x, labels=labels, rotation=0, ha="center")
    axes[1].set_ylim(0, max(1.05, max_lift * 1.18))
    axes[1].set_title("AUPRC lift")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle(f"{model_label}: AUPRC compared with prevalence", fontsize=14, fontweight="bold")
    return _save_figure(fig, path)


def _plot_curve(
    datasets: dict[str, pd.DataFrame],
    score_col: str,
    model_label: str,
    level: str,
    curve: str,
    path: Path,
) -> str:
    _setup_style()
    # ROC and PR curves use saved risk scores and are drawn separately for each split.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for split in SPLITS:
        if split not in datasets:
            continue
        df_split = datasets[split]
        if level == LEVEL_EPISODE:
            y, score = _aggregate_prediction_by_episode(df_split, score_col)
        else:
            y = df_split[TARGET].astype(int).to_numpy()
            score = df_split[score_col].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue

        if curve == "roc":
            x_values, y_values = _roc_points(y, score)
            label = f"{_split_label(split)} ({_auroc(y, score):.3f})"
            ax.plot(x_values, y_values, linewidth=2.2, label=label, color=COLORS[split])
        elif curve == "pr":
            x_values, y_values = _pr_points(y, score)
            label = f"{_split_label(split)} ({_auprc(y, score):.3f})"
            ax.plot(x_values, y_values, linewidth=2.2, label=label, color=COLORS[split])
        else:
            raise ValueError("curve must be 'roc' or 'pr'.")

    if curve == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["neutral"], linewidth=1)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"{model_label}: ROC curves ({_level_label(level)})")
        ax.legend(loc="lower right", fontsize=8)
    else:
        prevalences = []
        for df_split in datasets.values():
            if level == LEVEL_EPISODE:
                y_prev, _ = _aggregate_prediction_by_episode(df_split, score_col)
                prevalences.append(float(y_prev.mean()))
            else:
                prevalences.append(float(df_split[TARGET].mean()))
        if prevalences:
            ax.axhline(
                float(np.mean(prevalences)),
                linestyle="--",
                color=COLORS["neutral"],
                linewidth=1,
                label="Mean prevalence",
            )
        ax.set_xlabel("Sensitivity / recall")
        ax.set_ylabel("Precision / PPV")
        ax.set_title(f"{model_label}: precision-recall curves ({_level_label(level)})")
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.22)
    return _save_figure(fig, path)


def _plot_risk_distribution(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    title: str,
    path: Path,
) -> str:
    _setup_style()
    # The two distributions show how well predicted risks separate the two classes.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(score[y == 0], bins=50, alpha=0.72, density=True, color=COLORS["no_sepsis"], label="No sepsis")
    ax.hist(score[y == 1], bins=30, alpha=0.72, density=True, color=COLORS["sepsis"], label="Sepsis")
    ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label=f"threshold {threshold:.3f}")
    ax.set_xlabel("Estimated risk")
    ax.set_ylabel("Density")
    ax.set_title(f"Risk distribution: {title}")
    ax.legend()
    ax.grid(alpha=0.22)
    return _save_figure(fig, path)


def _aggregate_prediction_by_episode(df_split: pd.DataFrame, score_col: str) -> tuple[np.ndarray, np.ndarray]:
    # An episode is positive if any of its rows is positive; its risk is the maximum daily risk.
    tmp = df_split[[ID_COL, TARGET, score_col]].copy()
    episode = tmp.groupby(ID_COL).agg(
        y_true=(TARGET, "max"),
        y_score=(score_col, "max"),
    )
    return (
        episode["y_true"].astype(int).to_numpy(),
        episode["y_score"].astype(float).to_numpy(),
    )


def _roc_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    p = max(int((y == 1).sum()), 1)
    n = max(int((y == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tp / p, [1.0]])
    fpr = np.concatenate([[0.0], fp / n, [1.0]])
    return fpr, tpr


def _pr_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    p = max(int((y == 1).sum()), 1)
    recall = np.concatenate([[0.0], tp / p])
    precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1)])
    return recall, precision


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr = _roc_points(y, score)
    return float(np.trapz(tpr, fpr))


def _auprc(y: np.ndarray, score: np.ndarray) -> float:
    recall, precision = _pr_points(y, score)
    return float(np.trapz(precision, recall))


def _annotate_bars(ax: plt.Axes, bars, fmt: str, dy: float) -> None:
    _shared_annotate_bars(ax, bars, fmt, dy, color="#111111")


def _nanmax_or_default(values: pd.Series, default: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        return default
    return float(finite.max())


if __name__ == "__main__":
    main()



