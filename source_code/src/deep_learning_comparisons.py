from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DEEP_LEARNING_OUTPUTS_DIR, OUTPUTS_DIR
from src.figure_style import PALETTE, save_report_figure
from src.output_contracts import (
    LEVEL_EPISODE,
    LEVEL_NEXT_DAY,
    METRIC_AUPRC,
    SPLIT_REAL,
    SPLIT_TEST,
    STANDARD_SPLITS,
    deep_metrics_filename,
)
from src.output_paths import deep_metrics_path
from src.plot_utils import (
    annotate_bars as _annotate_bars,
    clear_pngs as _clear_pngs,
    metric_label as _metric_label,
    metric_ymax as _metric_ymax,
    setup_report_style as _setup_style,
)
from src.real_policies import real_policy_labels
from src.progress import log_end, log_start, step


POLICIES = real_policy_labels(short=True)

MODELS = {
    "transformer": {"prefix": "transformer_24h", "label": "Transformer"},
    "lstm": {"prefix": "lstm_24h", "label": "LSTM"},
    "rnn": {"prefix": "rnn_24h", "label": "RNN"},
}

COMPARISON_DIR = DEEP_LEARNING_OUTPUTS_DIR / "comparison"
FIGURES_DIR = COMPARISON_DIR / "figures"
INDEX_PATH = COMPARISON_DIR / "deep_learning_24h_comparison_index.json"
COMPARISON_PATH = COMPARISON_DIR / "deep_learning_24h_comparison.csv"

COLORS = {
    SPLIT_TEST: PALETTE["blue"],
    SPLIT_REAL: PALETTE["orange"],
    "policy_alt": PALETTE["teal"],
    LEVEL_EPISODE: PALETTE["purple"],
    "lift": PALETTE["green"],
    "neutral": PALETTE["muted"],
}
POLICY_COLORS = [PALETTE["orange"], PALETTE["teal"], PALETTE["blue"]]


def main() -> None:
    """Build CSV comparisons and figures from completed deep-learning runs."""
    log_start("comparative deep-learning figures")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Remove previous figures so the index contains only the current comparison.
    _clear_pngs(FIGURES_DIR)

    with step("Load comparable metrics", number=1, total=2):
        # All models and real-cohort policies are assembled into one comparable table.
        comparison = _build_comparison_table()
        if comparison.empty:
            raise FileNotFoundError(
                "No comparable deep-learning metrics with policies were found. "
                "Run scripts/06_deep_learning.py first."
            )
        comparison.to_csv(COMPARISON_PATH, index=False)

    # Use complementary figures to compare policies, evaluation levels and thresholds.
    figures: dict[str, str] = {}
    with step("Generate figures", number=2, total=2):
        figures["comparison_01_real_auprc"] = _plot_metric_real_by_policy(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_01_real_auprc_by_policy.png",
        )
        figures["comparison_01b_real_auprc_bars"] = _plot_metric_real_by_policy_bars(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_01b_real_auprc_by_policy_bars.png",
        )
        figures["comparison_02_real_auprc_row_vs_episode"] = _plot_row_vs_episode(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_02_real_auprc_row_vs_episode.png",
        )
        figures["comparison_03_best_models_summary"] = _plot_best_summary(
            comparison,
            path=FIGURES_DIR / "comparison_03_best_deep_learning_models.png",
        )
        figures["comparison_04_real_sensitivity_ppv"] = _plot_sensitivity_ppv_real(
            comparison,
            path=FIGURES_DIR / "comparison_04_real_sensitivity_ppv.png",
        )

    payload = {
        "figures_dir": str(FIGURES_DIR),
        "comparison_csv": str(COMPARISON_PATH),
        "figures": figures,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Generated figures:", FIGURES_DIR)
    print("Comparison table:", COMPARISON_PATH)
    print("Index:", INDEX_PATH)
    log_end("comparative deep-learning figures")


def _build_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_key, model_info in MODELS.items():
        base_prefix = str(model_info["prefix"])
        for policy_key, policy_label in POLICIES.items():
            metrics_path = _model_metrics_path(model_key, base_prefix, policy_key)
            if not metrics_path.exists():
                continue
            metrics = pd.read_csv(metrics_path)
            row = {
                "model": model_key,
                "label": model_info["label"],
                "real_policy": policy_key,
                "policy_label": policy_label,
                "source": str(metrics_path),
            }
            # Store both next-day row metrics and episode-level metrics for each split.
            for split in STANDARD_SPLITS:
                day = _metric_row(metrics, level=LEVEL_NEXT_DAY, split=split)
                episode = _metric_row(metrics, level=LEVEL_EPISODE, split=split)
                for metric in ("n", "positives", "prevalence", "auroc", "auprc", "auprc_lift", "sensitivity", "ppv", "f1"):
                    row[f"{split}_{metric}"] = _value(day, metric)
                    row[f"{split}_episode_{metric}"] = _value(episode, metric)
            # Positive deltas indicate higher performance on the real cohort than test.
            row["delta_auprc_real_test"] = _num(row.get("real_auprc")) - _num(row.get("test_auprc"))
            row["delta_auroc_real_test"] = _num(row.get("real_auroc")) - _num(row.get("test_auroc"))
            rows.append(row)
    return pd.DataFrame(rows)


def _model_metrics_path(model_key: str, base_prefix: str, policy_key: str) -> Path:
    """Return the current organized metrics path for one model and policy."""
    return deep_metrics_path(
        DEEP_LEARNING_OUTPUTS_DIR,
        OUTPUTS_DIR,
        model_key,
        policy_key,
        deep_metrics_filename(base_prefix),
    )


def _metric_row(metrics: pd.DataFrame, level: str, split: str) -> pd.Series | None:
    # A metrics file contains separate rows for next-day predictions and episodes.
    subset = metrics.loc[(metrics["level"] == level) & (metrics["split"] == split)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _value(row: pd.Series | None, col: str) -> float | int | None:
    if row is None or col not in row.index:
        return None
    value = pd.to_numeric(row[col], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _plot_metric_real_by_policy(df: pd.DataFrame, metric: str, path: Path) -> str:
    _setup_style()
    # Heatmap values are sorted by the complete real cohort when available.
    pivot = df.pivot(index="label", columns="policy_label", values=f"real_{metric}")
    preferred_policies = ["All", "New", "Readmitted"]
    policies = [policy for policy in preferred_policies if policy in pivot.columns]
    policies.extend([policy for policy in pivot.columns if policy not in policies])
    pivot = pivot.loc[:, policies]
    sort_policy = "All" if "All" in pivot.columns else policies[0]
    pivot = pivot.sort_values(sort_policy, ascending=False)

    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    color_top = float(np.nanmax(finite_values)) * 1.08 if finite_values.size else 1.0

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=color_top)
    ax.set_xticks(np.arange(len(policies)), policies)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Real-world 2026 policy")
    ax.set_ylabel("")
    ax.set_title(f"Real-world {_metric_label(metric)} by policy")
    ax.grid(False)

    fmt = "{:.2f}" if metric != "auprc_lift" else "{:.1f}"
    threshold = float(np.nanmax(values)) * 0.58 if np.isfinite(values).any() else 0.5
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            color = "white" if value >= threshold else COLORS["neutral"]
            ax.text(col_idx, row_idx, fmt.format(value), ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(_metric_label(metric))
    return save_report_figure(fig, path)


def _plot_metric_real_by_policy_bars(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Compare one real-cohort metric across policies using grouped bars."""
    _setup_style()
    pivot = df.pivot(index="label", columns="policy_label", values=f"real_{metric}")
    preferred_policies = ["All", "New", "Readmitted"]
    policies = [policy for policy in preferred_policies if policy in pivot.columns]
    policies.extend([policy for policy in pivot.columns if policy not in policies])
    pivot = pivot.loc[:, policies]
    sort_policy = "All" if "All" in pivot.columns else policies[0]
    pivot = pivot.sort_values(sort_policy, ascending=False)

    models = pivot.index.tolist()
    x = np.arange(len(models))
    width = min(0.80 / max(len(policies), 1), 0.28)

    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    for idx, policy in enumerate(policies):
        values = pd.to_numeric(pivot[policy], errors="coerce").to_numpy()
        offset = (idx - (len(policies) - 1) / 2) * width
        color = POLICY_COLORS[idx % len(POLICY_COLORS)]
        bars = ax.bar(x + offset, values, width, label=policy, color=color)
        _annotate_bars(ax, bars, "{:.2f}" if metric != "auprc_lift" else "{:.1f}", 0.012)

    ymax = float(np.nanmax(pivot.to_numpy(dtype=float))) if pivot.size else 1.0
    ax.set_ylim(0, min(1.0, max(0.12, ymax * 1.23)))
    ax.set_xticks(x, models, rotation=0, ha="center")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Real-world {_metric_label(metric)} by policy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(policies), frameon=False)
    ax.grid(axis="y", alpha=0.28)
    ax.grid(axis="x", visible=False)
    return save_report_figure(fig, path)


def _plot_row_vs_episode(df: pd.DataFrame, metric: str, path: Path) -> str:
    _setup_style()
    policies = df["policy_label"].drop_duplicates().tolist()
    models = df["label"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, len(policies), figsize=(13.4, 5.0), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    # Row-level performance evaluates daily predictions; episode-level performance
    # asks whether an episode contains at least one correctly identified event.
    for ax, policy in zip(axes, policies):
        subset = df.loc[df["policy_label"] == policy].set_index("label").reindex(models)
        row = pd.to_numeric(subset[f"real_{metric}"], errors="coerce").to_numpy()
        episode = pd.to_numeric(subset[f"real_episode_{metric}"], errors="coerce").to_numpy()
        row_bars = ax.bar(x - width / 2, row, width, label="Next-day row", color=COLORS[SPLIT_REAL])
        bars_epi = ax.bar(x + width / 2, episode, width, label="Episode", color=COLORS[LEVEL_EPISODE])
        _annotate_bars(ax, row_bars, "{:.2f}", 0.012)
        _annotate_bars(ax, bars_epi, "{:.2f}", 0.012)
        ax.set_title(policy)
        ax.set_xticks(x, models, rotation=0, ha="center")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(_metric_label(metric))
    fig.suptitle(f"Real {_metric_label(metric)}: row vs episode", fontsize=15, fontweight="bold")
    return save_report_figure(fig, path)


def _plot_best_summary(df: pd.DataFrame, path: Path) -> str:
    _setup_style()
    # Select the best model per policy using real-cohort AUPRC, then AUROC.
    best = (
        df.sort_values(["real_policy", "real_auprc", "real_auroc"], ascending=[True, False, False])
        .groupby("real_policy", as_index=False)
        .first()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    metrics = [
        ("real_auprc", "Real AUPRC", "{:.2f}", COLORS[SPLIT_REAL]),
        ("real_auroc", "Real AUROC", "{:.2f}", COLORS[SPLIT_TEST]),
        ("real_auprc_lift", "Real AUPRC lift", "{:.1f}", COLORS["lift"]),
    ]
    labels = best["policy_label"].tolist()
    x = np.arange(len(best))
    for ax, (col, label, fmt, color) in zip(axes, metrics):
        values = pd.to_numeric(best[col], errors="coerce")
        bars = ax.bar(x, values, color=color)
        _annotate_bars(ax, bars, fmt, 0.012)
        ax.set_xticks(x, labels, rotation=0, ha="center")
        ax.set_ylim(0, _metric_ymax(values.to_numpy()))
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
        for xpos, model in zip(x, best["label"]):
            ax.text(xpos, 0.03, model, ha="center", va="bottom", fontsize=8, color="white")
    fig.suptitle("Best deep-learning model by real AUPRC", fontsize=15, fontweight="bold")
    return save_report_figure(fig, path)


def _plot_sensitivity_ppv_real(df: pd.DataFrame, path: Path) -> str:
    """Show real-cohort sensitivity and PPV to inspect the selected threshold."""
    _setup_style()
    policies = df["policy_label"].drop_duplicates().tolist()
    models = df["label"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, len(policies), figsize=(13.4, 5.0), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    # These metrics show the practical trade-off induced by the selected threshold.
    for ax, policy in zip(axes, policies):
        subset = df.loc[df["policy_label"] == policy].set_index("label").reindex(models)
        sens = pd.to_numeric(subset["real_sensitivity"], errors="coerce").to_numpy()
        ppv = pd.to_numeric(subset["real_ppv"], errors="coerce").to_numpy()
        bars_sens = ax.bar(x - width / 2, sens, width, label="Sensitivity", color=COLORS[SPLIT_REAL])
        bars_ppv = ax.bar(x + width / 2, ppv, width, label="PPV", color=COLORS[LEVEL_EPISODE])
        _annotate_bars(ax, bars_sens, "{:.2f}", 0.012)
        _annotate_bars(ax, bars_ppv, "{:.2f}", 0.012)
        ax.set_title(policy)
        ax.set_xticks(x, models, rotation=0, ha="center")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    fig.suptitle("Real cohort threshold performance", fontsize=15, fontweight="bold")
    return save_report_figure(fig, path)


def _num(value: object) -> float:
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if not pd.isna(value) else float("nan")


if __name__ == "__main__":
    main()

