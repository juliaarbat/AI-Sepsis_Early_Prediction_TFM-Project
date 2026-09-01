from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS_CLASSICS_OUTPUTS_DIR
from src.classic_models_24h import CLASSIC_MODEL_FILES, TRAINED_STATUSES
from src.figure_style import PALETTE, save_report_figure
from src.output_contracts import METRIC_AUPRC, SPLIT_REAL, SPLIT_TEST
from src.plot_utils import (
    annotate_bars as _annotate_bars,
    clear_pngs as _clear_pngs,
    metric_label as _metric_label,
    setup_report_style as _setup_style,
)
from src.real_policies import real_policy_labels
from src.progress import log_end, log_start, step


POLICIES = real_policy_labels(short=False)
POLICY_SHORT_LABELS = real_policy_labels(short=True)

COMPARISON_DIR = MODELS_CLASSICS_OUTPUTS_DIR / "comparison"
FIGURES_DIR = COMPARISON_DIR / "figures"
INDEX_PATH = COMPARISON_DIR / "classic_models_24h_comparison_index.json"
COMPARISON_PATH = COMPARISON_DIR / "classic_models_24h_comparison.csv"

COLORS = {
    SPLIT_TEST: PALETTE["blue"],
    SPLIT_REAL: PALETTE["orange"],
    "real_alt": PALETTE["teal"],
    "accent": PALETTE["purple"],
    "neutral": PALETTE["muted"],
    "soft": PALETTE["panel"],
}
POLICY_COLORS = [PALETTE["orange"], PALETTE["teal"], PALETTE["blue"], PALETTE["purple"]]

COMPARISON_METRICS = {
    "test_auprc": "test_auprc",
    "real_auprc": "real_auprc",
    "test_episode_auprc": "test_episode_auprc",
    "real_episode_auprc": "real_episode_auprc",
    "test_auroc": "test_auroc",
    "real_auroc": "real_auroc",
    "test_episode_auroc": "test_episode_auroc",
    "real_episode_auroc": "real_episode_auroc",
    "test_ppv": "test_ppv",
    "real_ppv": "real_ppv",
    "test_sensitivity": "test_sensitivity",
    "real_sensitivity": "real_sensitivity",
    "test_episode_ppv": "test_episode_ppv",
    "real_episode_ppv": "real_episode_ppv",
    "test_episode_sensitivity": "test_episode_sensitivity",
    "real_episode_sensitivity": "real_episode_sensitivity",
}

OPTIONAL_COMPARISON_METRICS = {
    "test_prevalence": "test_prevalence",
    "real_prevalence": "real_prevalence",
    "test_auprc_lift": "test_auprc_lift",
    "real_auprc_lift": "real_auprc_lift",
    "test_episode_prevalence": "test_episode_prevalence",
    "real_episode_prevalence": "real_episode_prevalence",
    "test_episode_auprc_lift": "test_episode_auprc_lift",
    "real_episode_auprc_lift": "real_episode_auprc_lift",
}


def main() -> None:
    """Build CSV comparisons and figures from completed classic-model runs."""
    title = "classic model comparison figures"
    log_start(title)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Remove stale figures so the index describes only the current comparison.
    _clear_pngs(FIGURES_DIR)

    with step("Load results", number=1, total=2):
        # Only completed models are included in the cross-policy comparison.
        results = _load_results()
        comparison = _build_comparison(results)
        comparison.to_csv(COMPARISON_PATH, index=False)

    # Generate a small set of complementary views rather than one figure per metric.
    figures: dict[str, str] = {}
    with step("Generate curated comparison figures", number=2, total=2):
        figures["real_auprc_by_policy"] = _plot_real_metric_comparison(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_01_real_auprc_by_policy.png",
        )
        figures["real_auprc_by_policy_bars"] = _plot_real_metric_grouped_bar_comparison(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_01b_real_auprc_by_policy_bars.png",
        )
        figures["best_models_summary"] = _plot_best_models_summary(
            comparison,
            path=FIGURES_DIR / "comparison_02_best_models_summary.png",
        )
        figures["real_auprc_row_vs_episode"] = _plot_real_level_comparison(
            comparison,
            metric=METRIC_AUPRC,
            path=FIGURES_DIR / "comparison_03_real_auprc_row_vs_episode.png",
        )
        figures["real_sensitivity_ppv"] = _plot_real_sensitivity_ppv(
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
    print("Index:", INDEX_PATH)
    print("Comparison:", COMPARISON_PATH)
    log_end(title)


def _load_results() -> pd.DataFrame:
    """Load trained-model metrics for every configured policy."""
    frames: list[pd.DataFrame] = []
    for policy_key, policy_label in POLICIES.items():
        path = _policy_result_path(policy_key, "results")
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        df = pd.read_csv(path)
        # Keep the policy labels with each row so results remain identifiable after concatenation.
        df = df.loc[df["status"].isin(TRAINED_STATUSES)].copy()
        df.insert(0, "real_policy", policy_key)
        df.insert(1, "policy_label", policy_label)
        df.insert(2, "policy_short_label", POLICY_SHORT_LABELS[policy_key])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _policy_result_path(policy_key: str, key: str) -> Path:
    """Return the expected result path for the current output structure."""
    return MODELS_CLASSICS_OUTPUTS_DIR / policy_key / CLASSIC_MODEL_FILES[key]


def _build_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Create one comparable row per model and policy."""
    rows = [_comparison_row(row) for _, row in results.iterrows()]
    return pd.DataFrame(rows)


def _comparison_row(row: pd.Series) -> dict[str, object]:
    """Normalize one model-result row for cross-policy comparison."""
    output = {
        "real_policy": row["real_policy"],
        "policy_label": row["policy_label"],
        "policy_short_label": row["policy_short_label"],
        "model": row["model"],
        "label": row["label"],
    }
    for output_col, source_col in COMPARISON_METRICS.items():
        output[output_col] = _num(row, source_col)
    for output_col, source_col in OPTIONAL_COMPARISON_METRICS.items():
        output[output_col] = _num_optional(row, source_col)

    # Positive deltas indicate better performance on the real cohort than on test.
    output["delta_auprc_real_test"] = output["real_auprc"] - output["test_auprc"]
    output["delta_episode_auprc_real_test"] = (
        output["real_episode_auprc"] - output["test_episode_auprc"]
    )
    output["delta_auroc_real_test"] = output["real_auroc"] - output["test_auroc"]
    output["delta_episode_auroc_real_test"] = (
        output["real_episode_auroc"] - output["test_episode_auroc"]
    )
    return output


def _plot_real_level_comparison(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Compare real-cohort row and episode metrics across policies."""
    _setup_style()
    policies = df["policy_short_label"].drop_duplicates().tolist()
    models = df["label"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    colors = {
        "row": COLORS[SPLIT_REAL],
        "episode": COLORS["accent"],
    }

    fig, axes = plt.subplots(1, len(policies), figsize=(14.0, 5.2), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    # Row-level and episode-level metrics answer different evaluation questions.
    for ax, policy in zip(axes, policies):
        subset = df.loc[df["policy_short_label"] == policy].set_index("label").reindex(models)
        row = pd.to_numeric(subset[f"real_{metric}"], errors="coerce").to_numpy()
        episode = pd.to_numeric(subset[f"real_episode_{metric}"], errors="coerce").to_numpy()
        row_bars = ax.bar(x - width / 2, row, width, label="D+1 row", color=colors["row"])
        episode_bars = ax.bar(x + width / 2, episode, width, label="Episode", color=colors["episode"])
        _annotate_bars(ax, row_bars, fmt="{:.2f}", dy=0.012)
        _annotate_bars(ax, episode_bars, fmt="{:.2f}", dy=0.012)
        ax.set_title(policy)
        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8, ncol=2)
    axes[0].set_ylabel(metric.upper())
    fig.suptitle(
        f"Real {metric.upper()}: row vs episode",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    return save_report_figure(fig, path)


def _plot_real_metric_comparison(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Compare one real-cohort metric across policies."""
    _setup_style()
    col = f"real_{metric}"
    pivot = df.pivot(index="label", columns="policy_short_label", values=col)
    preferred_policies = ["All", "New", "Readmitted"]
    policies = [policy for policy in preferred_policies if policy in pivot.columns]
    policies.extend([policy for policy in pivot.columns if policy not in policies])
    pivot = pivot.loc[:, policies]
    # Sort by the complete cohort when available so the main comparison is stable.
    sort_policy = "All" if "All" in pivot.columns else policies[0]
    pivot = pivot.sort_values(sort_policy, ascending=False)

    values = pivot.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    color_top = float(np.nanmax(finite_values)) * 1.08 if finite_values.size else 1.0
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=color_top)

    ax.set_xticks(np.arange(len(policies)), policies)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Real-world 2026 policy")
    ax.set_ylabel("")
    ax.set_title(f"Real-world {_metric_label(metric)} by policy")
    ax.grid(False)
    ax.tick_params(axis="x", top=False, bottom=True, labeltop=False, labelbottom=True)

    threshold = float(np.nanmax(values)) * 0.58 if np.isfinite(values).any() else 0.5
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            color = "white" if value >= threshold else COLORS["neutral"]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(_metric_label(metric))
    return save_report_figure(fig, path)


def _plot_real_metric_grouped_bar_comparison(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Compare one real-cohort metric across policies using grouped bars."""
    _setup_style()
    col = f"real_{metric}"
    pivot = df.pivot(index="label", columns="policy_short_label", values=col)
    preferred_policies = ["All", "New", "Readmitted"]
    policies = [policy for policy in preferred_policies if policy in pivot.columns]
    policies.extend([policy for policy in pivot.columns if policy not in policies])
    pivot = pivot.loc[:, policies]
    sort_policy = "All" if "All" in pivot.columns else policies[0]
    pivot = pivot.sort_values(sort_policy, ascending=False)

    models = pivot.index.tolist()
    x = np.arange(len(models))
    width = min(0.78 / max(len(policies), 1), 0.24)

    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    for idx, policy in enumerate(policies):
        values = pd.to_numeric(pivot[policy], errors="coerce").to_numpy(dtype=float)
        offset = (idx - (len(policies) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=policy, color=_policy_color(idx))
        _annotate_bars(ax, bars, fmt="{:.2f}", dy=0.012)

    ymax = float(np.nanmax(pivot.to_numpy(dtype=float))) if pivot.size else 1.0
    ax.set_ylim(0, min(1.0, max(0.12, ymax * 1.23)))
    ax.set_xticks(x, models, rotation=0, ha="center")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(f"Real-world {_metric_label(metric)} by policy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(policies), frameon=False)
    ax.grid(axis="y", alpha=0.28)
    ax.grid(axis="x", visible=False)
    return save_report_figure(fig, path)


def _plot_best_models_summary(df: pd.DataFrame, path: Path) -> str:
    """Summarize the best model per policy in one figure."""
    _setup_style()
    # Select the model with the highest real-cohort AUPRC, using AUROC as a tie-breaker.
    best = (
        df.sort_values(["real_policy", "real_auprc", "real_auroc"], ascending=[True, False, False])
        .groupby("real_policy", as_index=False)
        .first()
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    metrics = [
        ("real_auprc", "Real AUPRC", "{:.2f}", COLORS[SPLIT_REAL]),
        ("real_auroc", "Real AUROC", "{:.2f}", COLORS[SPLIT_TEST]),
        ("real_ppv", "Real PPV", "{:.2f}", COLORS["accent"]),
    ]
    labels = best["policy_short_label"].tolist()
    x = np.arange(len(best))
    for ax, (col, label, fmt, color) in zip(axes, metrics):
        bars = ax.bar(x, best[col], color=color)
        _annotate_bars(ax, bars, fmt=fmt, dy=0.012)
        ax.set_xticks(x, labels, rotation=0, ha="center")
        ax.set_ylim(0, max(1.0 if col != "real_ppv" else 0.35, float(best[col].max()) * 1.25))
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
        for xpos, model in zip(x, best["label"]):
            ax.text(xpos, 0.03, model, ha="center", va="bottom", fontsize=8, color="white")
    fig.suptitle("Best classic model by real AUPRC", fontsize=16, fontweight="bold", y=1.02)
    return save_report_figure(fig, path)


def _plot_real_sensitivity_ppv(df: pd.DataFrame, path: Path) -> str:
    """Show how the selected threshold behaves on the real cohort."""
    _setup_style()
    policies = df["policy_short_label"].drop_duplicates().tolist()
    models = df["label"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, len(policies), figsize=(14.0, 5.0), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    # Sensitivity and PPV show the practical trade-off at each model's selected threshold.
    for ax, policy in zip(axes, policies):
        subset = df.loc[df["policy_short_label"] == policy].set_index("label").reindex(models)
        sens = pd.to_numeric(subset["real_sensitivity"], errors="coerce").to_numpy()
        ppv = pd.to_numeric(subset["real_ppv"], errors="coerce").to_numpy()
        bars_sens = ax.bar(x - width / 2, sens, width, label="Sensitivity", color=COLORS[SPLIT_REAL])
        bars_ppv = ax.bar(x + width / 2, ppv, width, label="PPV", color=COLORS["accent"])
        _annotate_bars(ax, bars_sens, fmt="{:.2f}", dy=0.012)
        _annotate_bars(ax, bars_ppv, fmt="{:.2f}", dy=0.012)
        ax.set_title(policy)
        ax.set_xticks(x, models, rotation=0, ha="center")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8, ncol=2)
    axes[0].set_ylabel("Value")
    fig.suptitle("Real cohort threshold performance", fontsize=16, fontweight="bold", y=1.02)
    return save_report_figure(fig, path)


def _num(row: pd.Series, col: str) -> float:
    """Read a numeric value from a row and return NaN when missing."""
    return float(pd.to_numeric(row[col], errors="coerce"))


def _num_optional(row: pd.Series, col: str) -> float:
    """Read an optional numeric value from a row."""
    if col not in row.index:
        return float("nan")
    return float(pd.to_numeric(row[col], errors="coerce"))


def _policy_color(index: int) -> str:
    """Return a stable color for a policy position."""
    return POLICY_COLORS[index % len(POLICY_COLORS)]




if __name__ == "__main__":
    main()

