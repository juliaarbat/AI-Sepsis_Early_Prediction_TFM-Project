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
from src.figure_style import PALETTE, apply_report_style, save_report_figure
from src.real_policies import REAL_POLICIES


REAL_POLICIES_TO_COMPARE = REAL_POLICIES


def create_policy_comparison(
    output_dirs: dict[str, Path],
    summaries: dict[str, object],
    output_stem: str = "classic_models_24h_real_overlap",
) -> dict[str, Path]:
    """Create comparison outputs across trained classic-model real policies.

    Saved metrics and split audits are loaded to produce comparison tables,
    figures, and an index JSON. No models are trained here.
    """
    comparison = _load_results_comparison(output_dirs)
    splits = _load_split_comparison(output_dirs)

    comparison_dir = MODELS_CLASSICS_OUTPUTS_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / f"{output_stem}_comparison.csv"
    splits_path = comparison_dir / f"{output_stem}_splits.csv"
    summary_path = comparison_dir / f"{output_stem}_summary.json"
    figures_dir = comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(comparison_path, index=False)
    splits.to_csv(splits_path, index=False)

    figures = {
        "auprc_test_real": _plot_metric_test_real(
            comparison,
            metric="auprc",
            path=figures_dir / "01_auprc_test_real_by_policy.png",
        ),
        "auroc_test_real": _plot_metric_test_real(
            comparison,
            metric="auroc",
            path=figures_dir / "02_auroc_test_real_by_policy.png",
        ),
        "delta_auprc_real_test": _plot_delta_metric(
            comparison,
            metric="auprc",
            path=figures_dir / "03_delta_auprc_real_minus_test.png",
        ),
        "splits_rows": _plot_splits_count(
            splits,
            value_col="n_rows",
            ylabel="Rows",
            path=figures_dir / "04_split_size_by_policy.png",
        ),
        "splits_prevalence": _plot_splits_count(
            splits,
            value_col="prevalence",
            ylabel="Prevalence",
            path=figures_dir / "05_split_prevalence_by_policy.png",
        ),
    }
    figure_index_path = comparison_dir / f"{output_stem}_figures_index.json"
    with open(figure_index_path, "w", encoding="utf-8") as f:
        json.dump(figures, f, ensure_ascii=False, indent=2)

    summary_payload = {
        policy_key: {
            "output_dir": str(output_dirs[policy_key]),
            "cohort": summary["cohort"],
            "splits": summary["splits"],
            "filter_audit": summary.get("filter_audit", {}),
        }
        for policy_key, summary in summaries.items()
    }
    summary_payload["comparison"] = {
        "results": str(comparison_path),
        "splits": str(splits_path),
        "figures_index": str(figure_index_path),
        "figures": figures,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return {
        "comparison": comparison_path,
        "splits": splits_path,
        "summary": summary_path,
        "figures_index": figure_index_path,
    }


def _policy_labels() -> dict[str, str]:
    """Return short labels for real-cohort policies."""
    return {str(policy["key"]): str(policy["label"]) for policy in REAL_POLICIES_TO_COMPARE}


def _load_results_comparison(output_dirs: dict[str, Path]) -> pd.DataFrame:
    """Load classic-model result tables from several output folders."""
    rows: list[dict[str, object]] = []
    policy_labels = _policy_labels()
    for policy_key, output_dir in output_dirs.items():
        results = pd.read_csv(_classic_file(output_dir, "results"))
        trained = results.loc[results["status"].isin(TRAINED_STATUSES)].copy()
        for _, row in trained.iterrows():
            test_auroc = float(row["test_auroc"])
            test_auprc = float(row["test_auprc"])
            real_auroc = float(row["real_auroc"])
            real_auprc = float(row["real_auprc"])
            rows.append(
                {
                    "real_policy": policy_key,
                    "policy_label": policy_labels.get(policy_key, policy_key),
                    "model": row["model"],
                    "label": row["label"],
                    "test_auroc": test_auroc,
                    "test_auprc": test_auprc,
                    "real_auroc": real_auroc,
                    "real_auprc": real_auprc,
                    "real_sensitivity": float(row["real_sensitivity"]),
                    "real_ppv": float(row["real_ppv"]),
                    "delta_real_test_auroc": real_auroc - test_auroc,
                    "delta_real_test_auprc": real_auprc - test_auprc,
                }
            )
    return pd.DataFrame(rows)


def _load_split_comparison(output_dirs: dict[str, Path]) -> pd.DataFrame:
    """Load split-distribution tables from several output folders."""
    frames: list[pd.DataFrame] = []
    policy_labels = _policy_labels()
    for policy_key, output_dir in output_dirs.items():
        splits = pd.read_csv(_classic_file(output_dir, "split_audit"))
        splits.insert(0, "real_policy", policy_key)
        splits.insert(1, "policy_label", policy_labels.get(policy_key, policy_key))
        frames.append(splits)
    return pd.concat(frames, ignore_index=True)


def _classic_file(output_dir: Path, key: str) -> Path:
    """Return the expected classic-model output path."""
    return output_dir / CLASSIC_MODEL_FILES[key]


def _plot_metric_test_real(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Plot test and real values for one metric across policies."""
    apply_report_style()
    test_col = f"test_{metric}"
    real_col = f"real_{metric}"
    models = df["model"].drop_duplicates().tolist()
    series: list[tuple[str, np.ndarray]] = []
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    for policy_key in df["real_policy"].drop_duplicates().tolist():
        policy = df.loc[df["real_policy"] == policy_key].set_index("model")
        label = str(policy["policy_label"].iloc[0])
        series.append((f"{label} - test", policy.reindex(models)[test_col].to_numpy()))
        series.append((f"{label} - real", policy.reindex(models)[real_col].to_numpy()))

    x = np.arange(len(models))
    width = min(0.80 / max(len(series), 1), 0.18)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, (idx, (label, values)) in zip(offsets, enumerate(series)):
        bars = ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, labels=models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} test vs real by patient policy")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return save_report_figure(fig, path)


def _plot_delta_metric(df: pd.DataFrame, metric: str, path: Path) -> str:
    """Plot the real-minus-test gap for one metric."""
    apply_report_style()
    delta_col = f"delta_real_test_{metric}"
    models = df["model"].drop_duplicates().tolist()
    policies = df["real_policy"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = min(0.80 / max(len(policies), 1), 0.34)
    offsets = (np.arange(len(policies)) - (len(policies) - 1) / 2) * width
    colors = [PALETTE["orange"], PALETTE["blue"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (offset, policy_key) in enumerate(zip(offsets, policies)):
        subset = df.loc[df["real_policy"] == policy_key].set_index("model")
        values = subset.reindex(models)[delta_col].to_numpy()
        label = str(subset["policy_label"].iloc[0])
        ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)])
    ax.axhline(0, color=PALETTE["ink"], linewidth=1)
    ax.set_xticks(x, labels=models, rotation=20, ha="right")
    ax.set_ylabel(f"Delta real - test ({metric.upper()})")
    ax.set_title(f"Real performance change versus test ({metric.upper()})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return save_report_figure(fig, path)


def _plot_splits_count(df: pd.DataFrame, value_col: str, ylabel: str, path: Path) -> str:
    """Plot split sizes for each policy."""
    apply_report_style()
    split_order = ["train", "valid", "test", "real"]
    policies = df["real_policy"].drop_duplicates().tolist()
    x = np.arange(len(split_order))
    width = min(0.80 / max(len(policies), 1), 0.34)
    offsets = (np.arange(len(policies)) - (len(policies) - 1) / 2) * width
    colors = [PALETTE["green"], PALETTE["purple"]]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for idx, (offset, policy_key) in enumerate(zip(offsets, policies)):
        subset = df.loc[df["real_policy"] == policy_key].set_index("split")
        values = subset.reindex(split_order)[value_col].to_numpy()
        label = str(subset["policy_label"].iloc[0])
        ax.bar(x + offset, values, width, label=label, color=colors[idx % len(colors)])
    ax.set_xticks(x, labels=split_order)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by split and real-cohort policy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return save_report_figure(fig, path)




