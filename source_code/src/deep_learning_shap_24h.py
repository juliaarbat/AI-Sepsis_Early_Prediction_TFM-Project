from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

from src.data_loading import load_sepsis_model_with_sofa
from src.config import SOFA_VITALS_FFILL_LIMIT_DAYS, SOFA_LAB_FFILL_LIMIT_DAYS
from src.feature_utils import feature_to_variable, format_policy_label, safe_filename
from src.figure_style import PALETTE, apply_report_style, save_report_figure
from src.predictive_model_24h import Preprocessor, prepare_model_dataset_24h, transform_features
from src.temporal_model_24h import (
    MultiModalSequenceDataset,
    add_admission_diagnosis_features,
    _create_sequence_model_24h,
    _create_final_temporal_split,
    _filter_real_from_start_date,
    _import_torch,
    _preprocessor_frame,
)


EPISODE_MISSINGNESS_THRESHOLD = 0.50
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS

N_BACKGROUND = 64
N_EXPLAIN = 128
SHAP_NSAMPLES = 100
TOP_N = 20
SEED = 42


def calculate_deep_learning_shap(
    candidate: dict[str, object],
    df_sofa: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Calculate SHAP for one trained deep-learning run."""
    if df_sofa is None:
        # Use the same SOFA cohort definition as the trained model.
        df_sofa = load_sepsis_model_with_sofa(
            episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        )
    data = _prepare_shap_data(df_sofa, candidate)
    outputs = _calculate_and_save_shap(candidate, data, output_dir=output_dir)
    _print_shap_outputs(outputs)
    return outputs


def _print_shap_outputs(outputs: dict[str, str]) -> None:
    """Print the main SHAP artifacts with English labels."""
    print("SHAP table:", outputs["variable_importance"])
    print("SHAP top-variable figure:", outputs["top_variables_figure"])
    print("SHAP direction figure:", outputs["direction_figure"])
    print("Beeswarm SHAP:", outputs["beeswarm_top20"])
    print("SHAP summary:", outputs["summary"])


def _prepare_shap_data(df_sofa: pd.DataFrame, candidate: dict[str, object]) -> dict[str, object]:
    torch, _, _, _ = _import_torch()
    # Reuse the checkpoint preprocessors and split definitions from training.
    checkpoint = _load_checkpoint(Path(candidate["model_path"]), torch)
    summary = checkpoint["summary"]

    temporal_preprocessor = _preprocessor_from_dict(checkpoint["temporal_preprocessor"])
    static_preprocessor = _preprocessor_from_dict(checkpoint["static_preprocessor"])
    temporal_columns = list(summary["original_temporal_variables"])
    static_columns = list(summary["original_static_variables"])

    # Rebuild the same split used during training.
    df_model = add_admission_diagnosis_features(prepare_model_dataset_24h(df_sofa))
    proportions = _split_proportions_from_summary(summary)
    split_map, _ = _create_final_temporal_split(
        df_model,
        proportions=proportions,
        split_unit=str(summary.get("split_unit", "patient")),
        real_start_date=summary.get("real_start_date"),
        real_overlap_policy=str(summary.get("real_overlap_policy", candidate["policy"])),
    )
    df_model["split"] = df_model["Episodi"].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()
    df_model, _ = _filter_real_from_start_date(
        df_model,
        real_start_date=summary.get("real_start_date"),
        enabled=bool(
            summary.get("evaluate_real_from_real_start", False)
        ),
    )
    df_model = df_model.sort_values(["split", "Episodi", "data_index"], kind="stable").reset_index(drop=True)

    datasets = {}
    for split in ("train", "test", "real"):
        df_split = df_model.loc[df_model["split"] == split].copy().reset_index(drop=True)
        if df_split.empty:
            continue
        x_temporal = transform_features(
            _preprocessor_frame(df_split, temporal_columns),
            temporal_preprocessor,
        ).astype("float32")
        x_static = transform_features(
            _preprocessor_frame(df_split, static_columns),
            static_preprocessor,
        ).astype("float32")
        datasets[split] = MultiModalSequenceDataset(
            df_split,
            x_temporal,
            x_static,
            int(summary["max_sequence_len"]),
            torch,
        )

    explained_split = "real" if "real" in datasets else "test"
    if "train" not in datasets or explained_split not in datasets:
        raise ValueError("Train and real/test datasets are required to calculate SHAP.")

    return {
        "checkpoint": checkpoint,
        "summary": summary,
        "datasets": datasets,
        "explained_split": explained_split,
        "temporal_feature_names": list(checkpoint["temporal_feature_names"]),
        "static_feature_names": list(checkpoint["static_feature_names"]),
    }


def _calculate_and_save_shap(
    candidate: dict[str, object],
    data: dict[str, object],
    output_dir: Path | None = None,
) -> dict[str, str]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The shap package is missing. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    torch, nn, _, _ = _import_torch()
    summary = data["summary"]
    checkpoint = data["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct the exact architecture saved during training.
    model = _create_sequence_model_24h(
        model_type=str(summary["model_type"]),
        n_temporal_features=int(summary["n_temporal_features_per_day"]),
        n_static_features=int(summary["n_static_features"]),
        d_model=int(summary["d_model"]),
        n_heads=int(summary["n_heads"]),
        n_layers=int(summary["n_layers"]),
        dropout=float(summary["dropout"]),
        max_len=int(summary["max_sequence_len"]),
        recurrent_hidden_size=_optional_int(summary.get("recurrent_hidden_size")),
        recurrent_bidirectional=bool(summary.get("recurrent_bidirectional", False)),
        nn=nn,
        torch=torch,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapper = _ModelWithSigmoid(model).to(device)
    # Use training rows as background and the selected evaluation split as target data.
    background = _sample_dataset_tensors(data["datasets"]["train"], N_BACKGROUND, SEED, torch, device)
    explain_tensors = _sample_dataset_tensors(
        data["datasets"][data["explained_split"]],
        N_EXPLAIN,
        SEED + 1,
        torch,
        device,
    )

    # Explain predicted probabilities rather than raw model logits.
    explainer = shap.GradientExplainer(wrapper, list(background))
    shap_raw = explainer.shap_values(list(explain_tensors), nsamples=SHAP_NSAMPLES)
    shap_temporal, shap_static = _unpack_shap_values(shap_raw)

    importance = _aggregate_variable_importance(
        shap_temporal=shap_temporal,
        shap_static=shap_static,
        temporal_feature_names=data["temporal_feature_names"],
        static_feature_names=data["static_feature_names"],
    )

    out_dir = Path(output_dir) if output_dir is not None else DEEP_LEARNING_OUTPUTS_DIR / "shap_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = safe_filename(str(candidate.get("prefix", "deep_learning_24h")), fallback="deep_learning_24h")
    table_path = out_dir / f"{safe_prefix}_shap_variable_importance.csv"
    figure_path = out_dir / f"{safe_prefix}_shap_top20_variables.png"
    direction_path = out_dir / f"{safe_prefix}_shap_top20_variable_direction.png"
    beeswarm_path = out_dir / f"{safe_prefix}_shap_beeswarm_top20.png"
    summary_path = out_dir / f"{safe_prefix}_shap_summary.json"
    policy = str(candidate["policy"])
    policy_label = format_policy_label(policy)
    title_context = f"{candidate['model_key']} - policy: {policy_label}"

    importance.to_csv(table_path, index=False)
    _plot_top_variables(importance, figure_path, title_context)
    _plot_direction_variables(importance, direction_path, title_context)
    _plot_beeswarm_variables(
        shap_module=shap,
        shap_temporal=shap_temporal,
        shap_static=shap_static,
        x_temporal=explain_tensors[0].detach().cpu().numpy(),
        x_static=explain_tensors[1].detach().cpu().numpy(),
        temporal_feature_names=data["temporal_feature_names"],
        static_feature_names=data["static_feature_names"],
        importance=importance,
        title_context=title_context,
        path=beeswarm_path,
    )

    outputs = {
        "variable_importance": str(table_path),
        "top_variables_figure": str(figure_path),
        "direction_figure": str(direction_path),
        "beeswarm_top20": str(beeswarm_path),
        "summary": str(summary_path),
    }
    payload = {
        "objective": "SHAP interpretation of the trained deep-learning model",
        "note": (
            "SHAP is calculated on a small sample. Temporal variables are "
            "aggregated across the days in the window, and one-hot variables are "
            "aggregated back to the original variable name. Positive direction "
            "means increased predicted sepsis probability. The beeswarm shows the "
            "top 20 variables after aggregating SHAP values by sample and variable."
        ),
        "model_key": candidate["model_key"],
        "policy": policy,
        "policy_label": policy_label,
        "valid_auprc": candidate["valid_auprc"],
        "valid_auroc": candidate["valid_auroc"],
        "model_path": str(candidate["model_path"]),
        "metrics_path": str(candidate["metrics_path"]),
        "explained_split": data["explained_split"],
        "n_background_train": int(background[0].shape[0]),
        "n_explained_rows": int(explain_tensors[0].shape[0]),
        "shap_nsamples": SHAP_NSAMPLES,
        "outputs": outputs,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return outputs


class _ModelWithSigmoid(torch.nn.Module):
    """Wrapper so SHAP explains probability rather than logits."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, seq, static_x):
        # The dataset uses all-zero left padding. Rebuild the mask inside the
        # wrapper so SHAP differentiates only temporal and static variables.
        padding_mask = seq.abs().sum(dim=2).eq(0)
        logits = self.model(seq, padding_mask, static_x)
        return torch.sigmoid(logits).unsqueeze(1)


def _sample_dataset_tensors(dataset, n: int, seed: int, torch, device):
    rng = np.random.default_rng(seed)
    size = min(int(n), len(dataset))
    indices = np.sort(rng.choice(len(dataset), size=size, replace=False))

    seqs = []
    statics = []
    for idx in indices:
        seq, padding_mask, static_x, _, _ = dataset[int(idx)]
        seqs.append(seq)
        statics.append(static_x)

    return (
        torch.stack(seqs).to(device),
        torch.stack(statics).to(device),
    )


def _aggregate_variable_importance(
    shap_temporal: np.ndarray,
    shap_static: np.ndarray,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
) -> pd.DataFrame:
    # Collapse sequence days and encoded features back to clinical variables.
    shap_temporal = _squeeze_output_dim(np.asarray(shap_temporal))
    shap_static = _squeeze_output_dim(np.asarray(shap_static))

    rows: list[dict[str, object]] = []
    temporal_abs = np.abs(shap_temporal).mean(axis=(0, 1))
    temporal_signed = shap_temporal.mean(axis=(0, 1))
    temporal_positive_pct = 100 * (shap_temporal > 0).mean(axis=(0, 1))
    static_abs = np.abs(shap_static).mean(axis=0)
    static_signed = shap_static.mean(axis=0)
    static_positive_pct = 100 * (shap_static > 0).mean(axis=0)

    for feature, value, signed, positive_pct in zip(
        temporal_feature_names,
        temporal_abs,
        temporal_signed,
        temporal_positive_pct,
    ):
        rows.append(
            {
                "branch": "temporal",
                "variable": feature_to_variable(feature),
                "encoded_feature": feature,
                "mean_abs_shap": float(value),
                "mean_shap": float(signed),
                "positive_shap_pct_feature": float(positive_pct),
                "weighted_positive_pct": float(value * positive_pct),
            }
        )
    for feature, value, signed, positive_pct in zip(
        static_feature_names,
        static_abs,
        static_signed,
        static_positive_pct,
    ):
        rows.append(
            {
                "branch": "static",
                "variable": feature_to_variable(feature),
                "encoded_feature": feature,
                "mean_abs_shap": float(value),
                "mean_shap": float(signed),
                "positive_shap_pct_feature": float(positive_pct),
                "weighted_positive_pct": float(value * positive_pct),
            }
        )

    detail = pd.DataFrame(rows)
    grouped = (
        detail
        .groupby(["branch", "variable"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_shap=("mean_shap", "sum"),
            weighted_positive_pct=("weighted_positive_pct", "sum"),
            n_encoded_features=("encoded_feature", "nunique"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["mean_abs_shap"].sum())
    grouped["shap_importance_pct"] = (
        100 * grouped["mean_abs_shap"] / total if total > 0 else np.nan
    )
    grouped["positive_shap_pct"] = np.where(
        grouped["mean_abs_shap"] > 0,
        grouped["weighted_positive_pct"] / grouped["mean_abs_shap"],
        np.nan,
    )
    grouped["mean_direction"] = np.where(
        grouped["mean_shap"] >= 0,
        "increases_mean_predicted_risk",
        "decreases_mean_predicted_risk",
    )
    grouped = grouped.drop(columns=["weighted_positive_pct"])
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped


def _plot_top_variables(importance: pd.DataFrame, path: Path, title_context: str) -> None:
    apply_report_style()
    top = importance.head(TOP_N).copy()
    top["label"] = top["variable"] + " (" + top["branch"] + ")"
    top = top.sort_values("mean_abs_shap", ascending=True)
    total_importance = importance["mean_abs_shap"].sum()
    top["importance_pct"] = np.where(
        total_importance > 0,
        top["mean_abs_shap"] / total_importance * 100,
        0.0,
    )
    colors = top["branch"].map({"temporal": PALETTE["blue"], "static": PALETTE["teal"]}).fillna(PALETTE["muted"])

    fig, ax = plt.subplots(figsize=(11.2, max(7.2, 0.42 * len(top))), constrained_layout=True)
    bars = ax.barh(top["label"], top["mean_abs_shap"], color=colors)
    ax.set_title("Top SHAP predictors in the optimised Transformer model", pad=30, fontsize=18)
    ax.set_xlabel("Mean absolute SHAP importance", fontsize=13)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = float(top["mean_abs_shap"].max()) if not top.empty else 0.0
    ax.set_xlim(0, x_max * 1.16 if x_max > 0 else 1)
    for bar, pct in zip(bars, top["importance_pct"]):
        ax.text(
            bar.get_width() + x_max * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
        )

    legend_handles = [
        Patch(facecolor=PALETTE["blue"], label="Temporal"),
        Patch(facecolor=PALETTE["teal"], label="Static"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=10)
    fig.text(
        0.5,
        0.925,
        "Values indicate each variable's share of total mean absolute SHAP importance.",
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
    )
    save_report_figure(fig, path)


def _plot_direction_variables(importance: pd.DataFrame, path: Path, title_context: str) -> None:
    apply_report_style()
    top = importance.head(TOP_N).copy()
    top["label"] = top["variable"] + " (" + top["branch"] + ")"
    top = top.sort_values("mean_shap", ascending=True)
    colors = np.where(top["mean_shap"] >= 0, PALETTE["orange"], PALETTE["blue"])

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(top))))
    ax.barh(top["label"], top["mean_shap"], color=colors)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.9)
    ax.set_xlabel("Mean SHAP value (positive = higher predicted risk)")
    ax.set_title(f"Mean SHAP direction\n{title_context}")
    save_report_figure(fig, path)


def _plot_beeswarm_variables(
    shap_module,
    shap_temporal: np.ndarray,
    shap_static: np.ndarray,
    x_temporal: np.ndarray,
    x_static: np.ndarray,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
    importance: pd.DataFrame,
    title_context: str,
    path: Path,
) -> None:
    shap_plot, x_plot = _prepare_beeswarm_variables(
        shap_temporal=shap_temporal,
        shap_static=shap_static,
        x_temporal=x_temporal,
        x_static=x_static,
        temporal_feature_names=temporal_feature_names,
        static_feature_names=static_feature_names,
        importance=importance,
    )
    if x_plot.empty:
        return

    apply_report_style()
    plt.figure(figsize=(9, 7))
    shap_module.summary_plot(
        shap_plot,
        x_plot,
        plot_type="dot",
        max_display=min(TOP_N, x_plot.shape[1]),
        show=False,
    )
    plt.title(f"SHAP beeswarm top variables\n{title_context}")
    plt.tight_layout()
    save_report_figure(plt.gcf(), path)


def _prepare_beeswarm_variables(
    shap_temporal: np.ndarray,
    shap_static: np.ndarray,
    x_temporal: np.ndarray,
    x_static: np.ndarray,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
    importance: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    # Aggregate temporal values across days for a readable variable-level beeswarm.
    shap_temporal = _squeeze_output_dim(np.asarray(shap_temporal))
    shap_static = _squeeze_output_dim(np.asarray(shap_static))

    temporal_positions = _positions_per_variable(temporal_feature_names)
    static_positions = _positions_per_variable(static_feature_names)

    grouped_shap: list[np.ndarray] = []
    grouped_values: dict[str, np.ndarray] = {}
    for _, row in importance.head(TOP_N).iterrows():
        branch = str(row["branch"])
        variable = str(row["variable"])
        label = f"{variable} ({branch})"

        if branch == "temporal":
            positions = temporal_positions.get(variable, [])
            if not positions:
                continue
            grouped_shap.append(shap_temporal[:, :, positions].sum(axis=(1, 2)))
            grouped_values[label] = x_temporal[:, :, positions].mean(axis=(1, 2))
        elif branch == "static":
            positions = static_positions.get(variable, [])
            if not positions:
                continue
            grouped_shap.append(shap_static[:, positions].sum(axis=1))
            grouped_values[label] = x_static[:, positions].mean(axis=1)

    if not grouped_shap:
        return np.empty((x_temporal.shape[0], 0)), pd.DataFrame(index=np.arange(x_temporal.shape[0]))

    return np.column_stack(grouped_shap), pd.DataFrame(grouped_values)


def _positions_per_variable(feature_names: list[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for idx, feature in enumerate(feature_names):
        positions.setdefault(feature_to_variable(str(feature)), []).append(idx)
    return positions


def _load_checkpoint(path: Path, torch) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _preprocessor_from_dict(data: dict[str, object]) -> Preprocessor:
    return Preprocessor(
        numeric_columns=list(data["numeric_columns"]),
        categorical_levels={str(k): list(v) for k, v in dict(data["categorical_levels"]).items()},
        medians={str(k): float(v) for k, v in dict(data["medians"]).items()},
        means={str(k): float(v) for k, v in dict(data["means"]).items()},
        stds={str(k): float(v) for k, v in dict(data["stds"]).items()},
        max_missing_ratio=float(data["max_missing_ratio"]),
        excluded_high_missing_columns=list(data["excluded_high_missing_columns"]),
        missing_ratios={str(k): float(v) for k, v in dict(data["missing_ratios"]).items()},
    )


def _split_proportions_from_summary(summary: dict[str, object]) -> tuple[float, float, float]:
    split_info = dict(summary.get("split_temporal", {}))
    proportions = dict(split_info.get("target_proportions", {}))
    return (
        float(proportions.get("train", 0.70)),
        float(proportions.get("valid", 0.15)),
        float(proportions.get("test", 0.15)),
    )


def _unpack_shap_values(shap_raw):
    if isinstance(shap_raw, list):
        if len(shap_raw) == 2:
            return shap_raw
        if len(shap_raw) == 1 and isinstance(shap_raw[0], list):
            return shap_raw[0]
    if isinstance(shap_raw, tuple) and len(shap_raw) == 2:
        return shap_raw
    raise ValueError("Unexpected SHAP format. Temporal and static values could not be separated.")


def _squeeze_output_dim(values: np.ndarray) -> np.ndarray:
    if values.ndim >= 1 and values.shape[-1] == 1:
        return values[..., 0]
    return values


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)




