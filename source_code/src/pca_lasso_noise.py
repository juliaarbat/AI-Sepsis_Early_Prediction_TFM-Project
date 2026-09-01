"""Diagnose potentially noisy variables with PCA and LASSO.

This script is exploratory and does not replace the predictive pipeline:
- PCA checks which features contribute little to the leading components.
- Logistic LASSO checks which features are zero or have very small effects.

Both readings are combined to suggest variables for review, not to delete
clinical variables automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loading import load_sepsis_model_with_sofa
from src.config import (
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    OUTPUTS_DIR,
)
from src.figure_style import PALETTE, apply_report_style, save_report_figure
from src.real_policies import REAL_ALL_2026
from src.predictive_model_24h import (
    TARGET,
    fit_preprocessor,
    transform_features,
)
from src.temporal_model_24h import DIAGNOSTIC_INGRES_DERIVED_COLUMNS
from src.classic_models_24h import (
    create_chronological_episode_split,
    prepare_classic_model_data,
)


OUTPUT_DIR = OUTPUTS_DIR / "pca_lasso_noise"
RANDOM_STATE = 42
MAX_MISSING_RATIO = 0.80
# Keep the real-period definition aligned with the model evaluation pipeline.
REAL_START_DATE = "2026-01-01"
REAL_OVERLAP_POLICY = REAL_ALL_2026
SPLIT_UNIT = "patient"


@dataclass
class PcaLassoData:
    """Prepared data used by the PCA and LASSO diagnostic."""

    preparation_info: dict[str, object]
    split_info: dict[str, object]
    preprocessor: object
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    x_real: np.ndarray | None
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    y_real: np.ndarray | None
    feature_names: list[str]
    split_sizes: dict[str, int]


def main() -> None:
    """Run the complete PCA/LASSO diagnostic and write tables plus figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare one train-fitted representation for both diagnostics.
    data = prepare_pca_lasso_data()

    print("[pca-lasso] Fitting PCA...", flush=True)
    pca_tables = run_pca(data.x_train, data.feature_names)

    print("[pca-lasso] Fitting logistic LASSO...", flush=True)
    lasso_tables, lasso_metrics = run_lasso(
        x_train=data.x_train,
        y_train=data.y_train,
        x_valid=data.x_valid,
        y_valid=data.y_valid,
        x_test=data.x_test,
        y_test=data.y_test,
        x_real=data.x_real,
        y_real=data.y_real,
        feature_names=data.feature_names,
    )

    variable_summary = summarize_variables(
        feature_names=data.feature_names,
        pca_feature_scores=pca_tables["feature_scores"],
        lasso_feature_scores=lasso_tables["feature_scores"],
        original_missing_ratios=data.preprocessor.missing_ratios,
        high_missing_columns=data.preprocessor.excluded_high_missing_columns,
    )

    write_outputs(
        preparation_info=data.preparation_info,
        split_info=data.split_info,
        preprocessor=data.preprocessor,
        pca_tables=pca_tables,
        lasso_tables=lasso_tables,
        lasso_metrics=lasso_metrics,
        variable_summary=variable_summary,
        split_sizes=data.split_sizes,
    )

    print(f"[pca-lasso] Done. Outputs at: {OUTPUT_DIR}", flush=True)


def prepare_pca_lasso_data() -> PcaLassoData:
    """Load, split, preprocess, and transform the model-ready data."""
    print("[pca-lasso] Loading dataset with SOFA...", flush=True)
    df_sofa = load_sepsis_model_with_sofa(regenerate=False)

    print("[pca-lasso] Preparing rows and candidate variables...", flush=True)
    df_model, preparation_info = prepare_classic_model_data(df_sofa, exclude_microbiology=False)
    split_map, split_info = create_chronological_episode_split(
        df_model,
        split_unit=SPLIT_UNIT,
        real_start_date=REAL_START_DATE,
        real_overlap_policy=REAL_OVERLAP_POLICY,
    )
    df_model["split"] = df_model["Episodi"].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()

    train = df_model.loc[df_model["split"] == "train"].copy()
    valid = df_model.loc[df_model["split"] == "valid"].copy()
    test = df_model.loc[df_model["split"] == "test"].copy()
    real = df_model.loc[df_model["split"] == "real"].copy()

    # Fit preprocessing on train only to avoid validation/test information leakage.
    diagnostic_columns = [
        col for col in DIAGNOSTIC_INGRES_DERIVED_COLUMNS if col in train.columns
    ]
    preprocessor = fit_preprocessor(
        train,
        max_missing_ratio=MAX_MISSING_RATIO,
        force_categorical_columns=set(diagnostic_columns),
    )

    # Apply the train-fitted preprocessing consistently to every split.
    print("[pca-lasso] Transforming feature matrices...", flush=True)
    return PcaLassoData(
        preparation_info=preparation_info,
        split_info=split_info,
        preprocessor=preprocessor,
        x_train=transform_features(train, preprocessor),
        x_valid=transform_features(valid, preprocessor),
        x_test=transform_features(test, preprocessor),
        x_real=transform_features(real, preprocessor) if not real.empty else None,
        y_train=train[TARGET].astype(int).to_numpy(),
        y_valid=valid[TARGET].astype(int).to_numpy(),
        y_test=test[TARGET].astype(int).to_numpy(),
        y_real=real[TARGET].astype(int).to_numpy() if not real.empty else None,
        feature_names=preprocessor.feature_names,
        split_sizes={
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
            "real": int(len(real)),
        },
    )


def run_pca(x_train: np.ndarray, feature_names: list[str]) -> dict[str, pd.DataFrame]:
    """Fit PCA on train features and return component and feature tables."""
    # PCA is descriptive here; it is not used to reduce model inputs.
    n_components = min(30, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
    pca.fit(x_train)

    explained = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    k80 = int(np.searchsorted(explained["cumulative_explained_variance"], 0.80) + 1)
    k = min(max(k80, 2), n_components)
    weighted_loading = (pca.components_[:k] ** 2).T @ pca.explained_variance_ratio_[:k]
    max_abs_loading = np.abs(pca.components_[:k]).max(axis=0)
    feature_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "pca_weighted_loading_top_components": weighted_loading,
            "pca_max_abs_loading_top_components": max_abs_loading,
            "pca_components_used": k,
        }
    ).sort_values("pca_weighted_loading_top_components", ascending=True)

    top_loadings_rows = []
    for component_idx, component in enumerate(pca.components_[: min(10, n_components)], start=1):
        order = np.argsort(np.abs(component))[::-1][:15]
        for rank, idx in enumerate(order, start=1):
            top_loadings_rows.append(
                {
                    "component": component_idx,
                    "rank": rank,
                    "feature": feature_names[idx],
                    "loading": float(component[idx]),
                    "abs_loading": float(abs(component[idx])),
                }
            )

    return {
        "explained_variance": explained,
        "feature_scores": feature_scores,
        "top_loadings": pd.DataFrame(top_loadings_rows),
    }


def run_lasso(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_real: np.ndarray | None,
    y_real: np.ndarray | None,
    feature_names: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Fit a small L1 logistic-regression grid and score all splits."""
    # Choose regularisation with validation AUPRC, then report all splits.
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    tuning_rows = []
    best_model = None
    best_row = None

    for c_value in c_grid:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train)
        valid_score = model.predict_proba(x_valid)[:, 1]
        coef = model.coef_.ravel()
        row = {
            "C": c_value,
            "valid_auprc": float(average_precision_score(y_valid, valid_score)),
            "valid_auroc": safe_auroc(y_valid, valid_score),
            "n_features_non_zero": int(np.count_nonzero(np.abs(coef) > 1e-8)),
            "pct_features_non_zero": float(np.mean(np.abs(coef) > 1e-8)),
        }
        tuning_rows.append(row)
        if best_row is None or row["valid_auprc"] > best_row["valid_auprc"]:
            best_row = row
            best_model = model

    if best_model is None or best_row is None:
        raise RuntimeError("No LASSO model could be fitted.")

    coef = best_model.coef_.ravel()
    feature_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "lasso_coef": coef,
            "lasso_abs_coef": np.abs(coef),
            "lasso_selected": np.abs(coef) > 1e-8,
        }
    ).sort_values("lasso_abs_coef", ascending=True)

    metrics = {
        "best": best_row,
        "splits": {
            "train": score_split(best_model, x_train, y_train),
            "valid": score_split(best_model, x_valid, y_valid),
            "test": score_split(best_model, x_test, y_test),
        },
    }
    if x_real is not None and y_real is not None and len(y_real):
        metrics["splits"]["real"] = score_split(best_model, x_real, y_real)

    return {
        "tuning": pd.DataFrame(tuning_rows).sort_values("valid_auprc", ascending=False),
        "feature_scores": feature_scores,
    }, metrics


def summarize_variables(
    feature_names: list[str],
    pca_feature_scores: pd.DataFrame,
    lasso_feature_scores: pd.DataFrame,
    original_missing_ratios: dict[str, float],
    high_missing_columns: list[str],
) -> pd.DataFrame:
    """Aggregate encoded-feature diagnostics back to original variables."""
    # Combine one-hot and missingness features before flagging review candidates.
    summary = pca_feature_scores.merge(lasso_feature_scores, on="feature", how="outer")
    summary["variable"] = summary["feature"].map(base_variable_name)
    summary["is_missing_indicator"] = summary["feature"].str.endswith("__missing")
    summary["is_one_hot_level"] = summary.apply(
        lambda row: (not row["is_missing_indicator"]) and "__" in row["feature"],
        axis=1,
    )

    grouped = (
        summary.groupby("variable", dropna=False)
        .agg(
            n_features=("feature", "count"),
            n_lasso_selected=("lasso_selected", "sum"),
            max_lasso_abs_coef=("lasso_abs_coef", "max"),
            mean_lasso_abs_coef=("lasso_abs_coef", "mean"),
            min_pca_weighted_loading=("pca_weighted_loading_top_components", "min"),
            max_pca_weighted_loading=("pca_weighted_loading_top_components", "max"),
            mean_pca_weighted_loading=("pca_weighted_loading_top_components", "mean"),
            n_missing_indicators=("is_missing_indicator", "sum"),
            n_one_hot_levels=("is_one_hot_level", "sum"),
        )
        .reset_index()
    )
    grouped["missing_ratio_train_original"] = grouped["variable"].map(original_missing_ratios)
    grouped["excluded_before_model_high_missing"] = grouped["variable"].isin(high_missing_columns)

    lasso_threshold = grouped["max_lasso_abs_coef"].quantile(0.25)
    pca_threshold = grouped["max_pca_weighted_loading"].quantile(0.25)
    # This is a screening signal for review, not an automatic exclusion rule.
    grouped["possible_noise_flag"] = (
        (grouped["n_lasso_selected"] == 0)
        & (grouped["max_lasso_abs_coef"] <= lasso_threshold)
        & (grouped["max_pca_weighted_loading"] <= pca_threshold)
    )
    grouped["review_reason"] = np.select(
        [
            grouped["excluded_before_model_high_missing"],
            grouped["possible_noise_flag"],
            (grouped["n_lasso_selected"] == 0),
        ],
        [
            "Excluded before the model because train missingness is too high",
            "Low PCA loading and not selected by LASSO",
            "Not selected by LASSO",
        ],
        default="Kept or showing signal in at least one reading",
    )

    missing_only = [
        col for col in high_missing_columns if col not in set(grouped["variable"])
    ]
    if missing_only:
        grouped = pd.concat(
            [
                grouped,
                pd.DataFrame(
                    {
                        "variable": missing_only,
                        "n_features": 0,
                        "n_lasso_selected": 0,
                        "max_lasso_abs_coef": np.nan,
                        "mean_lasso_abs_coef": np.nan,
                        "min_pca_weighted_loading": np.nan,
                        "max_pca_weighted_loading": np.nan,
                        "mean_pca_weighted_loading": np.nan,
                        "n_missing_indicators": 0,
                        "n_one_hot_levels": 0,
                        "missing_ratio_train_original": [
                            original_missing_ratios.get(col) for col in missing_only
                        ],
                        "excluded_before_model_high_missing": True,
                        "possible_noise_flag": False,
                        "review_reason": "Excluded before the model because train missingness is too high",
                    }
                ),
            ],
            ignore_index=True,
        )

    return grouped.sort_values(
        ["possible_noise_flag", "excluded_before_model_high_missing", "max_lasso_abs_coef"],
        ascending=[False, False, True],
    )


def base_variable_name(feature: str) -> str:
    """Recover the original variable name from encoded feature names."""
    if feature.endswith("__missing"):
        return feature.removesuffix("__missing")
    if "__" in feature:
        return feature.split("__", 1)[0]
    return feature


def display_feature_name(feature: str, max_len: int = 42) -> str:
    """Return a short English-facing label for plot axes."""
    label = public_feature_name(feature)
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip() + "..."
    return label


def public_feature_name(feature: str) -> str:
    """Return the full English-facing name for exported feature tables."""
    label = str(feature)
    replacements = {
        "__missing": " missing",
        "grup_diagnostic_ingres": "Admission diagnosis group",
        "diagnostic_ingres_codi": "Admission diagnosis code",
        "diagnostic_ingres_prefix3": "Admission diagnosis prefix",
        "codi_servei_admissor": "Admission service",
        "edat": "Age",
        "sexe": "Sex",
        "porta_o2": "Oxygen therapy",
        "dispositius_invasius_previs": "Previous invasive devices",
        "dies_des_ingres": "Days since admission",
        "dia_relatiu": "Relative day",
        "hospitalitzacio_recent_90d": "Recent hospitalization 90d",
        "reingres_30d": "Readmission 30d",
        "cirurgia": "Surgery",
        "urgencia_cirurgia": "Urgent surgery",
        "temps_cirurgia": "Surgery time",
        "creatinina": "Creatinine",
        "previous_mean": "previous mean",
        "lactat_arterial": "Arterial lactate",
        "lactat_venos": "Venous lactate",
        "ph_arterial": "Arterial pH",
        "ph_venos": "Venous pH",
        "pao2_arterial": "Arterial PaO2",
        "pao2_venos": "Venous PaO2",
        "paco2_arterial": "Arterial PaCO2",
        "paco2_venos": "Venous PaCO2",
        "bicarbonat_arterial": "Arterial bicarbonate",
        "bicarbonat_venos": "Venous bicarbonate",
        "exc_base_arterial": "Arterial base excess",
        "exc_base_venos": "Venous base excess",
        "hematocrit": "Hematocrit",
        "hemoglobina": "Hemoglobin",
        "leucocits": "Leukocytes",
        "pct_neutrofils": "Neutrophil percentage",
        "granulocits_immadurs": "Immature granulocytes",
        "plaquetes": "Platelets",
        "temps_protrombina_pct": "Prothrombin time percentage",
        "procalcitonina": "Procalcitonin",
        "glucosa": "Glucose",
        "bilirubina_total": "Total bilirubin",
        "proteines_totals": "Total proteins",
        "troponina": "Troponin",
        "hemocultiu_positiu": "Positive blood culture",
        "hemocultiu_germen": "Blood culture organism",
        "hemocultiu_temps_positivitat_h": "Blood culture time to positivity h",
        "urocultiu_resultat": "Urine culture result",
        "aspirat_traqueal_germen": "Tracheal aspirate organism",
        "broncoaspirat_germen": "Bronchoaspirate organism",
        "bal_germen": "BAL organism",
        "ag_pneumococ": "Pneumococcal antigen",
        "ag_legionella": "Legionella antigen",
        "colonitzacio_previa_blee": "Previous ESBL colonization",
        "colonitzacio_previa_cre": "Previous CRE colonization",
        "colonitzacio_previa_mrsa": "Previous MRSA colonization",
        "colonitzacio_previa_vre": "Previous VRE colonization",
        "cultiu_positiu_previ_90d": "Previous positive culture 90d",
        "vasopressor_qualsevol": "Any vasopressor",
        "vasopressor_multiple": "Multiple vasopressors",
        "diagnostic_ingres": "Admission diagnosis",
        "font_admissio": "Admission source",
        "vasopressor_adrenalina": "Adrenaline vasopressor",
        "vasopressor_dopamina": "Dopamine vasopressor",
        "vasopressor_dobutamina": "Dobutamine vasopressor",
        "vasopressor_noradrenalina": "Noradrenaline vasopressor",
        "codi": "code",
        "Oncologiques": "Oncology",
        "Neurologiques": "Neurology",
        "Urgències": "Emergency",
        "Programat": "Scheduled",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("__", ": ").replace("_", " ")
    return " ".join(label.split())


def score_split(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Score one split with AUROC and AUPRC."""
    score = model.predict_proba(x)[:, 1]
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "auprc": float(average_precision_score(y, score)),
        "auroc": safe_auroc(y, score),
    }


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return AUROC, or NaN if the split contains only one class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def write_outputs(
    preparation_info: dict[str, object],
    split_info: dict[str, object],
    preprocessor,
    pca_tables: dict[str, pd.DataFrame],
    lasso_tables: dict[str, pd.DataFrame],
    lasso_metrics: dict[str, object],
    variable_summary: pd.DataFrame,
    split_sizes: dict[str, int],
) -> None:
    """Write diagnostic CSV files, JSON summary, and figures."""
    # Keep raw diagnostics and human-readable exports together for auditability.
    _clear_legacy_outputs()

    pca_tables["explained_variance"].to_csv(
        OUTPUT_DIR / "pca_explained_variance.csv", index=False, encoding="utf-8-sig"
    )
    _export_feature_table(pca_tables["feature_scores"]).to_csv(
        OUTPUT_DIR / "pca_feature_scores.csv", index=False, encoding="utf-8-sig"
    )
    _export_feature_table(pca_tables["top_loadings"]).to_csv(
        OUTPUT_DIR / "pca_top_loadings.csv", index=False, encoding="utf-8-sig"
    )
    lasso_tables["tuning"].to_csv(
        OUTPUT_DIR / "lasso_tuning.csv", index=False, encoding="utf-8-sig"
    )
    _export_feature_table(lasso_tables["feature_scores"]).to_csv(
        OUTPUT_DIR / "lasso_feature_scores.csv", index=False, encoding="utf-8-sig"
    )
    _export_variable_table(variable_summary).to_csv(
        OUTPUT_DIR / "variable_noise_summary.csv", index=False, encoding="utf-8-sig"
    )

    write_figures(pca_tables, lasso_tables, variable_summary)

    top_noise = _export_variable_table(
        variable_summary.loc[
            variable_summary["possible_noise_flag"],
            [
                "variable",
                "n_features",
                "max_lasso_abs_coef",
                "max_pca_weighted_loading",
                "missing_ratio_train_original",
            ],
        ].head(30)
    )
    summary = {
        "objective": "Exploration of potentially noisy variables with PCA and logistic LASSO",
        "target": TARGET,
        "split_sizes": split_sizes,
        "config": {
            "max_missing_ratio": MAX_MISSING_RATIO,
            "shared_episode_missingness_threshold": MODEL_EPISODE_MISSINGNESS_THRESHOLD,
            "real_start_date": REAL_START_DATE,
            "real_overlap_policy": REAL_OVERLAP_POLICY,
            "split_unit": SPLIT_UNIT,
        },
        "preprocessing": {
            "n_numeric_columns": len(preprocessor.numeric_columns),
            "n_categorical_columns": len(preprocessor.categorical_levels),
            "n_model_features": len(preprocessor.feature_names),
            "n_excluded_leakage_columns": len(preprocessor.excluded_leakage_columns),
            "excluded_leakage_columns": _public_names(preprocessor.excluded_leakage_columns),
            "n_excluded_high_missing_columns": len(preprocessor.excluded_high_missing_columns),
            "excluded_high_missing_columns": _public_names(preprocessor.excluded_high_missing_columns),
        },
        "preparation_info": preparation_info,
        "split_info": split_info,
        "lasso_metrics": lasso_metrics,
        "pca_first_components": pca_tables["explained_variance"].head(10).to_dict(orient="records"),
        "n_possible_noise_variables": int(variable_summary["possible_noise_flag"].sum()),
        "top_possible_noise_variables": top_noise.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "pca_lasso_noise_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_legacy_outputs() -> None:
    """Remove stale output files whose names used non-English terms."""
    for path in (OUTPUT_DIR / "pca_lasso_noise_resum.json",):
        if path.exists():
            path.unlink()


def _export_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a human-facing copy with English feature names."""
    out = df.copy()
    if "feature" in out.columns:
        out["feature"] = out["feature"].map(public_feature_name)
    if "variable" in out.columns:
        out["variable"] = out["variable"].map(public_feature_name)
    return out


def _export_variable_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a human-facing copy with English variable names."""
    out = df.copy()
    if "variable" in out.columns:
        out["variable"] = out["variable"].map(public_feature_name)
    return out


def _public_names(values: list[str]) -> list[str]:
    """Map raw source names to English names for JSON outputs."""
    return [public_feature_name(value) for value in values]


def write_figures(
    pca_tables: dict[str, pd.DataFrame],
    lasso_tables: dict[str, pd.DataFrame],
    variable_summary: pd.DataFrame,
) -> None:
    """Write the small set of figures that are useful for the noise review."""
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    explained = pca_tables["explained_variance"]
    ax.plot(
        explained["component"],
        explained["cumulative_explained_variance"],
        marker="o",
        color=PALETTE["blue"],
    )
    ax.axhline(0.80, color=PALETTE["orange"], linestyle="--", linewidth=1, label="80% reference")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA: cumulative explained variance")
    ax.legend(loc="lower right")
    save_report_figure(fig, OUTPUT_DIR / "pca_cumulative_variance.png")

    top_lasso = lasso_tables["feature_scores"].sort_values("lasso_abs_coef", ascending=False).head(15)
    labels = [display_feature_name(feature) for feature in top_lasso["feature"]]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(labels[::-1], top_lasso["lasso_abs_coef"][::-1], color=PALETTE["teal"])
    ax.set_xlabel("|LASSO coefficient|")
    ax.set_title("LASSO: strongest features")
    ax.grid(axis="y", visible=False)
    save_report_figure(fig, OUTPUT_DIR / "lasso_top_features.png")

    candidates = variable_summary.loc[variable_summary["possible_noise_flag"]].head(25)
    if not candidates.empty:
        table = candidates[
            [
                "variable",
                "n_features",
                "n_lasso_selected",
                "max_lasso_abs_coef",
                "max_pca_weighted_loading",
                "missing_ratio_train_original",
            ]
        ].copy()
        table["max_lasso_abs_coef"] = table["max_lasso_abs_coef"].map(lambda value: f"{value:.4g}")
        table["max_pca_weighted_loading"] = table["max_pca_weighted_loading"].map(
            lambda value: f"{value:.4g}"
        )
        table["variable"] = table["variable"].map(lambda value: display_feature_name(value, max_len=46))
        table["missing_ratio_train_original"] = table["missing_ratio_train_original"].map(
            lambda value: f"{100 * value:.1f}%" if pd.notna(value) else ""
        )
        table = table.rename(
            columns={
                "variable": "Variable",
                "n_features": "Features",
                "n_lasso_selected": "LASSO sel.",
                "max_lasso_abs_coef": "Max |coef LASSO|",
                "max_pca_weighted_loading": "Max PCA loading",
                "missing_ratio_train_original": "Train missing",
            }
        )

        fig_height = max(2.8, 0.45 * len(table) + 1.3)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis("off")
        ax.set_title(
            "Variables with low PCA and LASSO signal",
            fontsize=14,
            pad=14,
        )
        rendered_table = ax.table(
            cellText=table.values,
            colLabels=table.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.36, 0.10, 0.11, 0.15, 0.15, 0.13],
        )
        rendered_table.auto_set_font_size(False)
        rendered_table.set_fontsize(9)
        rendered_table.scale(1, 1.45)
        for (row, _col), cell in rendered_table.get_celld().items():
            if row == 0:
                cell.set_facecolor(PALETTE["blue"])
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(PALETTE["panel"])
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor(PALETTE["border"])
        save_report_figure(fig, OUTPUT_DIR / "noise_candidate_variables.png")


if __name__ == "__main__":
    main()


