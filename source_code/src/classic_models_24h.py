"""Classic models for predicting sepsis on the following day.

The file is organized as a top-to-bottom recipe:
1. prepare the modeling dataset;
2. create train/validation/test/real splits;
3. transform the predictors;
4. train a small set of classic machine-learning models;
5. save metrics, predictions, feature importances, and figures.

By default it uses a short, explainable grid. When `optuna_trials` is set,
Optuna tunes only the model requested by the caller; test and real cohorts are
used only for the final evaluation.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MODELS_CLASSICS_OUTPUTS_DIR
from .feature_utils import (
    add_rank_and_pct,
    aggregate_encoded_importance,
    feature_to_variable as _feature_to_variable,
    format_policy_label,
    safe_filename,
)
from .figure_style import PALETTE, apply_report_style, save_report_figure
from .output_contracts import SPLIT_REAL, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from .plot_utils import annotate_bars, clear_pngs as _clear_pngs
from .predictive_model_24h import (
    ID_COL,
    PATIENT_COL,
    TARGET,
    fit_preprocessor,
    calculate_metrics,
    calculate_episode_metrics,
    prepare_model_dataset_24h,
    summarize_cohort,
    summarize_splits,
    select_minimum_sensitivity_threshold,
    select_youden_threshold,
    transform_features,
)
from .temporal_model_24h import (
    DIAGNOSTIC_INGRES_DERIVED_COLUMNS,
    MICROBIOLOGY_COLUMNS,
    add_admission_diagnosis_features,
    audit_operational_microbiology,
)
from .real_policies import (
    REAL_ALL_2026,
    REAL_READMITTED_2026,
    normalize_real_policy,
    select_excluded_real_units,
    select_real_units,
)
from .split_utils import (
    normalize_split_unit as _normalize_split_unit,
    split_date as _split_date,
    split_unit as _shared_split_unit,
    summarize_split_audit,
    validate_train_valid_test_splits as _validate_splits,
)


DEFAULT_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
    "lightgbm",
)

SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
SPLIT_ORDER_WITH_REAL = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST, SPLIT_REAL)
TRAINED_STATUSES = {"trained"}

# Stable English filenames for every classic-model execution.
CLASSIC_MODEL_FILES = {
    "summary": "classic_models_24h_summary.json",
    "results": "classic_models_24h_results.csv",
    "predictions": "classic_models_24h_predictions.csv",
    "tuning": "classic_models_24h_tuning.csv",
    "cv_folds": "classic_models_24h_cv_folds.csv",
    "cv_summary": "classic_models_24h_cv_summary.csv",
    "split_audit": "classic_models_24h_split_audit.csv",
    "split_distribution": "classic_models_24h_split_distribution.csv",
    "feature_selection": "classic_models_24h_feature_selection.csv",
    "feature_importance": "classic_models_24h_feature_importance.csv",
    "variable_importance": "classic_models_24h_variable_importance.csv",
    "figures_index": "classic_models_24h_figures_index.json",
    "model_pickle": "classic_models_24h.pkl",
}

# These variables are very close to the SOFA calculation or the care process.
# They are excluded to reduce target leakage risk.
ANTI_LEAKAGE_EXCLUDED_VARIABLES = {
    "n_missing_sofa_components",
    "pct_missing_sofa_components_row",
    "sofa_respiratory_available",
    "sofa_coagulation_available",
    "sofa_hepatic_available",
    "sofa_neurologic_available",
    "sofa_renal_available",
    "sofa_cardiovascular_available",
}


@dataclass
class ModelSpec:
    """Define one model and the small set of configurations to try."""

    key: str
    label: str
    estimator: Any | None
    param_grid: list[dict[str, Any]]
    available: bool = True
    reason: str | None = None


@dataclass
class ClassicRunData:
    """Prepared data used by all classic-model runs."""

    df_model: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    real: pd.DataFrame
    preprocessor: Any
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    x_real: np.ndarray | None
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    y_real: np.ndarray | None
    split_audit: pd.DataFrame
    split_distribution: pd.DataFrame
    preparation_info: dict[str, object]
    split_info: dict[str, object]
    real_filter_info: dict[str, object]


@dataclass
class ClassicRunArtifacts:
    """Tables, figures, and fitted objects produced by one run."""

    results: pd.DataFrame
    tuning: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    variable_importance: pd.DataFrame
    feature_selection: pd.DataFrame
    cv_folds: pd.DataFrame
    cv_summary: pd.DataFrame
    figures: dict[str, str]
    shap_outputs: dict[str, dict[str, str]]
    fitted_models: dict[str, Any]


def train_and_evaluate_classic_models_24h(
    df_sofa: pd.DataFrame,
    output_dir: Path = MODELS_CLASSICS_OUTPUTS_DIR / "execucio_simple",
    max_missing_ratio: float = 0.80,
    model_keys: tuple[str, ...] = DEFAULT_MODELS,
    seed: int = 42,
    cv_folds: int | None = 10,
    split_proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_unit: str = "patient",
    real_start_date: str | pd.Timestamp | None = "2026-01-01",
    real_overlap_policy: str = REAL_ALL_2026,
    evaluate_real_from_real_start: bool = False,
    exclude_microbiology: bool = False,
    include_temporal_features: bool = True,
    optuna_trials: int | None = None,
    calculate_shap: bool = False,
    shap_sample_n: int = 512,
    shap_split: str = "real",
    generate_figures: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Train, evaluate, and write outputs for classic 24h models.

    The input cohort is prepared, split, optionally tuned with Optuna, and
    evaluated with comparable patient-day and episode-level metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the cohort once so every model uses identical rows, predictors and splits.
    run_data = _prepare_classic_run_data(
        df_sofa,
        max_missing_ratio=max_missing_ratio,
        split_proportions=split_proportions,
        split_unit=split_unit,
        real_start_date=real_start_date,
        real_overlap_policy=real_overlap_policy,
        evaluate_real_from_real_start=evaluate_real_from_real_start,
        exclude_microbiology=exclude_microbiology,
        include_temporal_features=include_temporal_features,
    )

    # Model specifications are created after preprocessing so class weighting
    # can use the training prevalence only.
    models = _create_model_specs(seed=seed, y_train=run_data.y_train, model_keys=model_keys)
    artifacts = _train_classic_model_specs(
        run_data=run_data,
        model_specs=models,
        output_dir=output_dir,
        seed=seed,
        optuna_trials=optuna_trials,
        calculate_shap=calculate_shap,
        shap_sample_n=shap_sample_n,
        shap_split=shap_split,
        generate_figures=generate_figures,
        real_overlap_policy=real_overlap_policy,
        max_missing_ratio=max_missing_ratio,
        cv_folds=cv_folds,
        verbose=verbose,
    )

    # Save all artefacts together so downstream reports use one consistent run.
    _write_classic_run_outputs(
        output_dir=output_dir,
        run_data=run_data,
        artifacts=artifacts,
    )
    summary = _build_classic_run_summary(
        output_dir=output_dir,
        run_data=run_data,
        artifacts=artifacts,
        optuna_trials=optuna_trials,
        cv_folds=cv_folds,
    )
    (output_dir / CLASSIC_MODEL_FILES["summary"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print("[classic models] Outputs written.", flush=True)
    return summary


def _prepare_classic_run_data(
    df_sofa: pd.DataFrame,
    max_missing_ratio: float,
    split_proportions: tuple[float, float, float],
    split_unit: str,
    real_start_date: str | pd.Timestamp | None,
    real_overlap_policy: str,
    evaluate_real_from_real_start: bool,
    exclude_microbiology: bool,
    include_temporal_features: bool,
) -> ClassicRunData:
    """Prepare splits and matrices once, then reuse them everywhere."""
    # Feature preparation is performed before splitting, while any learned
    # preprocessing parameters are fitted later on the training partition only.
    df_model, preparation_info = prepare_classic_model_data(
        df_sofa,
        exclude_microbiology=exclude_microbiology,
        include_temporal_features=include_temporal_features,
    )
    split_map, split_info = create_chronological_episode_split(
        df_model,
        proportions=split_proportions,
        split_unit=split_unit,
        real_start_date=real_start_date,
        real_overlap_policy=real_overlap_policy,
    )
    df_model["split"] = df_model[ID_COL].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()

    df_model, real_filter_info = filter_real_from_start_date(
        df_model,
        real_start_date=real_start_date,
        enabled=evaluate_real_from_real_start,
    )
    split_audit = audit_temporal_splits(df_model)
    split_distribution = split_audit.copy()

    train, valid, test, real = _split_model_frames(df_model)
    # The real cohort is kept separate from development data for external evaluation.
    _validate_splits(train, valid, test)

    # The preprocessor learns missingness, encodings, and scaling only from train.
    preprocessor = _fit_classic_preprocessor(train, max_missing_ratio)
    x_train = transform_features(train, preprocessor)
    x_valid = transform_features(valid, preprocessor)
    x_test = transform_features(test, preprocessor)
    x_real = transform_features(real, preprocessor) if not real.empty else None

    return ClassicRunData(
        df_model=df_model,
        train=train,
        valid=valid,
        test=test,
        real=real,
        preprocessor=preprocessor,
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        x_real=x_real,
        y_train=train[TARGET].astype(int).to_numpy(),
        y_valid=valid[TARGET].astype(int).to_numpy(),
        y_test=test[TARGET].astype(int).to_numpy(),
        y_real=real[TARGET].astype(int).to_numpy() if not real.empty else None,
        split_audit=split_audit,
        split_distribution=split_distribution,
        preparation_info=preparation_info,
        split_info=split_info,
        real_filter_info=real_filter_info,
    )


def _split_model_frames(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, test, and real data frames."""
    return (
        df_model.loc[df_model["split"] == "train"].copy(),
        df_model.loc[df_model["split"] == "valid"].copy(),
        df_model.loc[df_model["split"] == "test"].copy(),
        df_model.loc[df_model["split"] == "real"].copy(),
    )


def _fit_classic_preprocessor(train: pd.DataFrame, max_missing_ratio: float) -> Any:
    """Fit the shared tabular preprocessor on train only."""
    diagnostic_columns = [
        col for col in DIAGNOSTIC_INGRES_DERIVED_COLUMNS if col in train.columns
    ]
    return fit_preprocessor(
        train,
        max_missing_ratio=max_missing_ratio,
        force_categorical_columns=set(diagnostic_columns),
    )


def _train_classic_model_specs(
    run_data: ClassicRunData,
    model_specs: list[ModelSpec],
    output_dir: Path,
    seed: int,
    optuna_trials: int | None,
    calculate_shap: bool,
    shap_sample_n: int,
    shap_split: str,
    generate_figures: bool,
    real_overlap_policy: str,
    max_missing_ratio: float,
    cv_folds: int | None,
    verbose: bool,
) -> ClassicRunArtifacts:
    """Train the requested models and collect all output tables."""
    result_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    shap_outputs: dict[str, dict[str, str]] = {}
    fitted_models: dict[str, Any] = {}

    for spec in model_specs:
        if verbose:
            print(f"[classic models] Model: {spec.label} ({spec.key})", flush=True)

        if not spec.available:
            result_rows.append(_row_not_available(spec))
            tuning_rows.append(_tuning_not_available(spec))
            continue

        # Choose hyperparameters on validation before fitting the final model.
        best = _select_best_configuration(
            spec=spec,
            x_train=run_data.x_train,
            y_train=run_data.y_train,
            x_valid=run_data.x_valid,
            y_valid=run_data.y_valid,
            valid=run_data.valid,
            seed=seed,
            optuna_trials=optuna_trials,
            verbose=verbose,
        )
        tuning_rows.extend(best["tuning_rows"])

        if best["model"] is None:
            result_rows.append(_row_error(spec, str(best["error"])))
            continue
        if best.get("valid_score") is None:
            result_rows.append(_row_error(spec, "No validation score was produced."))
            continue

        # CV below should reuse the exact hyperparameters selected on validation.
        spec.param_grid = [dict(best["params"])]
        final_model = _fit_final_model(best["model"], run_data)
        fitted_models[spec.key] = final_model

        # Scores are retained for all partitions to support diagnostics and reports.
        score_train = _predict_proba(final_model, run_data.x_train)
        score_valid = _predict_proba(final_model, run_data.x_valid)
        score_test = _predict_proba(final_model, run_data.x_test)
        score_real = _predict_proba(final_model, run_data.x_real) if run_data.x_real is not None else None

        # Thresholds come from the validation model trained only on train.
        thresholds = _choose_thresholds(run_data.y_valid, np.asarray(best["valid_score"]))
        # Threshold-based metrics are evaluated only after the validation threshold
        # has been fixed, keeping test and real evaluation independent.
        result_row = _evaluate_model(
            spec=spec,
            params=best["params"],
            thresholds=thresholds,
            test=run_data.test,
            y_test=run_data.y_test,
            score_test=score_test,
            real=run_data.real,
            y_real=run_data.y_real,
            score_real=score_real,
        )
        result_rows.append(result_row)
        if verbose:
            _print_model_results(result_row)

        prediction_frames.append(_split_predictions(spec, run_data.train, score_train, "train"))
        prediction_frames.append(_split_predictions(spec, run_data.valid, score_valid, "valid"))
        prediction_frames.append(_split_predictions(spec, run_data.test, score_test, "test"))
        if score_real is not None:
            prediction_frames.append(_split_predictions(spec, run_data.real, score_real, "real"))

        importance_frames.append(
            _compute_model_importances(
                model=final_model,
                model_key=spec.key,
                label=spec.label,
                feature_names=run_data.preprocessor.feature_names,
            )
        )
        if calculate_shap:
            shap_outputs[spec.key] = _compute_and_save_classic_model_shap(
                model=final_model,
                model_key=spec.key,
                label=spec.label,
                feature_names=run_data.preprocessor.feature_names,
                x_train_valid=np.vstack([run_data.x_train, run_data.x_valid]),
                x_test=run_data.x_test,
                test=run_data.test,
                x_real=run_data.x_real,
                real=run_data.real,
                output_dir=output_dir,
                sample_n=shap_sample_n,
                preferred_split=shap_split,
                real_overlap_policy=real_overlap_policy,
                seed=seed,
            )

    cv_folds_df, cv_summary_df = _run_simple_cross_validation(
        df_model=run_data.df_model,
        model_specs=model_specs,
        preprocessor_max_missing=max_missing_ratio,
        cv_folds=cv_folds,
        seed=seed,
        verbose=verbose,
    )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    feature_importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["model", "label", "feature", "importancia"])
    )
    variable_importance = _aggregate_variable_importance(feature_importance)
    figures = {}
    if generate_figures:
        figures = _generate_classic_model_figures(
            results=results,
            split_distribution=run_data.split_distribution,
            predictions=predictions,
            variable_importance=variable_importance,
            output_dir=output_dir,
        )

    return ClassicRunArtifacts(
        results=results,
        tuning=pd.DataFrame(tuning_rows),
        predictions=predictions,
        feature_importance=feature_importance,
        variable_importance=variable_importance,
        feature_selection=_build_feature_table(run_data.preprocessor),
        cv_folds=cv_folds_df,
        cv_summary=cv_summary_df,
        figures=figures,
        shap_outputs=shap_outputs,
        fitted_models=fitted_models,
    )


def _fit_final_model(selected_model: Any, run_data: ClassicRunData) -> Any:
    """Retrain the selected configuration on train+valid before final scoring."""
    final_model = _clone_estimator(selected_model)
    # Test and real data remain untouched; train+valid are combined only after
    # hyperparameters and thresholds have been selected on validation.
    final_model.fit(
        np.vstack([run_data.x_train, run_data.x_valid]),
        np.concatenate([run_data.y_train, run_data.y_valid]),
    )
    return final_model


def _write_classic_run_outputs(
    output_dir: Path,
    run_data: ClassicRunData,
    artifacts: ClassicRunArtifacts,
) -> None:
    """Write the standard files consumed by downstream scripts."""
    # CSV files are the tabular contract; the pickle stores the fitted pipeline
    # and models needed for later predictions or interpretation.
    artifacts.results.to_csv(output_dir / CLASSIC_MODEL_FILES["results"], index=False)
    artifacts.predictions.to_csv(output_dir / CLASSIC_MODEL_FILES["predictions"], index=False)
    artifacts.tuning.to_csv(output_dir / CLASSIC_MODEL_FILES["tuning"], index=False)
    artifacts.cv_folds.to_csv(output_dir / CLASSIC_MODEL_FILES["cv_folds"], index=False)
    artifacts.cv_summary.to_csv(output_dir / CLASSIC_MODEL_FILES["cv_summary"], index=False)
    run_data.split_audit.to_csv(output_dir / CLASSIC_MODEL_FILES["split_audit"], index=False)
    run_data.split_distribution.to_csv(output_dir / CLASSIC_MODEL_FILES["split_distribution"], index=False)
    artifacts.feature_selection.to_csv(output_dir / CLASSIC_MODEL_FILES["feature_selection"], index=False)
    artifacts.feature_importance.to_csv(output_dir / CLASSIC_MODEL_FILES["feature_importance"], index=False)
    artifacts.variable_importance.to_csv(output_dir / CLASSIC_MODEL_FILES["variable_importance"], index=False)

    with open(output_dir / CLASSIC_MODEL_FILES["model_pickle"], "wb") as f:
        pickle.dump(
            {
                "preprocessor": run_data.preprocessor,
                "models": artifacts.fitted_models,
            },
            f,
        )


def _build_classic_run_summary(
    output_dir: Path,
    run_data: ClassicRunData,
    artifacts: ClassicRunArtifacts,
    optuna_trials: int | None,
    cv_folds: int | None,
) -> dict[str, object]:
    """Build the JSON summary written by every classic-model main."""
    return {
        "objective": "Simple comparison of classic models for next-day sepsis prediction",
        "output_dir": str(output_dir),
        "cohort": summarize_cohort(run_data.df_model),
        "splits": summarize_splits(run_data.df_model),
        "preparation": run_data.preparation_info,
        "temporal_split": run_data.split_info,
        "filter_audit": run_data.real_filter_info,
        "hyperparameter_tuning": {
            "method": "optuna" if optuna_trials is not None and optuna_trials > 0 else "simple_grid",
            "criterion": "Best validation AUPRC at D+1/row level",
            "saved_secondary_metrics": [
                "valid_auroc",
                "valid_episode_auprc",
                "valid_episode_auroc",
            ],
            "optuna_trials": optuna_trials,
            "path": str(output_dir / CLASSIC_MODEL_FILES["tuning"]),
        },
        "robustness_cv": {
            "method": "patient-grouped folds within the development cohort",
            "configuration": "final hyperparameters selected by validation/Optuna",
            "cv_folds": cv_folds,
            "path_folds": str(output_dir / CLASSIC_MODEL_FILES["cv_folds"]),
            "summary_path": str(output_dir / CLASSIC_MODEL_FILES["cv_summary"]),
        },
        "n_features": len(run_data.preprocessor.feature_names),
        "excluded_sofa_leakage_variable_count": len(run_data.preprocessor.excluded_leakage_columns),
        "excluded_sofa_leakage_variables": run_data.preprocessor.excluded_leakage_columns,
        "models": artifacts.results.to_dict(orient="records"),
        "outputs": _standard_classic_output_paths(output_dir, artifacts.shap_outputs),
        "figures": artifacts.figures,
        "shap": artifacts.shap_outputs,
    }


def _standard_classic_output_paths(
    output_dir: Path,
    shap_outputs: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Return stable output names used by reports and comparison scripts."""
    return {
        "results": str(output_dir / CLASSIC_MODEL_FILES["results"]),
        "predictions": str(output_dir / CLASSIC_MODEL_FILES["predictions"]),
        "tuning": str(output_dir / CLASSIC_MODEL_FILES["tuning"]),
        "cv_folds": str(output_dir / CLASSIC_MODEL_FILES["cv_folds"]),
        "cv_summary": str(output_dir / CLASSIC_MODEL_FILES["cv_summary"]),
        "split_audit": str(output_dir / CLASSIC_MODEL_FILES["split_audit"]),
        "split_distribution": str(output_dir / CLASSIC_MODEL_FILES["split_distribution"]),
        "feature_selection": str(output_dir / CLASSIC_MODEL_FILES["feature_selection"]),
        "feature_importance": str(output_dir / CLASSIC_MODEL_FILES["feature_importance"]),
        "variable_importance": str(output_dir / CLASSIC_MODEL_FILES["variable_importance"]),
        "figures_index": str(output_dir / CLASSIC_MODEL_FILES["figures_index"]),
        "model_pickle": str(output_dir / CLASSIC_MODEL_FILES["model_pickle"]),
        "shap": shap_outputs,
    }


def prepare_classic_model_data(
    df_sofa: pd.DataFrame,
    exclude_microbiology: bool,
    include_temporal_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prepare the classic-model table and return preparation metadata.

    This applies admission-diagnosis encoding, anti-leakage filtering,
    optional microbiology exclusion, and optional temporal-window features.
    """
    df_model = prepare_model_dataset_24h(df_sofa)
    df_model = add_admission_diagnosis_features(df_model)

    # Leakage filtering happens before train/validation/test splitting so every
    # cohort is evaluated with the same predictor set.
    df_model, anti_leakage = remove_anti_leakage_variables(df_model)
    df_model, microbiology_columns = remove_microbiology_variables(
        df_model,
        exclude_microbiology=exclude_microbiology,
    )
    microbiology_info = audit_operational_microbiology(df_model)
    if include_temporal_features:
        df_model, temporal_info = add_temporal_window_features(df_model)
    else:
        temporal_info = {
            "created_columns": [],
            "n_created_columns": 0,
            "enabled": False,
            "reason": "previous_day_only_ablation",
        }

    info = {
        "n_rows": int(len(df_model)),
        "anti_leakage_excluded_variables": anti_leakage,
        "microbiology_excluded_variables": microbiology_columns,
        "microbiology_audit": microbiology_info,
        "temporal_features": temporal_info,
    }
    return df_model, info


def create_chronological_episode_split(
    df_model: pd.DataFrame,
    proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    split_unit: str = "patient",
    real_start_date: str | pd.Timestamp | None = "2026-01-01",
    real_overlap_policy: str = REAL_ALL_2026,
) -> tuple[dict[object, str], dict[str, object]]:
    """Create chronological splits and return an episode-to-split map.

    If a real cohort exists, it is separated first. The remaining units are
    assigned to train, validation, and test by the first date of each patient
    or episode. When splitting by patient, the patient is kept entirely in one
    partition, even if some of that patient's records conceptually fall on
    different sides of a date boundary. This prioritizes patient-level
    separation over strict row-level date cutoffs.
    """
    # Splitting by patient prevents records from the same patient crossing
    # development partitions and reduces information leakage.
    split_unit = _normalize_split_unit(split_unit)
    real_overlap_policy = normalize_real_policy(real_overlap_policy)

    work = df_model.copy()
    work["_split_unit"] = _split_unit(work, split_unit)
    work["_data_split"] = _split_date(work)

    units = (
        work.groupby("_split_unit", dropna=False)
        .agg(
            data_min=("_data_split", "min"),
            data_max=("_data_split", "max"),
        )
        .reset_index()
    )

    real_units: set[object] = set()
    excluded_units: set[object] = set()
    if real_start_date is not None and units["data_min"].notna().any():
        cutoff_date = pd.Timestamp(real_start_date)
        indexed_units = units.set_index("_split_unit", drop=False)
        # The real cohort is separated before creating train/valid/test to keep
        # 2026 as an external temporal evaluation whenever requested.
        real_mask = select_real_units(
            indexed_units,
            policy=real_overlap_policy,
            real_start=cutoff_date,
            start_date_col="data_min",
            end_date_col="data_max",
        )
        real_units = set(indexed_units.loc[real_mask, "_split_unit"])
        excluded_mask = select_excluded_real_units(
            indexed_units,
            policy=real_overlap_policy,
            real_start=cutoff_date,
            start_date_col="data_min",
            end_date_col="data_max",
        )
        excluded_units = set(indexed_units.loc[excluded_mask, "_split_unit"]) - real_units

    dev_units = units.loc[
        ~units["_split_unit"].isin(real_units | excluded_units)
    ].copy()
    dev_units = dev_units.sort_values(["data_min", "_split_unit"], kind="mergesort")

    train_units, valid_units, test_units = _cut_units(dev_units["_split_unit"].tolist(), proportions)
    unit_to_split = {unit: "train" for unit in train_units}
    unit_to_split.update({unit: "valid" for unit in valid_units})
    unit_to_split.update({unit: "test" for unit in test_units})
    unit_to_split.update({unit: "real" for unit in real_units})

    split_map = (
        work[[ID_COL, "_split_unit"]]
        .drop_duplicates()
        .assign(split=lambda df: df["_split_unit"].map(unit_to_split))
        .dropna(subset=["split"])
        .set_index(ID_COL)["split"]
        .to_dict()
    )

    info = {
        "split_unit": split_unit,
        "real_overlap_policy": real_overlap_policy,
        "real_start_date": str(real_start_date) if real_start_date is not None else None,
        "train_units": len(train_units),
        "valid_units": len(valid_units),
        "test_units": len(test_units),
        "real_units": len(real_units),
        "excluded_units": len(excluded_units),
        "unit_exclusion_reason": (
            f"New 2026 patients excluded under the {REAL_READMITTED_2026} policy"
            if excluded_units
            else None
        ),
        "real_readmitted_units": int(
            (
                (units.set_index("_split_unit").loc[list(real_units), "data_min"] < pd.Timestamp(real_start_date))
                & (units.set_index("_split_unit").loc[list(real_units), "data_max"] >= pd.Timestamp(real_start_date))
            ).sum()
        )
        if real_units and real_start_date is not None
        else 0,
        "real_new_units": int(
            (units.set_index("_split_unit").loc[list(real_units), "data_min"] >= pd.Timestamp(real_start_date)).sum()
        )
        if real_units and real_start_date is not None
        else 0,
    }
    return split_map, info


def audit_temporal_splits(
    df_model: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize size, dates, and prevalence for each split."""
    return summarize_split_audit(df_model, SPLIT_ORDER_WITH_REAL).rename(
        columns={
            "n_rows": "n_rows",
            "n_files_positives": "n_positive_rows",
            "prevalence": "prevalence",
        }
    )


def remove_anti_leakage_variables(df_model: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove variables that may give overly direct clues about the target."""
    columns = [col for col in ANTI_LEAKAGE_EXCLUDED_VARIABLES if col in df_model.columns]
    return df_model.drop(columns=columns), sorted(columns)


def remove_microbiology_variables(
    df_model: pd.DataFrame,
    exclude_microbiology: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Allow the analysis to be repeated without microbiology variables."""
    if not exclude_microbiology:
        return df_model, []
    columns = [col for col in MICROBIOLOGY_COLUMNS if col in df_model.columns]
    return df_model.drop(columns=columns), sorted(columns)


def filter_real_from_start_date(
    df_model: pd.DataFrame,
    real_start_date: str | pd.Timestamp | None,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Optionally remove pre-cutoff rows from the real split.

    This is used when the real cohort should be evaluated only from the
    operational start date onward.
    """
    if not enabled or real_start_date is None or "real" not in set(df_model["split"]):
        return df_model, {"filtre_real_des_de_data_inici": False, "files_eliminades": 0}

    dates = _split_date(df_model)
    cutoff_date = pd.Timestamp(real_start_date)
    remove_mask = (df_model["split"] == "real") & (dates < cutoff_date)
    filtered = df_model.loc[~remove_mask].copy()
    return filtered, {
        "filtre_real_des_de_data_inici": True,
        "real_start_date": str(real_start_date),
        "files_eliminades": int(remove_mask.sum()),
    }


def add_temporal_window_features(
    df_model: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add simple historical means for selected numeric variables.

    For each episode, only previous or current rows are used so each row
    represents what would be known up to that point.
    """
    df = df_model.copy()
    candidates = [
        "SBP",
        "DBP",
        "TAM",
        "HR",
        "RESP",
        "O2SAT",
        "TEMP",
        "lactat_arterial",
        "lactat_venos",
        "leucocits",
        "pcr",
        "procalcitonina",
        "creatinina",
    ]
    present_columns = [col for col in candidates if col in df.columns]
    if not present_columns:
        return df, {"created_columns": []}

    sort_cols = [ID_COL]
    if "data_index" in df.columns:
        sort_cols.append("data_index")
    elif "dia_relatiu" in df.columns:
        sort_cols.append("dia_relatiu")
    df = df.sort_values(sort_cols, kind="mergesort").copy()

    created_columns: list[str] = []
    for col in present_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        # Expanding means use information available up to the current day only.
        df[f"{col}_previous_mean"] = (
            series.groupby(df[ID_COL]).expanding().mean().reset_index(level=0, drop=True)
        )
        created_columns.append(f"{col}_previous_mean")
    return df, {"created_columns": created_columns}


def _create_model_specs(seed: int, y_train: np.ndarray, model_keys: tuple[str, ...]) -> list[ModelSpec]:
    """Create available model specs. Optional packages are used only when installed."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    scale_pos_weight = _scale_pos_weight(y_train)
    specs: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            key="logistic_regression",
            label="Logistic regression",
            estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            param_grid=[{"C": 0.1}, {"C": 1.0}],
        ),
        "random_forest": ModelSpec(
            key="random_forest",
            label="Random forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
            param_grid=[
                {"max_depth": None, "min_samples_leaf": 10},
                {"max_depth": 10, "min_samples_leaf": 20},
            ],
        ),
    }

    specs["xgboost"] = _spec_xgboost(seed, scale_pos_weight)
    specs["catboost"] = _spec_catboost(seed)
    specs["lightgbm"] = _spec_lightgbm(seed)

    return [specs.get(key, _unknown_spec(key)) for key in model_keys]


def _spec_xgboost(seed: int, scale_pos_weight: float) -> ModelSpec:
    """Build the XGBoost model specification when the package is installed."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on the local environment.
        return _unavailable_spec("xgboost", "XGBoost", "xgboost", exc)
    return ModelSpec(
        key="xgboost",
        label="XGBoost",
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
        param_grid=[
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
        ],
    )


def _spec_catboost(seed: int) -> ModelSpec:
    """Build the CatBoost model specification when the package is installed."""
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover - depends on the local environment.
        return _unavailable_spec("catboost", "CatBoost", "catboost", exc)
    return ModelSpec(
        key="catboost",
        label="CatBoost",
        estimator=CatBoostClassifier(
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        param_grid=[
            {"iterations": 300, "depth": 4, "learning_rate": 0.05},
            {"iterations": 300, "depth": 6, "learning_rate": 0.03},
        ],
    )


def _spec_lightgbm(seed: int) -> ModelSpec:
    """Build the LightGBM model specification when the package is installed."""
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover - depends on the local environment.
        return _unavailable_spec("lightgbm", "LightGBM", "lightgbm", exc)
    return ModelSpec(
        key="lightgbm",
        label="LightGBM",
        estimator=LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        ),
        param_grid=[
            {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
            {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.03},
        ],
    )


def _select_best_configuration(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid: pd.DataFrame,
    seed: int,
    optuna_trials: int | None,
    verbose: bool,
) -> dict[str, object]:
    """Train each candidate and keep the one with the best validation AUPRC."""
    if optuna_trials is not None and optuna_trials > 0:
        return _select_best_optuna_configuration(
            spec=spec,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            valid=valid,
            seed=seed,
            optuna_trials=int(optuna_trials),
            verbose=verbose,
        )

    best_model = None
    best_params: dict[str, Any] = {}
    best_valid_score: np.ndarray | None = None
    best_score = (-np.inf, -np.inf)
    best_error: str | None = None
    rows: list[dict[str, object]] = []

    for idx, params in enumerate(spec.param_grid, start=1):
        started = time.time()
        if verbose:
            print(
                f"[classic models]   Trial {idx}/{len(spec.param_grid)}: {params}",
                flush=True,
            )
        try:
            model = _clone_estimator(spec.estimator)
            model.set_params(**params)
            model.fit(x_train, y_train)
            score_valid = _predict_proba(model, x_valid)
            metrics = calculate_metrics(y_valid, score_valid, threshold=0.5)
            episode_metrics = calculate_episode_metrics(valid, score_valid, threshold=0.5)
            auprc = float(metrics["auprc"])
            auroc = float(metrics["auroc"])
            episode_auprc = float(episode_metrics["auprc"])
            episode_auroc = float(episode_metrics["auroc"])
            # AUPRC is the primary tuning metric because next-day sepsis is rare.
            # AUROC is used only as a tie-breaker between similar candidates.
            row = {
                "model": spec.key,
                "label": spec.label,
                "tuning_method": "simple_grid",
                "candidate": idx,
                "hyperparameters": json.dumps(params, ensure_ascii=False),
                "status": "ok",
                "valid_auprc": auprc,
                "valid_auroc": auroc,
                "valid_episode_auprc": episode_auprc,
                "valid_episode_auroc": episode_auroc,
                "seconds": round(time.time() - started, 3),
                "best": False,
            }
            rows.append(row)
            if verbose:
                print(
                    "[classic models]     "
                    f"valid AUPRC={_format_float(auprc)} "
                    f"AUROC={_format_float(auroc)} "
                    f"| episode AUPRC={_format_float(episode_auprc)} "
                    f"AUROC={_format_float(episode_auroc)} "
                    f"time={row['seconds']}s",
                    flush=True,
                )
            if (auprc, auroc) > best_score:
                best_score = (auprc, auroc)
                best_model = model
                best_params = params
                best_valid_score = score_valid
        except Exception as exc:  # pragma: no cover - depends on data/packages.
            best_error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "model": spec.key,
                    "label": spec.label,
                    "tuning_method": "simple_grid",
                    "candidate": idx,
                    "hyperparameters": json.dumps(params, ensure_ascii=False),
                    "status": "error",
                    "error": best_error,
                    "seconds": round(time.time() - started, 3),
                    "best": False,
                }
            )
            if verbose:
                print(f"[classic models]     ERROR {best_error}", flush=True)

    if best_model is not None:
        for row in rows:
            row["best"] = row.get("hyperparameters") == json.dumps(best_params, ensure_ascii=False)
        if verbose:
            print(f"[classic models]   Best configuration: {best_params}", flush=True)

    return {
        "model": best_model,
        "params": best_params,
        "valid_score": best_valid_score,
        "tuning_rows": rows,
        "error": best_error,
    }


def _select_best_optuna_configuration(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid: pd.DataFrame,
    seed: int,
    optuna_trials: int,
    verbose: bool,
) -> dict[str, object]:
    """Short, auditable Optuna run that optimizes validation AUPRC."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Add it with `pip install optuna` "
            "or run with optuna_trials=None."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows: list[dict[str, object]] = []
    best_model = None
    best_params: dict[str, Any] = {}
    best_valid_score: np.ndarray | None = None
    best_score = (-np.inf, -np.inf)
    best_error: str | None = None

    if verbose:
        print(
            f"[classic models]   Optuna: {optuna_trials} trials on validation AUPRC",
            flush=True,
        )

    def objective(trial) -> float:
        nonlocal best_model, best_params, best_valid_score, best_score, best_error
        started = time.time()
        params = _suggest_optuna_params(spec.key, trial)
        row = {
            "model": spec.key,
            "label": spec.label,
            "tuning_method": "optuna",
            "candidate": int(trial.number) + 1,
            "trial": int(trial.number),
            "hyperparameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "status": "ok",
            "valid_auprc": np.nan,
            "valid_auroc": np.nan,
            "valid_episode_auprc": np.nan,
            "valid_episode_auroc": np.nan,
            "seconds": np.nan,
            "best": False,
        }
        try:
            model = _clone_estimator(spec.estimator)
            model.set_params(**params)
            model.fit(x_train, y_train)
            score_valid = _predict_proba(model, x_valid)
            metrics = calculate_metrics(y_valid, score_valid, threshold=0.5)
            episode_metrics = calculate_episode_metrics(valid, score_valid, threshold=0.5)
            auprc = float(metrics["auprc"])
            auroc = float(metrics["auroc"])
            episode_auprc = float(episode_metrics["auprc"])
            episode_auroc = float(episode_metrics["auroc"])
            row.update(
                {
                    "valid_auprc": auprc,
                    "valid_auroc": auroc,
                    "valid_episode_auprc": episode_auprc,
                    "valid_episode_auroc": episode_auroc,
                    "seconds": round(time.time() - started, 3),
                }
            )
            rows.append(row)
            if verbose:
                print(
                    f"[classic models]     Trial {trial.number + 1}/{optuna_trials}: "
                    f"valid AUPRC={_format_float(auprc)} "
                    f"AUROC={_format_float(auroc)} "
                    f"| episode AUPRC={_format_float(episode_auprc)} "
                    f"AUROC={_format_float(episode_auroc)} params={params}",
                    flush=True,
                )
            if (auprc, auroc) > best_score:
                best_score = (auprc, auroc)
                best_model = model
                best_params = params
                best_valid_score = score_valid
            return auprc if np.isfinite(auprc) else 0.0
        except Exception as exc:  # pragma: no cover - depends on data/packages.
            best_error = f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "status": "error",
                    "error": best_error,
                    "seconds": round(time.time() - started, 3),
                }
            )
            rows.append(row)
            if verbose:
                print(f"[classic models]     ERROR {best_error}", flush=True)
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

    if best_model is not None:
        best_params_json = json.dumps(best_params, ensure_ascii=False, sort_keys=True)
        for row in rows:
            row["best"] = row.get("hyperparameters") == best_params_json
        if verbose:
            print(f"[classic models]   Best Optuna configuration: {best_params}", flush=True)

    return {
        "model": best_model,
        "params": best_params,
        "valid_score": best_valid_score,
        "tuning_rows": rows,
        "error": best_error,
    }


def _suggest_optuna_params(model_key: str, trial) -> dict[str, Any]:
    """Small search spaces so Optuna does not become a large black box."""
    if model_key == "logistic_regression":
        return {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
    if model_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
        }
    if model_key == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if model_key == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
        }
    return {}


def _choose_thresholds(y_valid: np.ndarray, score_valid: np.ndarray) -> dict[str, float]:
    """Choose thresholds using validation only."""
    return {
        "youden": select_youden_threshold(y_valid, score_valid),
        "sens_80": select_minimum_sensitivity_threshold(y_valid, score_valid, 0.80),
    }


def _evaluate_model(
    spec: ModelSpec,
    params: dict[str, Any],
    thresholds: dict[str, float],
    test: pd.DataFrame,
    y_test: np.ndarray,
    score_test: np.ndarray,
    real: pd.DataFrame,
    y_real: np.ndarray | None,
    score_real: np.ndarray | None,
) -> dict[str, object]:
    """Calculate the metrics consumed by figure scripts."""
    row: dict[str, object] = {
        "model": spec.key,
        "label": spec.label,
        "status": "trained",
        "best_hyperparameters": json.dumps(params, ensure_ascii=False),
        "threshold_valid": thresholds["youden"],
        "threshold_sensitivity_80_valid": thresholds["sens_80"],
    }
    row.update(_metrics_prefix("test", test, y_test, score_test, thresholds))

    if score_real is not None and y_real is not None and len(real) > 0:
        row.update(_metrics_prefix("real", real, y_real, score_real, thresholds))
    else:
        row.update(_empty_metrics("real"))
    return row


def _metrics_prefix(
    prefix: str,
    df_split: pd.DataFrame,
    y_true: np.ndarray,
    score: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, object]:
    """Calculate row-level and episode-level metrics."""
    base = calculate_metrics(y_true, score, thresholds["youden"])
    episode = calculate_episode_metrics(df_split, score, thresholds["youden"])

    ms80 = calculate_metrics(y_true, score, thresholds["sens_80"])

    return {
        f"{prefix}_auroc": base["auroc"],
        f"{prefix}_auprc": base["auprc"],
        f"{prefix}_prevalence": base["prevalence"],
        f"{prefix}_auprc_lift": base["auprc_lift"],
        f"{prefix}_sensitivity": base["sensitivity"],
        f"{prefix}_specificity": base["specificity"],
        f"{prefix}_ppv": base["ppv"],
        f"{prefix}_f1": base["f1"],
        f"{prefix}_sensitivity_at_sensitivity_80_threshold": ms80["sensitivity"],
        f"{prefix}_ppv_at_sensitivity_80_threshold": ms80["ppv"],
        f"{prefix}_episode_auroc": episode["auroc"],
        f"{prefix}_episode_auprc": episode["auprc"],
        f"{prefix}_episode_prevalence": episode["prevalence"],
        f"{prefix}_episode_auprc_lift": episode["auprc_lift"],
        f"{prefix}_episode_sensitivity": episode["sensitivity"],
        f"{prefix}_episode_ppv": episode["ppv"],
    }


def _run_simple_cross_validation(
    df_model: pd.DataFrame,
    model_specs: list[ModelSpec],
    preprocessor_max_missing: float,
    cv_folds: int | None,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a short patient/episode-grouped CV inside train+valid+test."""
    del verbose
    if cv_folds is None or cv_folds < 2:
        return pd.DataFrame(), pd.DataFrame()

    from sklearn.model_selection import KFold

    dev = df_model.loc[df_model["split"].isin(SPLIT_ORDER)].copy()
    if dev.empty:
        return pd.DataFrame(), pd.DataFrame()

    dev["_unitat_cv"] = _split_unit(dev, "patient")
    units = pd.Series(dev["_unitat_cv"].unique(), dtype="object")
    n_splits = min(cv_folds, len(units))
    if n_splits < 2:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    # This CV is a robustness check for the selected model settings. It is not
    # used to choose the final test or real-cohort metrics.
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(units), start=1):
        train_units = set(units.iloc[train_idx])
        valid_units = set(units.iloc[valid_idx])
        train = dev.loc[dev["_unitat_cv"].isin(train_units)].copy()
        valid = dev.loc[dev["_unitat_cv"].isin(valid_units)].copy()

        preprocessor = _fit_classic_preprocessor(train, preprocessor_max_missing)
        x_train = transform_features(train, preprocessor)
        x_valid = transform_features(valid, preprocessor)
        y_train = train[TARGET].astype(int).to_numpy()
        y_valid = valid[TARGET].astype(int).to_numpy()

        for spec in model_specs:
            if not spec.available:
                continue
            try:
                model = _clone_estimator(spec.estimator)
                model.set_params(**spec.param_grid[0])
                model.fit(x_train, y_train)
                score = _predict_proba(model, x_valid)
                threshold = select_youden_threshold(y_valid, score)
                metrics = calculate_metrics(y_valid, score, threshold)
                episode_metrics = calculate_episode_metrics(valid, score, threshold)
                rows.append(
                    {
                        "fold": fold,
                        "model": spec.key,
                        "label": spec.label,
                        "status": "ok",
                        "hyperparameters": json.dumps(spec.param_grid[0], ensure_ascii=False, sort_keys=True),
                        "n_valid": len(valid),
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "sensitivity": metrics["sensitivity"],
                        "ppv": metrics["ppv"],
                        "episode_auroc": episode_metrics["auroc"],
                        "episode_auprc": episode_metrics["auprc"],
                        "episode_sensitivity": episode_metrics["sensitivity"],
                        "episode_ppv": episode_metrics["ppv"],
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on the package/model.
                rows.append(
                    {
                        "fold": fold,
                        "model": spec.key,
                        "label": spec.label,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    folds = pd.DataFrame(rows)
    if folds.empty or "auprc" not in folds.columns:
        return folds, pd.DataFrame()
    summary = (
        folds.dropna(subset=["auprc"])
        .groupby(["model", "label"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            episode_auroc_mean=("episode_auroc", "mean"),
            episode_auroc_sd=("episode_auroc", "std"),
            episode_auprc_mean=("episode_auprc", "mean"),
            episode_auprc_sd=("episode_auprc", "std"),
            n_folds=("fold", "nunique"),
        )
    )
    return folds, summary


def generate_classic_model_figures_from_outputs(output_dir: Path) -> dict[str, str]:
    """Regenerate classic-model figures from saved tables, without retraining."""
    results = pd.read_csv(_classic_output_path(output_dir, "results"))
    predictions = pd.read_csv(_classic_output_path(output_dir, "predictions"))
    split_distribution = pd.read_csv(_classic_output_path(output_dir, "split_distribution"))

    variable_path = _classic_output_path(output_dir, "variable_importance", required=False)
    if variable_path.exists():
        variable_importance = pd.read_csv(variable_path)
    else:
        feature_importance = pd.read_csv(_classic_output_path(output_dir, "feature_importance"))
        variable_importance = _aggregate_variable_importance(feature_importance)
        variable_importance.to_csv(output_dir / CLASSIC_MODEL_FILES["variable_importance"], index=False)

    return _generate_classic_model_figures(
        results=results,
        split_distribution=split_distribution,
        predictions=predictions,
        variable_importance=variable_importance,
        output_dir=output_dir,
    )


def _classic_output_path(output_dir: Path, key: str, required: bool = True) -> Path:
    """Find a standard classic-model output file."""
    path = output_dir / CLASSIC_MODEL_FILES[key]
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"Missing classic-model output: {path}")
    return path


def _generate_classic_model_figures(
    results: pd.DataFrame,
    split_distribution: pd.DataFrame,
    predictions: pd.DataFrame,
    variable_importance: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path | None = None,
) -> dict[str, str]:
    """Generate a small set of clear figures and save a JSON index."""
    figures_dir = figures_dir or (output_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(figures_dir)

    figures: dict[str, str] = {}
    trained = results.loc[results.get("status").isin(TRAINED_STATUSES)].copy()
    if not trained.empty:
        figures["test_metric_comparison"] = _plot_metric_comparison(
            trained,
            figures_dir / "01_test_metrics_comparison.png",
        )
    if not split_distribution.empty:
        figures["split_distribution"] = _plot_split_distribution(
            split_distribution,
            figures_dir / "00_split_distribution.png",
        )
    best_model_key = _best_trained_model_key(trained)
    if best_model_key is not None:
        imp = variable_importance.loc[variable_importance["model"] == best_model_key]
        if not imp.empty:
            label = str(imp["label"].iloc[0])
            figures["best_model_top_variables"] = _plot_top_model_variables(
                imp,
                label,
                figures_dir / "02_best_model_top_variables.png",
            )

    with open(output_dir / CLASSIC_MODEL_FILES["figures_index"], "w", encoding="utf-8") as f:
        json.dump(figures, f, ensure_ascii=False, indent=2)
    return figures


def _plot_metric_comparison(results: pd.DataFrame, path: Path) -> str:
    """Save a compact test AUROC/AUPRC comparison figure."""
    _style()
    labels = results["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars_auroc = ax.bar(x - width / 2, results["test_auroc"], width, label="AUROC", color=PALETTE["blue"])
    bars_auprc = ax.bar(x + width / 2, results["test_auprc"], width, label="AUPRC", color=PALETTE["orange"])
    _annotate_bars(ax, bars_auroc, "{:.2f}", 0.012)
    _annotate_bars(ax, bars_auprc, "{:.2f}", 0.012)
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Classic models: test performance")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    return save_report_figure(fig, path)


def _best_trained_model_key(trained: pd.DataFrame) -> str | None:
    """Select the best trained model for the one interpretability figure."""
    if trained.empty or "model" not in trained.columns:
        return None
    metric = "real_auprc" if "real_auprc" in trained.columns else "test_auprc"
    if metric not in trained.columns:
        return str(trained.iloc[0]["model"])
    ranked = trained.copy()
    ranked["_figure_score"] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=["_figure_score"]).sort_values("_figure_score", ascending=False)
    if ranked.empty:
        return str(trained.iloc[0]["model"])
    return str(ranked.iloc[0]["model"])


def _plot_split_distribution(split_distribution: pd.DataFrame, path: Path) -> str:
    """Save split size and prevalence figures."""
    _style()
    df = split_distribution.set_index("split").reindex(SPLIT_ORDER_WITH_REAL).dropna(how="all").reset_index()
    split_labels = df["split"].map(
        {"train": "Train", "valid": "Validation", "test": "Test", "real": "Real 2026"}
    ).fillna(df["split"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    split_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["orange"]]
    row_counts = df["n_rows"]
    bars = axes[0].bar(split_labels, row_counts, color=split_colors[: len(df)])
    _annotate_bars(axes[0], bars, "{:.0f}", max(float(row_counts.max()) * 0.02, 1.0))
    axes[0].set_title("Rows")
    axes[0].set_ylabel("Rows")
    prevalence = df["prevalence"] * 100
    bars = axes[1].bar(split_labels, prevalence, color=split_colors[: len(df)])
    _annotate_bars(axes[1], bars, "{:.1f}%", max(float(prevalence.max()) * 0.03, 0.1))
    axes[1].set_title("Positive cases")
    axes[1].set_ylabel("% positives")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Dataset split overview", fontsize=14, fontweight="bold")
    return save_report_figure(fig, path)


def _plot_top_model_variables(importance: pd.DataFrame, label: str, path: Path) -> str:
    """Save the most important variables for one model."""
    _style()
    top = importance.sort_values("importancia", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.34 * len(top))))
    ax.barh(top["variable"][::-1], top["importancia"][::-1], color=PALETTE["purple"])
    ax.set_title(f"Most influential variables: {label}")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="y", visible=False)
    return save_report_figure(fig, path)


def _annotate_bars(ax: plt.Axes, bars, fmt: str, dy: float) -> None:
    """Add small centered value labels to bar plots."""
    annotate_bars(ax, bars, fmt, dy, color=PALETTE["ink"])


def _compute_and_save_classic_model_shap(
    model: Any,
    model_key: str,
    label: str,
    feature_names: list[str],
    x_train_valid: np.ndarray,
    x_test: np.ndarray,
    test: pd.DataFrame,
    x_real: np.ndarray | None,
    real: pd.DataFrame,
    output_dir: Path,
    sample_n: int,
    preferred_split: str,
    real_overlap_policy: str,
    seed: int,
) -> dict[str, str]:
    """Compute SHAP explanations and save tables and figures for one model."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on the environment.
        raise ImportError(
            "The shap package is missing. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    x_explicar, df_explicat, split_explicat = _select_classic_shap_split(
        preferred_split=preferred_split,
        x_test=x_test,
        test=test,
        x_real=x_real,
        real=real,
    )
    # Explain a reproducible sample to keep SHAP computationally manageable.
    x_sample, sample_indices = _sample_rows_array(x_explicar, sample_n, seed)
    x_background, _ = _sample_rows_array(x_train_valid, min(sample_n, 512), seed + 17)

    shap_values = _compute_classic_shap_values(
        shap_module=shap,
        model=model,
        x_sample=x_sample,
        x_background=x_background,
    )
    shap_values = _normalize_classic_shap_values(shap_values)
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "The number of SHAP values does not match the preprocessor features: "
            f"{shap_values.shape[1]} vs {len(feature_names)}"
        )

    # Aggregate one-hot levels and missingness indicators back to clinical variables.
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    feature_importance = _ensure_shap_feature_variables(feature_importance)
    variable_importance = aggregate_encoded_importance(
        feature_importance,
        value_col="mean_abs_shap",
        count_col="n_encoded_features",
    )
    variable_importance = add_rank_and_pct(
        variable_importance,
        value_col="mean_abs_shap",
        pct_col="shap_importance_pct",
    )

    safe_key = safe_filename(str(model_key))
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    table_path = shap_dir / f"{safe_key}_shap_variable_importance.csv"
    figure_path = shap_dir / f"{safe_key}_shap_top20_variables.png"
    beeswarm_path = shap_dir / f"{safe_key}_shap_beeswarm_top20.png"
    dependence_dir = shap_dir / "dependence"
    summary_path = shap_dir / f"{safe_key}_shap_summary.json"
    policy_label = format_policy_label(real_overlap_policy)
    title_context = f"{label} - policy: {policy_label}"

    variable_importance.to_csv(table_path, index=False)
    _plot_top_variables_shap_classic(variable_importance, title_context, figure_path)
    x_sample_df = pd.DataFrame(x_sample, columns=feature_names)
    _plot_beeswarm_shap_classic(
        shap_module=shap,
        shap_values=shap_values,
        x_sample=x_sample_df,
        feature_importance=feature_importance,
        top_variables=variable_importance["variable"].head(20).tolist(),
        label=title_context,
        path=beeswarm_path,
    )
    dependence_paths = _plot_dependence_shap_classic(
        shap_module=shap,
        shap_values=shap_values,
        x_sample=x_sample_df,
        feature_importance=feature_importance,
        label=title_context,
        output_dir=dependence_dir,
    )

    payload = {
        "objective": "SHAP interpretability for the trained classic model",
        "model_key": model_key,
        "label": label,
        "real_overlap_policy": real_overlap_policy,
        "policy": policy_label,
        "explained_split": split_explicat,
        "split_rows": int(len(df_explicat)),
        "explained_rows": int(len(sample_indices)),
        "sample_n": int(sample_n),
        "sample_seed": int(seed),
        "note": (
            "SHAP is computed on a sample from the requested split. One-hot variables "
            "and missingness indicators are aggregated back to the original variable "
            "name; the beeswarm shows the top 20 variables by aggregated mean_abs_shap."
        ),
        "outputs": {
            "variable_importance": str(table_path),
            "top_variables_figure": str(figure_path),
            "beeswarm_top20": str(beeswarm_path),
            "dependence": dependence_paths,
            "summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "variable_importance": str(table_path),
        "top_variables_figure": str(figure_path),
        "beeswarm_top20": str(beeswarm_path),
        "dependence": dependence_paths,
        "summary": str(summary_path),
    }


def _select_classic_shap_split(
    preferred_split: str,
    x_test: np.ndarray,
    test: pd.DataFrame,
    x_real: np.ndarray | None,
    real: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Choose the split used for SHAP explanations."""
    # Prefer the requested external split, with test as a fallback when real data
    # are unavailable for the selected policy.
    if preferred_split == "real" and x_real is not None and not real.empty:
        return x_real, real, "real"
    if preferred_split == "test" and not test.empty:
        return x_test, test, "test"
    if x_real is not None and not real.empty:
        return x_real, real, "real"
    if not test.empty:
        return x_test, test, "test"
    raise ValueError("There is no test or real split to compute SHAP.")


def _sample_rows_array(
    x: np.ndarray,
    sample_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows reproducibly from a NumPy feature matrix."""
    size = min(int(sample_n), int(x.shape[0]))
    if size <= 0:
        raise ValueError("The SHAP sample is empty.")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=size, replace=False))
    return x[indices], indices


def _compute_classic_shap_values(
    shap_module: Any,
    model: Any,
    x_sample: np.ndarray,
    x_background: np.ndarray,
) -> Any:
    """Run the right SHAP explainer for a linear or tree model."""
    if hasattr(model, "coef_"):
        explainer = shap_module.LinearExplainer(model, x_background)
        values = explainer.shap_values(x_sample)
    else:
        explainer = shap_module.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
    return values


def _normalize_classic_shap_values(values: Any) -> np.ndarray:
    """Convert SHAP output variants into a 2D array."""
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP format for classic model: shape={values.shape}")
    return values


def _plot_top_variables_shap_classic(
    importance: pd.DataFrame,
    label: str,
    path: Path,
) -> str:
    """Save a bar plot of top SHAP variables."""
    _style()
    top = importance.head(20).sort_values("mean_abs_shap", ascending=True).copy()
    total_importance = importance["mean_abs_shap"].sum()
    top["importance_pct"] = np.where(
        total_importance > 0,
        top["mean_abs_shap"] / total_importance * 100,
        0.0,
    )

    fig, ax = plt.subplots(figsize=(11.2, max(7.2, 0.42 * len(top))), constrained_layout=True)
    bars = ax.barh(top["variable"], top["mean_abs_shap"], color="#0072B2")
    ax.set_title("Top SHAP predictors in the final LightGBM model", pad=30, fontsize=18)
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

    fig.text(
        0.5,
        0.925,
        "Values indicate each variable's share of total mean absolute SHAP importance.",
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
    )
    return save_report_figure(fig, path)


def _plot_beeswarm_shap_classic(
    shap_module: Any,
    shap_values: np.ndarray,
    x_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_variables: list[str],
    label: str,
    path: Path,
) -> str:
    """Save a SHAP beeswarm aggregated to original variables."""
    shap_values_plot, x_sample_plot = _aggregate_classic_shap_for_beeswarm(
        shap_values=shap_values,
        x_sample=x_sample,
        feature_importance=feature_importance,
        top_variables=top_variables,
    )
    plt.figure(figsize=(8.8, 6.8))
    shap_module.summary_plot(
        shap_values_plot,
        x_sample_plot,
        plot_type="dot",
        max_display=min(15, x_sample_plot.shape[1]),
        show=False,
    )
    plt.title(f"SHAP summary: {label}")
    plt.tight_layout()
    save_report_figure(plt.gcf(), path)
    return str(path)


def _aggregate_classic_shap_for_beeswarm(
    shap_values: np.ndarray,
    x_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_variables: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Aggregate encoded features back to original variables for beeswarm plots."""
    feature_importance = _ensure_shap_feature_variables(feature_importance)
    grouped_shap: list[np.ndarray] = []
    grouped_values: dict[str, np.ndarray] = {}
    available_features = set(x_sample.columns.astype(str))

    for variable in top_variables:
        variable = str(variable)
        subset = feature_importance.loc[feature_importance["variable"] == variable].copy()
        features = [str(feature) for feature in subset["feature"].tolist() if str(feature) in available_features]
        if not features:
            continue

        positions = [x_sample.columns.get_loc(feature) for feature in features]
        grouped_shap.append(shap_values[:, positions].sum(axis=1))
        representative_feature = _feature_dependence_for_variable(variable, feature_importance, x_sample)
        display_label = _shap_display_label_for_variable(variable, representative_feature)

        if variable in x_sample.columns:
            grouped_values[display_label] = x_sample[variable].to_numpy()
        else:
            if representative_feature in x_sample.columns:
                grouped_values[display_label] = x_sample[representative_feature].to_numpy()
            else:
                grouped_values[display_label] = x_sample.loc[:, features].sum(axis=1).to_numpy()

    if not grouped_shap:
        return shap_values, x_sample

    shap_grouped = np.column_stack(grouped_shap)
    x_grouped = pd.DataFrame(grouped_values, index=x_sample.index)
    return shap_grouped, x_grouped


def _shap_display_label_for_variable(variable: str, representative_feature: str) -> str:
    """Add the representative one-hot category to aggregated SHAP labels."""
    variable = str(variable)
    representative_feature = str(representative_feature)
    prefix = f"{variable}__"
    if representative_feature.startswith(prefix):
        level = representative_feature[len(prefix) :]
        return f"{variable}: {level}"
    return variable


def _plot_dependence_shap_classic(
    shap_module: Any,
    shap_values: np.ndarray,
    x_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    label: str,
    output_dir: Path,
    top_n: int = 6,
) -> dict[str, str]:
    """Save dependence plots for the top SHAP variables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_importance = _ensure_shap_feature_variables(feature_importance)
    variable_importance = (
        feature_importance.groupby("variable", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )
    paths: dict[str, str] = {}
    for variable in variable_importance["variable"].tolist():
        feature = _feature_dependence_for_variable(variable, feature_importance, x_sample)
        safe_feature = safe_filename(str(feature))
        path = output_dir / f"{safe_feature}_dependence.png"
        plt.figure(figsize=(7.2, 4.8))
        shap_module.dependence_plot(
            feature,
            shap_values,
            x_sample,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP dependence: {feature}")
        plt.tight_layout()
        save_report_figure(plt.gcf(), path)
        paths[str(feature)] = str(path)
    return paths


def _feature_dependence_for_variable(
    variable: str,
    feature_importance: pd.DataFrame,
    x_sample: pd.DataFrame,
) -> str:
    """Choose the encoded feature that best represents one original variable."""
    feature_importance = _ensure_shap_feature_variables(feature_importance)
    if variable in x_sample.columns:
        return variable
    subset = feature_importance.loc[feature_importance["variable"] == variable].copy()
    if subset.empty:
        return str(variable)
    subset["is_missing"] = subset["feature"].astype(str).str.endswith("__missing")
    subset = subset.sort_values(["is_missing", "mean_abs_shap"], ascending=[True, False])
    return str(subset.iloc[0]["feature"])


def _ensure_shap_feature_variables(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Return SHAP feature importance with original-variable names attached."""
    if "variable" in feature_importance.columns:
        return feature_importance
    feature_importance = feature_importance.copy()
    feature_importance["variable"] = feature_importance["feature"].map(_feature_to_variable)
    return feature_importance


def _split_predictions(spec: ModelSpec, df_split: pd.DataFrame, score: np.ndarray, split: str) -> pd.DataFrame:
    """Build the prediction table for one split and model."""
    cols = [ID_COL, TARGET]
    if PATIENT_COL in df_split.columns:
        cols.insert(1, PATIENT_COL)
    if "data_index" in df_split.columns:
        cols.append("data_index")
    pred = df_split[cols].copy()
    pred.insert(0, "model", spec.key)
    pred.insert(1, "label", spec.label)
    pred["split"] = split
    pred["y_true"] = df_split[TARGET].astype(int).to_numpy()
    pred["score"] = score
    return pred


def _compute_model_importances(
    model: Any,
    model_key: str,
    label: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract and normalize feature importance from a fitted model."""
    raw = _raw_importance(model)
    if raw is None:
        return pd.DataFrame(columns=["model", "label", "feature", "importancia"])
    raw = np.asarray(raw, dtype=float)
    if len(raw) != len(feature_names):
        return pd.DataFrame(columns=["model", "label", "feature", "importancia"])
    raw = np.abs(raw)
    total = float(raw.sum())
    importance = raw / total if total > 0 else raw
    return pd.DataFrame(
        {
            "model": model_key,
            "label": label,
            "feature": feature_names,
            "importancia": importance,
        }
    ).sort_values("importancia", ascending=False)


def _raw_importance(model: Any) -> np.ndarray | None:
    """Read raw importance values from models that expose them."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).reshape(-1)
    return None


def _aggregate_variable_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate encoded-feature importances to original variables."""
    return aggregate_encoded_importance(
        feature_importance,
        value_col="importancia",
        group_cols=["model", "label"],
        sort_cols=["model", "importancia"],
    )


def _build_feature_table(preprocessor: Any) -> pd.DataFrame:
    """Build a table linking transformed features to original variables."""
    rows = [{"feature": feature, "variable": _feature_to_variable(feature)} for feature in preprocessor.feature_names]
    return pd.DataFrame(rows)


def _split_unit(df_model: pd.DataFrame, split_unit: str) -> pd.Series:
    """Return the unit used for splitting rows into cohorts."""
    return _shared_split_unit(df_model, split_unit, missing_patient_label="episode")


def _cut_units(
    units: list[object],
    proportions: tuple[float, float, float],
) -> tuple[list[object], list[object], list[object]]:
    """Cut ordered units into train, validation, and test lists."""
    n = len(units)
    if n == 0:
        return [], [], []
    p_train, p_valid, _ = proportions
    n_train = max(1, int(round(n * p_train)))
    n_valid = max(1, int(round(n * p_valid))) if n >= 3 else 0
    n_train = min(n_train, n)
    n_valid = min(n_valid, max(0, n - n_train))
    return units[:n_train], units[n_train : n_train + n_valid], units[n_train + n_valid :]


def _clone_estimator(estimator: Any) -> Any:
    """Clone a scikit-learn compatible estimator."""
    from sklearn.base import clone

    return clone(estimator)


def _predict_proba(model: Any, x: np.ndarray | None) -> np.ndarray:
    """Return a positive-class risk score from different estimator APIs."""
    if x is None:
        return np.array([])
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x))
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(x), dtype=float)


def _row_not_available(spec: ModelSpec) -> dict[str, object]:
    """Build a result row for an unavailable model."""
    return {
        "model": spec.key,
        "label": spec.label,
        "status": "not_available",
        "reason": spec.reason,
    }


def _tuning_not_available(spec: ModelSpec) -> dict[str, object]:
    """Build a tuning row for an unavailable model."""
    return {
        "model": spec.key,
        "label": spec.label,
        "tuning_method": "simple_grid",
        "status": "not_available",
        "error": spec.reason,
    }


def _row_error(spec: ModelSpec, error: str) -> dict[str, object]:
    """Build a result row for a failed model."""
    return {
        "model": spec.key,
        "label": spec.label,
        "status": "error",
        "reason": error,
    }


def _empty_metrics(prefix: str) -> dict[str, object]:
    """Return NaN metrics for a missing evaluation split."""
    suffixes = [
        "auroc",
        "auprc",
        "prevalence",
        "auprc_lift",
        "sensitivity",
        "specificity",
        "ppv",
        "f1",
        "episode_auroc",
        "episode_auprc",
        "episode_prevalence",
        "episode_auprc_lift",
        "episode_sensitivity",
        "episode_ppv",
    ]
    return {f"{prefix}_{suffix}": float("nan") for suffix in suffixes}


def _print_model_results(row: dict[str, object]) -> None:
    """Print a short summary right after evaluating the model."""
    print(
        "[classic models]   Test: "
        f"AUPRC={_format_float(row.get('test_auprc'))} "
        f"AUROC={_format_float(row.get('test_auroc'))} "
        f"sens={_format_float(row.get('test_sensitivity'))} "
        f"PPV={_format_float(row.get('test_ppv'))}",
        flush=True,
    )
    if not pd.isna(row.get("real_auprc", float("nan"))):
        print(
            "[classic models]   Real: "
            f"AUPRC={_format_float(row.get('real_auprc'))} "
            f"AUROC={_format_float(row.get('real_auroc'))} "
            f"sens={_format_float(row.get('real_sensitivity'))} "
            f"PPV={_format_float(row.get('real_ppv'))}",
            flush=True,
        )


def _format_float(value: object, digits: int = 3) -> str:
    """Format numbers for progress messages."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _unavailable_spec(key: str, label: str, package: str, exc: Exception) -> ModelSpec:
    """Build a model spec for a missing optional package."""
    return ModelSpec(
        key=key,
        label=label,
        estimator=None,
        param_grid=[],
        available=False,
        reason=f"Could not import {package}: {type(exc).__name__}: {exc}",
    )


def _unknown_spec(key: str) -> ModelSpec:
    """Build a model spec for an unknown model key."""
    return ModelSpec(
        key=key,
        label=key,
        estimator=None,
        param_grid=[],
        available=False,
        reason="Unknown model.",
    )


def _scale_pos_weight(y: np.ndarray) -> float:
    """Calculate the negative/positive ratio for class weighting."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos > 0 else 1.0


def _safe_div(num: float, den: float) -> float:
    """Divide numbers while returning NaN for zero denominators."""
    return float(num / den) if den else float("nan")


def _style() -> None:
    """Apply a consistent Matplotlib style for classic-model figures."""
    apply_report_style()





