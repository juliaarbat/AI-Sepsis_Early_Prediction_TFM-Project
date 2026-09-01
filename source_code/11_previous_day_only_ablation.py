from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import pandas as pd

from src.classic_models_24h import CLASSIC_MODEL_FILES, train_and_evaluate_classic_models_24h
from src.config import (
    CLASSIC_OPTUNA_TRIALS,
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    OUTPUTS_DIR,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
)
from src.data_loading import load_sepsis_model_with_sofa
from src.output_contracts import deep_metrics_filename, deep_output_paths
from src.progress import log_end, log_start, step
from src.real_policies import REAL_ALL_2026, REAL_START_DATE_DEFAULT
from src.reporting import print_model_environment
from src.temporal_model_24h import train_and_evaluate_temporal_model_24h


OUTPUT_BASE = OUTPUTS_DIR / "previous_day_only_ablation"
CLASSIC_OUTPUT_DIR = OUTPUT_BASE / "classic_lightgbm_optuna"
DEEP_OUTPUT_DIR = OUTPUT_BASE / "deep_transformer_optuna"
INDEX_FILE = "previous_day_only_ablation_index.json"

EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
REAL_START_DATE = REAL_START_DATE_DEFAULT
REAL_POLICY = REAL_ALL_2026

# These two models are the selected final candidates used in the ablation.
CLASSIC_MODEL_KEY = "lightgbm"
DEEP_MODEL_KEY = "transformer"
DEEP_OUTPUT_PREFIX = "transformer_previous_day_only_24h"
DEEP_OPTUNA_TRIALS = 15
RUN_NEW_OPTUNA_TRIALS = True


def main() -> None:
    """Run the previous-day-only ablation for the selected final candidate models."""
    title = "Previous-day-only ablation with Optuna"
    log_start(title)
    print_model_environment(("optuna", "lightgbm", "torch"))

    # This controlled comparison removes longer temporal history while keeping
    # the cohort, split and evaluation policy unchanged.
    # Step 11 deliberately trains models only. SHAP and figures are generated in script 12.
    total_steps = 4 if RUN_NEW_OPTUNA_TRIALS else 3
    if RUN_NEW_OPTUNA_TRIALS:
        with step("Load clean dataset with SOFA scores and 24h labels", number=1, total=total_steps):
            df_sofa = _load_modeling_dataset()
        step_number = 2
    else:
        df_sofa = None
        step_number = 1
        # Reuse mode avoids retraining and reads the summaries from a previous run.
        print("RUN_NEW_OPTUNA_TRIALS = False; existing ablation outputs will be reused.")

    with step(
        "LightGBM previous-day-only outputs",
        number=step_number,
        total=total_steps,
        detail=_execution_detail(CLASSIC_OPTUNA_TRIALS, "temporal-window features disabled"),
    ):
        classic_summary = _run_or_reuse_previous_day_lightgbm(df_sofa)
    step_number += 1

    with step(
        "Transformer previous-day-only outputs",
        number=step_number,
        total=total_steps,
        detail=_execution_detail(DEEP_OPTUNA_TRIALS, "lookback_days fixed at 1"),
    ):
        deep_summary = _run_or_reuse_previous_day_transformer(df_sofa)
    step_number += 1

    with step("Write previous-day-only ablation index", number=step_number, total=total_steps):
        index_path = _write_ablation_index(classic_summary, deep_summary)
        print("Previous-day-only ablation index:", index_path)

    log_end(title)


def _load_modeling_dataset() -> pd.DataFrame:
    """Load the same SOFA-labelled dataset used by the main modelling scripts."""
    return load_sepsis_model_with_sofa(
        episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
        lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
        vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
        max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
    )


def _run_or_reuse_previous_day_lightgbm(df_sofa: pd.DataFrame | None) -> dict[str, object]:
    """Run LightGBM Optuna or reuse its previous-day-only summary."""
    if RUN_NEW_OPTUNA_TRIALS:
        if df_sofa is None:
            raise ValueError("df_sofa is required when RUN_NEW_OPTUNA_TRIALS is True.")
        return _run_previous_day_lightgbm(df_sofa)
    return _load_json(CLASSIC_OUTPUT_DIR / CLASSIC_MODEL_FILES["summary"])


def _run_previous_day_lightgbm(df_sofa: pd.DataFrame) -> dict[str, object]:
    """Tune LightGBM after disabling engineered temporal-window features."""
    # LightGBM keeps same-day predictors but loses rolling and aggregate history.
    return train_and_evaluate_classic_models_24h(
        df_sofa,
        output_dir=CLASSIC_OUTPUT_DIR,
        model_keys=(CLASSIC_MODEL_KEY,),
        split_unit="patient",
        real_start_date=REAL_START_DATE,
        real_overlap_policy=REAL_POLICY,
        evaluate_real_from_real_start=True,
        exclude_microbiology=False,
        include_temporal_features=False,
        optuna_trials=CLASSIC_OPTUNA_TRIALS,
        cv_folds=None,
        calculate_shap=False,
        generate_figures=False,
        verbose=True,
    )


def _run_or_reuse_previous_day_transformer(df_sofa: pd.DataFrame | None) -> dict[str, object]:
    """Run Transformer Optuna or reuse its previous-day-only summary."""
    if RUN_NEW_OPTUNA_TRIALS:
        if df_sofa is None:
            raise ValueError("df_sofa is required when RUN_NEW_OPTUNA_TRIALS is True.")
        return _run_previous_day_transformer(df_sofa)
    return _load_json(deep_output_paths(DEEP_OUTPUT_DIR, DEEP_OUTPUT_PREFIX)["summary"])


def _run_previous_day_transformer(df_sofa: pd.DataFrame) -> dict[str, object]:
    """Tune the Transformer while forcing the temporal window to one day."""
    # A lookback of one means the Transformer receives only the immediately
    # preceding patient-day when predicting the next-day label.
    return train_and_evaluate_temporal_model_24h(
        df_sofa,
        output_dir=DEEP_OUTPUT_DIR,
        output_prefix=DEEP_OUTPUT_PREFIX,
        model_type=DEEP_MODEL_KEY,
        lookback_days=1,
        tune_lookback_days=False,
        epochs=8,
        batch_size=64,
        learning_rate=5.526573e-4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1977553,
        train_parts=1,
        split_unit="patient",
        real_start_date=REAL_START_DATE,
        real_overlap_policy=REAL_POLICY,
        evaluate_real_from_real_start=True,
        imbalance_strategy="focal_loss",
        exclude_microbiology=False,
        optuna_trials=DEEP_OPTUNA_TRIALS,
        early_stopping_patience=4,
        verbose=True,
    )


def _load_json(path: Path) -> dict[str, object]:
    """Read a JSON output created by a previous ablation run."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot reuse previous-day-only outputs because this file is missing: {path}. "
            "Set RUN_NEW_OPTUNA_TRIALS = True and rerun this script first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _execution_detail(trials: int, note: str) -> str:
    """Return the progress detail for run or reuse mode."""
    if RUN_NEW_OPTUNA_TRIALS:
        return f"{trials} trials; {note}"
    return "Reusing existing outputs; Optuna is not run"


def _write_ablation_index(
    classic_summary: dict[str, object],
    deep_summary: dict[str, object],
) -> Path:
    """Write a compact JSON index consumed by the figure-generation script."""
    # The index keeps the paired model outputs discoverable by script 12.
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_BASE / INDEX_FILE
    index = _build_index(classic_summary, deep_summary)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def _build_index(
    classic_summary: dict[str, object],
    deep_summary: dict[str, object],
) -> dict[str, object]:
    """Create a compact index with all files produced by the ablation run."""
    deep_paths = deep_output_paths(DEEP_OUTPUT_DIR, DEEP_OUTPUT_PREFIX)
    return {
        "objective": (
            "Ablation analysis testing whether the best models retain performance "
            "when restricted to the immediately preceding patient-day."
        ),
        "real_policy": REAL_POLICY,
        "real_start_date": str(REAL_START_DATE),
        "classic": {
            "model": CLASSIC_MODEL_KEY,
            "mode": "same-day tabular predictors; temporal-window features disabled",
            "output_dir": str(CLASSIC_OUTPUT_DIR),
            "summary": str(CLASSIC_OUTPUT_DIR / CLASSIC_MODEL_FILES["summary"]),
            "results": str(CLASSIC_OUTPUT_DIR / CLASSIC_MODEL_FILES["results"]),
            "tuning": str(CLASSIC_OUTPUT_DIR / CLASSIC_MODEL_FILES["tuning"]),
            "variable_importance": str(CLASSIC_OUTPUT_DIR / CLASSIC_MODEL_FILES["variable_importance"]),
            "shap_variable_importance": str(
                CLASSIC_OUTPUT_DIR / "shap" / "lightgbm_shap_variable_importance.csv"
            ),
            "optuna_trials": CLASSIC_OPTUNA_TRIALS,
            "cohort": classic_summary.get("cohort", {}),
            "splits": classic_summary.get("splits", {}),
        },
        "deep_learning": {
            "model": DEEP_MODEL_KEY,
            "mode": "sequence model with lookback_days fixed to 1",
            "output_dir": str(DEEP_OUTPUT_DIR),
            "summary": str(deep_paths["summary"]),
            "metrics": str(DEEP_OUTPUT_DIR / deep_metrics_filename(DEEP_OUTPUT_PREFIX)),
            "tuning": str(deep_paths["tuning"]),
            "model_path": str(deep_paths["model"]),
            "shap_variable_importance": str(
                DEEP_OUTPUT_DIR / "shap" / f"{DEEP_OUTPUT_PREFIX}_shap_variable_importance.csv"
            ),
            "output_prefix": DEEP_OUTPUT_PREFIX,
            "optuna_trials": DEEP_OPTUNA_TRIALS,
            "tune_lookback_days": False,
            "cohort": deep_summary.get("cohort", {}),
            "splits": deep_summary.get("splits", {}),
        },
        "next_step": "Run scripts/12_previous_day_only_figures.py after this script finishes.",
    }


if __name__ == "__main__":
    main()
