from __future__ import annotations

from pathlib import Path


# Standard filenames produced by deep-learning runs.
DEEP_SUMMARY_SUFFIX = "_summary.json"
DEEP_PREDICTIONS_SUFFIX = "_predictions.csv"
DEEP_COMPARABLE_METRICS_SUFFIX = "_comparable_metrics.csv"
DEEP_TUNING_SUFFIX = "_tuning.csv"
DEEP_MODEL_SUFFIX = "_model.pt"
DEEP_EXCLUDED_MISSING_SUFFIX = "_excluded_missing_variables.csv"

# Shared artifacts generated after Optuna model selection.
POST_OPTUNA_COMPARISON_FILE = "01_final_model_comparison.csv"
POST_OPTUNA_DECISION_FILE = "02_final_model_decision.csv"
POST_OPTUNA_CV_FILE = "03_post_optuna_robustness_cv.csv"
POST_OPTUNA_INDEX_FILE = "post_optuna_final_index.json"
POST_OPTUNA_README_FILE = "README.md"

# Shared labels used in splits, evaluation levels, and model execution modes.
SPLIT_TRAIN = "train"
SPLIT_VALID = "valid"
SPLIT_TEST = "test"
SPLIT_REAL = "real"
STANDARD_SPLITS = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST, SPLIT_REAL)
TRAIN_VALID_TEST_SPLITS = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)

LEVEL_NEXT_DAY = "next_day"
LEVEL_EPISODE = "episode"
TOP_RISK_LEVEL_DAY = "day"
TOP_RISK_LEVEL_EPISODE = "episode"

EXECUTION_BASE = "base"
EXECUTION_OPTUNA = "optuna"

MODEL_FAMILY_CLASSIC = "classic"
MODEL_FAMILY_DEEP_LEARNING = "deep_learning"

# Metric names and comparable-output column names used across model families.
METRIC_AUROC = "auroc"
METRIC_AUPRC = "auprc"
METRIC_AUPRC_LIFT = "auprc_lift"
METRIC_SENSITIVITY = "sensitivity"
METRIC_PPV = "ppv"
STANDARD_METRICS = (METRIC_AUROC, METRIC_AUPRC, METRIC_AUPRC_LIFT, METRIC_SENSITIVITY, METRIC_PPV)

COL_TEST_AUPRC = "test_auprc"
COL_REAL_AUPRC = "real_auprc"
COL_TEST_AUROC = "test_auroc"
COL_REAL_AUROC = "real_auroc"
COL_REAL_AUPRC_LIFT = "real_auprc_lift"
COL_REAL_SENSITIVITY = "real_sensitivity"
COL_REAL_PPV = "real_ppv"
COL_TEST_EPISODE_AUPRC = "test_episode_auprc"
COL_REAL_EPISODE_AUPRC = "real_episode_auprc"
COL_REAL_EPISODE_AUROC = "real_episode_auroc"
COL_REAL_EPISODE_AUPRC_LIFT = "real_episode_auprc_lift"


def deep_summary_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_SUMMARY_SUFFIX}"


def deep_predictions_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_PREDICTIONS_SUFFIX}"


def deep_metrics_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_COMPARABLE_METRICS_SUFFIX}"


def deep_tuning_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_TUNING_SUFFIX}"


def deep_model_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_MODEL_SUFFIX}"


def deep_excluded_missing_filename(prefix: str) -> str:
    return f"{prefix}{DEEP_EXCLUDED_MISSING_SUFFIX}"


def deep_output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    """Return standard deep-learning output paths for one run."""
    return {
        "summary": output_dir / deep_summary_filename(prefix),
        "predictions": output_dir / deep_predictions_filename(prefix),
        "comparable_metrics": output_dir / deep_metrics_filename(prefix),
        "excluded_missing_variables": output_dir / deep_excluded_missing_filename(prefix),
        "model": output_dir / deep_model_filename(prefix),
        "tuning": output_dir / deep_tuning_filename(prefix),
    }

