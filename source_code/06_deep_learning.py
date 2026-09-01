from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import subprocess
import sys

from src.data_loading import load_sepsis_model_with_sofa
from src.config import (
    DEEP_LEARNING_OUTPUTS_DIR,
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
)
from src.temporal_model_24h import train_and_evaluate_temporal_model_24h
from src.output_contracts import (
    deep_metrics_filename,
    deep_model_filename,
    deep_predictions_filename,
    deep_summary_filename,
)
from src.real_policies import REAL_POLICIES, REAL_START_DATE_DEFAULT
from src.progress import log_end, log_start, step
from src.reporting import print_temporal_metrics


OPTUNA_TRIALS = None
EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
REAL_START_DATE = REAL_START_DATE_DEFAULT
REAL_POLICY_RUNS = REAL_POLICIES
# Shared settings keep the comparison between architectures and real-cohort
# policies consistent; hyperparameter optimisation is run separately later.
COMMON_PARAMS = {
    # Each prediction uses the previous 10 patient-days to predict the next day.
    "lookback_days": 10,
    "epochs": 8,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "train_parts": 1,
    "split_unit": "patient",
    "real_start_date": REAL_START_DATE,
    "evaluate_real_from_real_start": True,
    # Weight positive examples because next-day sepsis is the minority class.
    "imbalance_strategy": "pos_weight",
    "exclude_microbiology": False,
    "optuna_trials": OPTUNA_TRIALS,
}
MODEL_RUNS = (
    {
        "key": "transformer",
        "label": "Transformer",
        "prefix": "transformer_24h",
        "params": {
            **COMMON_PARAMS,
            "model_type": "transformer",
            "d_model": 48,
            "n_layers": 1,
        },
    },
    {
        "key": "lstm",
        "label": "LSTM",
        "prefix": "lstm_24h",
        "params": {
            **COMMON_PARAMS,
            "model_type": "lstm",
            "d_model": 64,
            "n_layers": 2,
            "dropout": 0.20,
            "recurrent_hidden_size": 64,
            "recurrent_bidirectional": False,
            "max_missing_ratio": 0.80,
            "early_stopping_patience": 4,
        },
    },
    {
        "key": "rnn",
        "label": "RNN",
        "prefix": "rnn_24h",
        "params": {
            **COMMON_PARAMS,
            "model_type": "rnn",
            "d_model": 64,
            "n_layers": 2,
            "dropout": 0.20,
            "recurrent_hidden_size": 64,
            "recurrent_bidirectional": False,
            "max_missing_ratio": 0.80,
            "early_stopping_patience": 4,
        },
    },
)


def generate_figures_in_separate_process(model: str, policy_key: str) -> None:
    """Generate plots outside the training process to avoid OpenMP DLL conflicts."""
    # Figure generation is isolated from training to avoid conflicts between
    # the numerical libraries used by the model and the plotting process.
    command = [
        sys.executable,
        "scripts/07_deep_learning_figures.py",
        "--model",
        model,
        "--policy",
        policy_key,
    ]
    subprocess.run(command, cwd=_bootstrap.PROJECT_ROOT, check=True)


def main() -> None:
    """Train all configured temporal models with the shared SOFA cohort policy."""
    log_start("Deep-learning training for next-day prediction")

    # Load the clean SOFA dataset once and reuse it for all sequence models.
    total_steps = 1 + len(MODEL_RUNS) * len(REAL_POLICY_RUNS)
    with step("Load clean dataset with SOFA and 24h labels", number=1, total=total_steps):
        df_sofa = load_sepsis_model_with_sofa(
            episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
            episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
            max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
        )

    summaries: dict[str, object] = {}
    step_index = 2
    # Run every architecture under every real-cohort policy using the same
    # input cohort and temporal split configuration.
    for model in MODEL_RUNS:
        model_key = str(model["key"])
        model_label = str(model["label"])
        output_prefix = str(model["prefix"])

        for policy in REAL_POLICY_RUNS:
            policy_key = str(policy["key"])
            output_dir = DEEP_LEARNING_OUTPUTS_DIR / model_key / policy_key
            params = dict(model["params"])
            params["real_overlap_policy"] = policy_key

            # Same temporal pipeline; only the sequence model and real policy change.
            with step(
                f"Train {model_label}: {policy['label']}",
                number=step_index,
                total=total_steps,
                detail=f"Base run without Optuna, epochs=8, real split from {REAL_START_DATE}",
            ):
                summary = train_and_evaluate_temporal_model_24h(
                    df_sofa,
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                    **params,
                )
                summaries[f"{model_key}_{policy_key}"] = summary

            generate_figures_in_separate_process(model_key, policy_key)
            print(f"{model_label} for next-day prediction trained - {policy['label']}")
            if model_key == "transformer":
                print(json.dumps(summary["cohort"], ensure_ascii=False, indent=2))
            print_temporal_metrics(summary, show_thresholds=True)
            print("Outputs:", output_dir)
            print("Summary:", output_dir / deep_summary_filename(output_prefix))
            print("Comparable metrics:", output_dir / deep_metrics_filename(output_prefix))
            print("Predictions:", output_dir / deep_predictions_filename(output_prefix))
            print("Figures:", output_dir / "figures")
            print("Model:", output_dir / deep_model_filename(output_prefix))
            step_index += 1

    # Save one compact summary for comparing all base sequence-model runs.
    global_summary_path = DEEP_LEARNING_OUTPUTS_DIR / "deep_learning_24h_base_summary.json"
    global_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                run_key: {
                    "cohort": summary["cohort"],
                    "splits": summary["splits"],
                    "split_temporal": summary["split_temporal"],
                }
                for run_key, summary in summaries.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Keep one short index per architecture for convenient output review.
    for model in MODEL_RUNS:
        model_key = str(model["key"])
        output_prefix = str(model["prefix"])
        model_summary_path = DEEP_LEARNING_OUTPUTS_DIR / model_key / f"{output_prefix}_policies_summary.json"
        model_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    policy_key: {
                        "cohort": summaries[f"{model_key}_{policy_key}"]["cohort"],
                        "splits": summaries[f"{model_key}_{policy_key}"]["splits"],
                        "split_temporal": summaries[f"{model_key}_{policy_key}"]["split_temporal"],
                    }
                    for policy_key in (str(policy["key"]) for policy in REAL_POLICY_RUNS)
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    print("Optuna tuning: run python scripts/08_deep_learning_optuna.py if this model is the best")
    print("Base deep-learning summary:", global_summary_path)
    log_end("Deep-learning training for next-day prediction")


if __name__ == "__main__":
    main()



