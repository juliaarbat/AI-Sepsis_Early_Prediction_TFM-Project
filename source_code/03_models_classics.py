from __future__ import annotations

# Importing _bootstrap registers the project root on sys.path before src imports.
import _bootstrap

assert _bootstrap.PROJECT_ROOT

import json
from pathlib import Path

from src.data_loading import load_sepsis_model_with_sofa
from src.training_comparisons import create_policy_comparison
from src.config import (
    MODEL_CV_FOLDS,
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    MODELS_CLASSICS_OUTPUTS_DIR,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
)
from src.classic_models_24h import (
    CLASSIC_MODEL_FILES,
    DEFAULT_MODELS,
    train_and_evaluate_classic_models_24h,
)
from src.real_policies import REAL_POLICIES, REAL_START_DATE_DEFAULT
from src.progress import log_end, log_start, step
from src.reporting import print_model_environment


EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
REAL_START_DATE = REAL_START_DATE_DEFAULT
CV_FOLDS = MODEL_CV_FOLDS
OPTUNA_TRIALS = None

REAL_POLICIES_TO_RUN = REAL_POLICIES


def main() -> None:
    """Run the full classic-model workflow across real-cohort policies."""
    title = "classic model comparison for next-day sepsis prediction"
    log_start(title)
    print_model_environment(("optuna", "xgboost", "catboost", "lightgbm"))

    total_steps = 2 + len(REAL_POLICIES_TO_RUN)
    with step("Load the clean dataset with SOFA scores and 24h labels", number=1, total=total_steps):
        df_sofa = load_sepsis_model_with_sofa(
            episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
            episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
            max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
        )

    summaries: dict[str, object] = {}
    output_dirs: dict[str, Path] = {}
    # Each policy defines a different interpretation of the external 2026 cohort.
    # Models are trained from scratch for each policy so the comparison is fair.
    for idx, policy in enumerate(REAL_POLICIES_TO_RUN, start=2):
        policy_key = str(policy["key"])
        output_dir = MODELS_CLASSICS_OUTPUTS_DIR / policy_key
        output_dirs[policy_key] = output_dir
        with step(
            f"Train models: {policy['label']}",
            number=idx,
            total=total_steps,
        ):
            print(policy["description"], flush=True)
            print("Each model trial and its main metrics will be shown.", flush=True)
            print(f"Cross-validation enabled: {CV_FOLDS} patient-grouped folds.", flush=True)
            summaries[policy_key] = train_and_evaluate_classic_models_24h(
                df_sofa,
                output_dir=output_dir,
                model_keys=DEFAULT_MODELS,
                split_unit="patient",
                real_start_date=REAL_START_DATE,
                real_overlap_policy=policy_key,
                evaluate_real_from_real_start=True,
                exclude_microbiology=False,
                optuna_trials=OPTUNA_TRIALS,
                cv_folds=CV_FOLDS,
                generate_figures=False,
                verbose=True,
            )

    with step(
        "Create the comparison across real-cohort policies",
        number=total_steps,
        total=total_steps,
    ):
        comparison_paths = create_policy_comparison(output_dirs, summaries)

    print("Classic next-day prediction models evaluated across the real 2026 cohorts")
    print(
        json.dumps(
            {
                policy_key: {
                    "cohort": summary["cohort"],
                    "splits": summary["splits"],
                    "filter_audit": summary.get("filter_audit", {}),
                }
                for policy_key, summary in summaries.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for policy_key, output_dir in output_dirs.items():
        print(f"Outputs {policy_key}: {output_dir}")
        print("  Summary:", output_dir / CLASSIC_MODEL_FILES["summary"])
        print("  Results:", output_dir / CLASSIC_MODEL_FILES["results"])
        print("  Predictions:", output_dir / CLASSIC_MODEL_FILES["predictions"])
        print("  Split audit:", output_dir / CLASSIC_MODEL_FILES["split_audit"])
        print("  Figures:", output_dir / CLASSIC_MODEL_FILES["figures_index"])

    print("Policy comparison:", comparison_paths["comparison"])
    print("Split comparison:", comparison_paths["splits"])
    print("Comparison summary:", comparison_paths["summary"])
    print("Comparison figures:", comparison_paths["figures_index"])
    log_end(title)


if __name__ == "__main__":
    main()




