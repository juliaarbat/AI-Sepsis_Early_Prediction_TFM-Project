from __future__ import annotations

# Importing _bootstrap registers the project root on sys.path before src imports.
import _bootstrap

assert _bootstrap.PROJECT_ROOT

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import load_sepsis_model_with_sofa
from src.config import (
    CLASSIC_OPTUNA_TRIALS,
    MODEL_CV_FOLDS,
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    MODELS_CLASSICS_OUTPUTS_DIR,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SHAP_SAMPLE_N_DEFAULT,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
)
from src.predictive_model_24h import ID_COL, transform_features
from src.classic_models_24h import (
    CLASSIC_MODEL_FILES,
    TRAINED_STATUSES,
    _compute_and_save_classic_model_shap,
    create_chronological_episode_split,
    filter_real_from_start_date,
    prepare_classic_model_data,
    train_and_evaluate_classic_models_24h,
)
from src.real_policies import REAL_ALL_2026, REAL_START_DATE_DEFAULT
from src.progress import log_end, log_start, step
from src.reporting import print_model_environment


EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
REAL_START_DATE = REAL_START_DATE_DEFAULT
CV_FOLDS = MODEL_CV_FOLDS
OPTUNA_TRIALS = CLASSIC_OPTUNA_TRIALS
RUN_NEW_OPTUNA_TRIALS = True
MODEL_SELECTION_CRITERION = "valid_auprc"
COMPUTE_SHAP_AFTER_OPTUNA = True
SHAP_SAMPLE_N = SHAP_SAMPLE_N_DEFAULT

OPTUNA_REAL_POLICY = REAL_ALL_2026
OPTUNA_OUTPUT_BASE = MODELS_CLASSICS_OUTPUTS_DIR / "optuna_best"
OPTUNA_INDEX_FILE = "classic_models_24h_optuna_best_summary.json"


def main() -> None:
    """Run or reuse Optuna tuning only for the complete 2026 real cohort."""
    title = "Optuna on the best classic model for the complete 2026 real cohort"
    log_start(title)
    print_model_environment(("optuna", "xgboost", "catboost", "lightgbm"))

    # Select the baseline model before tuning its hyperparameters with Optuna.
    best_info = _select_best_model_for_optuna()
    model_key = str(best_info["model"])
    recompute_shap = COMPUTE_SHAP_AFTER_OPTUNA and not RUN_NEW_OPTUNA_TRIALS
    needs_dataset = RUN_NEW_OPTUNA_TRIALS or recompute_shap
    total_steps = (2 if recompute_shap else 1) + int(needs_dataset)

    print("Model selected for Optuna:")
    print(f"  {OPTUNA_REAL_POLICY}: {model_key} ({best_info['criterion']}={best_info['value']:.4f})")

    step_num = 1
    df_sofa = None
    if needs_dataset:
        with step("Load the clean dataset with SOFA scores and 24h labels", number=step_num, total=total_steps):
            df_sofa = _load_modeling_dataset()
        step_num += 1

    if not RUN_NEW_OPTUNA_TRIALS:
        # Reuse saved results by default to avoid repeating the expensive search.
        print("No new Optuna trials will run. Existing executions will be reused.")
        print("To launch new trials, set RUN_NEW_OPTUNA_TRIALS = True.")

    output_dir = _classic_optuna_output_dir(model_key=model_key)

    summary, optuna_execution = _run_or_reuse_optuna(
        df_sofa=df_sofa,
        output_dir=output_dir,
        policy=OPTUNA_REAL_POLICY,
        model_key=model_key,
        step_num=step_num,
        total_steps=total_steps,
    )
    step_num += 1

    if recompute_shap:
        with step(
            f"Compute post-Optuna SHAP {OPTUNA_REAL_POLICY}: {model_key}",
            number=step_num,
            total=total_steps,
            detail="SHAP from the saved classic Optuna model; no new trials are run",
        ):
            shap_outputs = _compute_classic_shap_from_run(
                df_sofa=df_sofa,
                output_dir=output_dir,
                model_key=model_key,
                policy=OPTUNA_REAL_POLICY,
            )
            summary["shap"] = shap_outputs
            summary.setdefault("outputs", {})["shap"] = shap_outputs
            _save_classic_optuna_summary(output_dir, summary)

    run_summary = _build_run_summary(
        output_dir=output_dir,
        model_key=model_key,
        selection=best_info,
        execution=optuna_execution,
        run_summary=summary,
    )
    policy_summary: dict[str, object] = {OPTUNA_REAL_POLICY: run_summary}

    summary_path = OPTUNA_OUTPUT_BASE / OPTUNA_INDEX_FILE
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(policy_summary, f, ensure_ascii=False, indent=2)
    readme_path = _write_optuna_best_readme(policy_summary, summary_path.parent)

    print("Best classic-model Optuna summary:", summary_path)
    print("Best classic-model Optuna index:", readme_path)
    log_end(title)


def _load_modeling_dataset() -> pd.DataFrame:
    """Load the same clean SOFA dataset used by the baseline classic models."""
    return load_sepsis_model_with_sofa(
        episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
        lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
        vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
        episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
        max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
    )


def _run_or_reuse_optuna(
    df_sofa: pd.DataFrame | None,
    output_dir: Path,
    policy: str,
    model_key: str,
    step_num: int,
    total_steps: int,
) -> tuple[dict[str, object], str]:
    """Run a new Optuna execution or load an existing one."""
    if RUN_NEW_OPTUNA_TRIALS:
        if df_sofa is None:
            raise ValueError("df_sofa must be loaded before running new Optuna trials.")
        with step(
            f"Optuna {policy}: {model_key}",
            number=step_num,
            total=total_steps,
            detail=f"{OPTUNA_TRIALS} trials on validation AUPRC",
        ):
            # AUPRC is used for tuning because next-day sepsis is an imbalanced label.
            summary = train_and_evaluate_classic_models_24h(
                df_sofa,
                output_dir=output_dir,
                model_keys=(model_key,),
                split_unit="patient",
                real_start_date=REAL_START_DATE,
                real_overlap_policy=policy,
                evaluate_real_from_real_start=True,
                exclude_microbiology=False,
                optuna_trials=OPTUNA_TRIALS,
                cv_folds=CV_FOLDS,
                calculate_shap=COMPUTE_SHAP_AFTER_OPTUNA,
                shap_sample_n=SHAP_SAMPLE_N,
                shap_split="real",
                verbose=True,
            )
        return summary, "new"

    with step(
        f"Reuse Optuna {policy}: {model_key}",
        number=step_num,
        total=total_steps,
        detail="No new trials are run; existing results are loaded",
    ):
        summary = _load_classic_optuna_summary(output_dir)
        print("Reused classic Optuna run:", _classic_file(output_dir, "summary"))
    return summary, "reused"


def _build_run_summary(
    output_dir: Path,
    model_key: str,
    selection: dict[str, object],
    execution: str,
    run_summary: dict[str, object],
) -> dict[str, object]:
    """Build the compact Optuna index for the selected run."""
    return {
        "selected_model": model_key,
        "selection": selection,
        "output_dir": str(output_dir),
        "optuna_execution": execution,
        "optuna_criterion": "valid_auprc at D+1/row level",
        "tuning_metrics": [
            "valid_auprc",
            "valid_auroc",
            "valid_episode_auprc",
            "valid_episode_auroc",
        ],
        "cohort": run_summary["cohort"],
        "splits": run_summary["splits"],
        "filter_audit": run_summary.get("filter_audit", {}),
        "outputs": _execution_outputs(output_dir),
        "shap": run_summary.get("shap", {}),
    }


def _classic_file(output_dir: Path, key: str) -> Path:
    """Return the expected classic-model output path."""
    return output_dir / CLASSIC_MODEL_FILES[key]


def _execution_outputs(output_dir: Path) -> dict[str, str]:
    """Return the standard output paths for one classic-model run."""
    return {
        "summary": str(output_dir / CLASSIC_MODEL_FILES["summary"]),
        "results": str(output_dir / CLASSIC_MODEL_FILES["results"]),
        "predictions": str(output_dir / CLASSIC_MODEL_FILES["predictions"]),
        "tuning": str(output_dir / CLASSIC_MODEL_FILES["tuning"]),
        "cv_folds": str(output_dir / CLASSIC_MODEL_FILES["cv_folds"]),
        "cv_summary": str(output_dir / CLASSIC_MODEL_FILES["cv_summary"]),
        "model_pickle": str(output_dir / CLASSIC_MODEL_FILES["model_pickle"]),
        "figures_index": str(output_dir / CLASSIC_MODEL_FILES["figures_index"]),
        "shap_dir": str(output_dir / "shap"),
    }


def _classic_optuna_output_dir(
    model_key: str,
) -> Path:
    """Resolve where an Optuna run should be created or reused."""
    target = OPTUNA_OUTPUT_BASE / OPTUNA_REAL_POLICY / model_key
    if RUN_NEW_OPTUNA_TRIALS or _classic_file(target, "summary").exists():
        return target

    raise FileNotFoundError(
        "Could not find an existing classic Optuna execution to reuse. "
        f"Searched: {target}. "
        "To generate it, set RUN_NEW_OPTUNA_TRIALS = True."
    )


def _load_classic_optuna_summary(output_dir: Path) -> dict[str, object]:
    """Read the summary JSON produced by a classic Optuna run."""
    summary_path = _classic_file(output_dir, "summary")
    if not summary_path.exists():
        raise FileNotFoundError(f"The classic Optuna summary does not exist: {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def _save_classic_optuna_summary(output_dir: Path, summary: dict[str, object]) -> None:
    """Write the combined Optuna summary JSON."""
    summary_path = output_dir / CLASSIC_MODEL_FILES["summary"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _compute_classic_shap_from_run(
    df_sofa: pd.DataFrame | None,
    output_dir: Path,
    model_key: str,
    policy: str,
) -> dict[str, dict[str, str]]:
    """Reload a trained classic model run and generate SHAP outputs."""
    if df_sofa is None:
        raise ValueError("df_sofa must be loaded to recompute SHAP in reuse mode.")

    pickle_path = _classic_file(output_dir, "model_pickle")
    if not pickle_path.exists():
        raise FileNotFoundError(f"The serialized classic Optuna model does not exist: {pickle_path}")
    # SHAP is computed from the saved model so reuse mode does not retrain it.
    with open(pickle_path, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    models = dict(bundle["models"])
    if model_key not in models:
        raise KeyError(f"The pickle does not contain model {model_key!r}. Available models: {sorted(models)}")
    model = models[model_key]

    df_model, _ = prepare_classic_model_data(df_sofa, exclude_microbiology=False)
    split_map, _ = create_chronological_episode_split(
        df_model,
        proportions=(0.70, 0.15, 0.15),
        split_unit="patient",
        real_start_date=REAL_START_DATE,
        real_overlap_policy=policy,
    )
    df_model["split"] = df_model[ID_COL].map(split_map)
    df_model = df_model.loc[df_model["split"].notna()].copy()
    df_model, _ = filter_real_from_start_date(
        df_model,
        real_start_date=REAL_START_DATE,
        enabled=True,
    )

    # Keep the same temporal partitions used by the original model evaluation.
    train = df_model.loc[df_model["split"] == "train"].copy()
    valid = df_model.loc[df_model["split"] == "valid"].copy()
    test = df_model.loc[df_model["split"] == "test"].copy()
    real = df_model.loc[df_model["split"] == "real"].copy()

    x_train = transform_features(train, preprocessor)
    x_valid = transform_features(valid, preprocessor)
    x_test = transform_features(test, preprocessor)
    x_real = transform_features(real, preprocessor) if not real.empty else None

    shap_output = _compute_and_save_classic_model_shap(
        model=model,
        model_key=model_key,
        label=_classic_model_label(output_dir, model_key),
        feature_names=list(preprocessor.feature_names),
        x_train_valid=np.vstack([x_train, x_valid]),
        x_test=x_test,
        test=test,
        x_real=x_real,
        real=real,
        output_dir=output_dir,
        sample_n=SHAP_SAMPLE_N,
        preferred_split="real",
        real_overlap_policy=policy,
        seed=42,
    )
    return {model_key: shap_output}


def _classic_model_label(output_dir: Path, model_key: str) -> str:
    """Find the human-readable label for a trained classic model."""
    results_path = _classic_file(output_dir, "results")
    if results_path.exists():
        results = pd.read_csv(results_path)
        row = results.loc[results.get("model").eq(model_key)]
        if not row.empty and "label" in row.columns:
            return str(row.iloc[0]["label"])
    return str(model_key)


def _write_optuna_best_readme(
    policy_summary: dict[str, object],
    output_dir: Path,
) -> Path:
    """Write a small Markdown index for selected Optuna runs."""
    path = output_dir / "README.md"
    lines = [
        "# Classic models 24h - Optuna best",
        "",
        "This folder contains the Optuna run for the best classic model selected for the complete 2026 real cohort.",
        "Each run saves results, tuning, the serialized model, figures, and post-Optuna SHAP inside its subfolder.",
        f"RUN_NEW_OPTUNA_TRIALS = {RUN_NEW_OPTUNA_TRIALS}; when False, the script reuses existing executions.",
        "Optuna selects hyperparameters with validation AUPRC at D+1/row level; the tuning CSV also stores episode-level AUROC/AUPRC.",
        "After Optuna, a robustness cross-validation is computed only with the final optimized configuration.",
        "Post-Optuna SHAP includes aggregated importance, beeswarm, and dependence plots.",
        "",
        "## Run",
        "",
    ]
    info = dict(policy_summary[OPTUNA_REAL_POLICY])
    outputs = dict(info.get("outputs", {}))
    lines.extend(
        [
            f"- Policy: `{OPTUNA_REAL_POLICY}`",
            f"- Selected model: `{info.get('selected_model')}`",
            f"- Optuna execution: `{info.get('optuna_execution')}`",
            f"- Optuna criterion: `{info.get('optuna_criterion')}`",
            f"- Folder: `{info.get('output_dir')}`",
            f"- JSON summary: `{outputs.get('summary')}`",
            f"- Results: `{outputs.get('results')}`",
            f"- Tuning Optuna: `{outputs.get('tuning')}`",
            f"- Post-Optuna robustness CV: `{outputs.get('cv_summary')}`",
            f"- SHAP, beeswarm and dependence: `{outputs.get('shap_dir')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _select_best_model_for_optuna() -> dict[str, object]:
    """Select the best baseline classic model for the all-patient real policy."""
    return _select_best_model(MODELS_CLASSICS_OUTPUTS_DIR / OPTUNA_REAL_POLICY)


def _select_best_model(output_dir: Path) -> dict[str, object]:
    """Select the best model row from one policy result folder."""
    tuning_path = _classic_file(output_dir, "tuning")
    results_path = _classic_file(output_dir, "results")

    if tuning_path.exists():
        # Prefer the best validation result from Optuna; test data is not used here.
        tuning = pd.read_csv(tuning_path)
        ok = tuning.loc[tuning.get("status").eq("ok")].copy()
        if not ok.empty and {"model", "valid_auprc", "valid_auroc"}.issubset(ok.columns):
            ok["valid_auprc"] = pd.to_numeric(ok["valid_auprc"], errors="coerce")
            ok["valid_auroc"] = pd.to_numeric(ok["valid_auroc"], errors="coerce")
            best_per_model = (
                ok.dropna(subset=["valid_auprc"])
                .sort_values(["valid_auprc", "valid_auroc"], ascending=[False, False])
                .groupby("model", as_index=False)
                .first()
                .sort_values(["valid_auprc", "valid_auroc"], ascending=[False, False])
            )
            if not best_per_model.empty:
                row = best_per_model.iloc[0]
                return {
                    "model": str(row["model"]),
                    "criterion": MODEL_SELECTION_CRITERION,
                    "value": float(row["valid_auprc"]),
                    "valid_auroc": float(row["valid_auroc"]),
                    "source": str(tuning_path),
                }

    if not results_path.exists():
        raise FileNotFoundError(
            "No previous results were found to select the best model: "
            f"{results_path}"
        )

    results = pd.read_csv(results_path)
    trained = results.loc[results.get("status").isin(TRAINED_STATUSES)].copy()
    if trained.empty:
        raise ValueError(f"There are no trained models in: {results_path}")

    criterion_name = "test_auprc"
    trained[criterion_name] = pd.to_numeric(trained[criterion_name], errors="coerce")
    sort_columns = [criterion_name]
    ascending = [False]
    if "test_auroc" in trained.columns:
        trained["test_auroc"] = pd.to_numeric(
            trained["test_auroc"],
            errors="coerce",
        )
        sort_columns.append("test_auroc")
        ascending.append(False)

    row = trained.dropna(subset=[criterion_name]).sort_values(
        sort_columns,
        ascending=ascending,
    ).iloc[0]
    return {
        "model": str(row["model"]),
        "criterion": criterion_name,
        "value": float(row[criterion_name]),
        "source": str(results_path),
    }


if __name__ == "__main__":
    main()


