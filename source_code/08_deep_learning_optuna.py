from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import pandas as pd

from src.data_loading import load_sepsis_model_with_sofa
from src.config import (
    DEEP_LEARNING_OUTPUTS_DIR,
    MODEL_EPISODE_MISSINGNESS_THRESHOLD,
    OUTPUTS_DIR,
    PRE_SOFA_MAX_ANALYSIS_DATE,
    SOFA_VITALS_FFILL_LIMIT_DAYS,
    SOFA_LAB_FFILL_LIMIT_DAYS,
    SOFA_MAX_UNEXPLAINED_GAP_DAYS,
)
from src.temporal_model_24h import train_and_evaluate_temporal_model_24h
from src.output_contracts import deep_metrics_filename, deep_output_paths, deep_summary_filename
from src.output_paths import deep_metrics_path, deep_optuna_dirs
from src.real_policies import REAL_ALL_2026, REAL_START_DATE_DEFAULT
from src.progress import log_end, log_start, step
from src.reporting import print_model_environment


EPISODE_MISSINGNESS_THRESHOLD = MODEL_EPISODE_MISSINGNESS_THRESHOLD
LAB_FFILL_LIMIT_DAYS = SOFA_LAB_FFILL_LIMIT_DAYS
VITALS_FFILL_LIMIT_DAYS = SOFA_VITALS_FFILL_LIMIT_DAYS
EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS = SOFA_MAX_UNEXPLAINED_GAP_DAYS
MAX_PRE_SOFA_ANALYSIS_DATE = PRE_SOFA_MAX_ANALYSIS_DATE
REAL_START_DATE = REAL_START_DATE_DEFAULT
OPTUNA_TRIALS = 15
# Keep the default in reuse mode because each Optuna run is computationally expensive.
RUN_NEW_OPTUNA_TRIALS = True
CALCULATE_SHAP_AFTER_OPTUNA = True
OPTUNA_REAL_POLICY = REAL_ALL_2026
TUNING_METRICS = (
    "valid_auprc",
    "valid_auroc",
    "valid_episode_auprc",
    "valid_episode_auroc",
)
COMMON_TRAINING_PARAMS = {
    # Optuna changes selected hyperparameters while these cohort settings stay fixed.
    "lookback_days": 10,
    "epochs": 8,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "train_parts": 1,
    "split_unit": "patient",
    "real_start_date": REAL_START_DATE,
    "evaluate_real_from_real_start": True,
    "imbalance_strategy": "pos_weight",
    "exclude_microbiology": False,
}
RECURRENT_PARAMS = {
    **COMMON_TRAINING_PARAMS,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.20,
    "recurrent_hidden_size": 64,
    "recurrent_bidirectional": False,
    "max_missing_ratio": 0.80,
    "early_stopping_patience": 4,
}

CANDIDATE_MODELS = {
    "transformer": {
        "prefix": "transformer_24h",
        "params": {
            **COMMON_TRAINING_PARAMS,
            "model_type": "transformer",
            "d_model": 48,
            "n_heads": 4,
            "n_layers": 1,
            "dropout": 0.15,
        },
    },
    "lstm": {
        "prefix": "lstm_24h",
        "params": {
            **RECURRENT_PARAMS,
            "model_type": "lstm",
        },
    },
    "rnn": {
        "prefix": "rnn_24h",
        "params": {
            **RECURRENT_PARAMS,
            "model_type": "rnn",
        },
    },
}


def main() -> None:
    """Run or reuse Optuna only for the best baseline deep-learning model."""
    log_start("Optuna only on the best deep-learning model for the complete 2026 real cohort")
    print_model_environment(("torch", "optuna"))

    # Select the architecture with validation performance before tuning it.
    selected = select_best_deep_learning_model_for_optuna()
    model_key = str(selected["model"])
    print("Model selected for Optuna:")
    print(
        f"  {OPTUNA_REAL_POLICY}: {model_key} "
        f"(valid_auprc={selected['valid_auprc']:.4f}, "
        f"valid_auroc={selected['valid_auroc']:.4f})"
    )

    total_steps = 1 + (2 if CALCULATE_SHAP_AFTER_OPTUNA else 1)
    with step("Load clean dataset with SOFA and 24h labels", number=1, total=total_steps):
        df_sofa = load_sepsis_model_with_sofa(
            episode_missingness_threshold=EPISODE_MISSINGNESS_THRESHOLD,
            lab_ffill_limit_days=LAB_FFILL_LIMIT_DAYS,
            vitals_ffill_limit_days=VITALS_FFILL_LIMIT_DAYS,
            episode_gap_exclusion_threshold_days=EPISODE_GAP_EXCLUSION_THRESHOLD_DAYS,
            max_allowed_date=MAX_PRE_SOFA_ANALYSIS_DATE,
        )

    output_base = DEEP_LEARNING_OUTPUTS_DIR / "optuna_best"
    step_number = 2
    spec = CANDIDATE_MODELS[model_key]
    output_prefix = str(spec["prefix"])
    output_dir = _optuna_output_dir(
        output_base=output_base,
        model_key=model_key,
        output_prefix=output_prefix,
        run_new_trials=RUN_NEW_OPTUNA_TRIALS,
    )
    params = dict(spec["params"])
    params["real_overlap_policy"] = OPTUNA_REAL_POLICY
    params["optuna_trials"] = OPTUNA_TRIALS
    params["output_prefix"] = output_prefix

    if RUN_NEW_OPTUNA_TRIALS:
        with step(
            f"Optuna deep learning {OPTUNA_REAL_POLICY}: {model_key}",
            number=step_number,
            total=total_steps,
            detail=f"{OPTUNA_TRIALS} trials on validation AUPRC",
        ):
            # Validation AUPRC is the primary tuning criterion because the target
            # is imbalanced; test and real-cohort results are kept for evaluation.
            summary = train_and_evaluate_temporal_model_24h(
                df_sofa,
                output_dir=output_dir,
                **params,
            )
    else:
        with step(
            f"Reuse Optuna deep learning {OPTUNA_REAL_POLICY}: {model_key}",
            number=step_number,
            total=total_steps,
            detail="No new trials are run; existing results are loaded.",
        ):
            summary = _load_optuna_summary(
                output_dir=output_dir,
                output_prefix=output_prefix,
            )
            print(
                "Reused deep-learning Optuna run:",
                output_dir / deep_summary_filename(output_prefix),
            )
            print("To launch trials again, set RUN_NEW_OPTUNA_TRIALS = True.")
    step_number += 1

    run_summary = _run_summary(
        model_key=model_key,
        selected=selected,
        summary=summary,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    if CALCULATE_SHAP_AFTER_OPTUNA:
        with step(
            f"Calculate post-Optuna SHAP {OPTUNA_REAL_POLICY}: {model_key}",
            number=step_number,
            total=total_steps,
            detail=f"SHAP inside the {REAL_ALL_2026} optuna_best subfolder",
        ):
            # Explain the saved post-Optuna model without launching another search.
            from src.deep_learning_shap_24h import calculate_deep_learning_shap

            run_summary["shap"] = calculate_deep_learning_shap(
                candidate=_shap_candidate_for_run(
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                    model_key=model_key,
                    policy=OPTUNA_REAL_POLICY,
                    selected=selected,
                ),
                df_sofa=df_sofa,
                output_dir=output_dir / "shap",
            )

    summary_payload: dict[str, object] = {OPTUNA_REAL_POLICY: run_summary}

    summary_path = output_base / "deep_learning_24h_optuna_best_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    readme_path = _write_optuna_best_readme(summary_payload, output_base)

    print("Best deep-learning Optuna summary:", summary_path)
    print("Best deep-learning Optuna index:", readme_path)
    print("Outputs:", output_base)
    log_end("Optuna only on the best deep-learning model for the complete 2026 real cohort")


def _execution_outputs(output_dir, output_prefix: str) -> dict[str, str]:
    """Return the standard output paths for one deep-learning Optuna run."""
    paths = deep_output_paths(output_dir, output_prefix)
    return {
        "summary": str(paths["summary"]),
        "tuning": str(paths["tuning"]),
        "metrics": str(paths["comparable_metrics"]),
        "predictions": str(paths["predictions"]),
        "model": str(paths["model"]),
        "figures_index": str(output_dir / f"{output_prefix}_figures_index.json"),
        "figures_dir": str(output_dir / "figures"),
        "shap_dir": str(output_dir / "shap"),
    }


def _run_summary(
    model_key: str,
    selected: dict[str, object],
    summary: dict[str, object],
    output_dir: Path,
    output_prefix: str,
) -> dict[str, object]:
    """Build the compact index saved by this script."""
    # Store both selection metadata and output paths so the run is reproducible.
    return {
        "selected_model": model_key,
        "selection": selected,
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "optuna_criterion": "valid_auprc at D+1/row level",
        "tuning_metrics": list(TUNING_METRICS),
        "cohort": summary["cohort"],
        "splits": summary["splits"],
        "split_temporal": summary["split_temporal"],
        "tuning": summary.get("tuning"),
        "outputs": _execution_outputs(output_dir, output_prefix),
    }


def _optuna_output_dir(
    output_base: Path,
    model_key: str,
    output_prefix: str,
    run_new_trials: bool,
) -> Path:
    """Resolve where the Optuna run should be created or reused."""
    target = output_base / OPTUNA_REAL_POLICY / model_key
    if run_new_trials:
        return target

    # Support the current layout and legacy layouts when reusing previous runs.
    candidates = [
        target,
        output_base / model_key / OPTUNA_REAL_POLICY,
    ]
    candidates.extend(deep_optuna_dirs(output_base, OPTUNA_REAL_POLICY, model_key)[2:])
    for candidate in candidates:
        if (candidate / deep_summary_filename(output_prefix)).exists():
            return candidate

    raise FileNotFoundError(
        "No existing deep-learning Optuna run was found to reuse. "
        f"Searched: {', '.join(str(path) for path in candidates)}. "
        "To generate it, set RUN_NEW_OPTUNA_TRIALS = True."
    )


def _load_optuna_summary(
    output_dir: Path,
    output_prefix: str,
) -> dict[str, object]:
    """Load the saved summary for one deep-learning Optuna run."""
    summary_path = output_dir / deep_summary_filename(output_prefix)
    if not summary_path.exists():
        raise FileNotFoundError(f"The deep-learning Optuna summary does not exist: {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def _shap_candidate_for_run(
    output_dir,
    output_prefix: str,
    model_key: str,
    policy: str,
    selected: dict[str, object],
) -> dict[str, object]:
    """Build the SHAP input payload for one selected Optuna run."""
    # SHAP receives the exact model, metrics and summary produced by this run.
    return {
        "model_key": model_key,
        "prefix": output_prefix,
        "policy": policy,
        "output_dir": output_dir,
        "metrics_path": output_dir / deep_metrics_filename(output_prefix),
        "model_path": deep_output_paths(output_dir, output_prefix)["model"],
        "summary_path": output_dir / deep_summary_filename(output_prefix),
        "valid_auprc": float(selected.get("valid_auprc", float("nan"))),
        "valid_auroc": float(selected.get("valid_auroc", float("nan"))),
    }


def _write_optuna_best_readme(
    summary_payload: dict[str, object],
    output_base,
):
    """Write a short Markdown index for the selected deep-learning Optuna run."""
    path = output_base / "README.md"
    lines = [
        "# Deep learning 24h - Optuna best",
        "",
        "This folder contains the Optuna run for the best sequential model selected for the complete 2026 real cohort.",
        "Each run saves comparable metrics, predictions, the `.pt` model, summary, and figures inside its subfolder.",
        "Optuna selects hyperparameters with validation AUPRC at D+1/row level; the tuning CSV also stores episode-level AUROC/AUPRC.",
        "",
        "## Run",
        "",
    ]
    info = dict(summary_payload[OPTUNA_REAL_POLICY])
    outputs = dict(info.get("outputs", {}))
    shap = dict(info.get("shap", {}))
    lines.extend(
        [
            f"- Policy: `{OPTUNA_REAL_POLICY}`",
            f"- Selected model: `{info.get('selected_model')}`",
            f"- Optuna criterion: `{info.get('optuna_criterion')}`",
            f"- Folder: `{info.get('output_dir')}`",
            f"- Summary JSON: `{outputs.get('summary')}`",
            f"- Tuning Optuna: `{outputs.get('tuning')}`",
            f"- Metrics: `{outputs.get('metrics')}`",
            f"- Predictions: `{outputs.get('predictions')}`",
            f"- Model: `{outputs.get('model')}`",
            f"- SHAP: `{outputs.get('shap_dir')}`",
        ]
    )
    if shap:
        lines.extend(
            [
                f"- SHAP table: `{shap.get('variable_importance')}`",
                f"- Directional SHAP figure: `{shap.get('direction_figure')}`",
                f"- Beeswarm SHAP: `{shap.get('beeswarm_top20')}`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def select_best_deep_learning_model_for_optuna() -> dict[str, object]:
    """Select the best baseline deep-learning model for the all-patient real policy."""
    return select_best_deep_learning_model(OPTUNA_REAL_POLICY)


def select_best_deep_learning_model(
    real_overlap_policy: str = REAL_ALL_2026,
) -> dict[str, object]:
    """Select the best baseline model for one real-cohort policy."""
    # Compare models only on the validation split; the test and real cohorts
    # must remain untouched until the final evaluation.
    rows: list[dict[str, object]] = []
    for model_key, spec in CANDIDATE_MODELS.items():
        prefix = str(spec["prefix"])
        metrics_path = _model_metrics_path(model_key, prefix, real_overlap_policy)
        if not metrics_path.exists():
            raise FileNotFoundError(
                "Previous results are missing, so the best deep-learning model cannot be selected: "
                f"{metrics_path}"
            )

        metrics = pd.read_csv(metrics_path)
        valid = metrics.loc[
            (metrics["level"] == "next_day") & (metrics["split"] == "valid")
        ].copy()
        if valid.empty:
            raise ValueError(f"There is no validation metric at: {metrics_path}")
        row = valid.iloc[0]
        rows.append(
            {
                "model": model_key,
                "real_overlap_policy": real_overlap_policy,
                "valid_auprc": float(row["auprc"]),
                "valid_auroc": float(row["auroc"]),
                "source": str(metrics_path),
            }
        )

    return sorted(
        rows,
        key=lambda item: (float(item["valid_auprc"]), float(item["valid_auroc"])),
        reverse=True,
    )[0]


def _model_metrics_path(model_key: str, prefix: str, policy: str):
    """Search organized results first and then the old structure."""
    return deep_metrics_path(
        DEEP_LEARNING_OUTPUTS_DIR,
        OUTPUTS_DIR,
        model_key,
        policy,
        deep_metrics_filename(prefix),
    )


if __name__ == "__main__":
    main()


