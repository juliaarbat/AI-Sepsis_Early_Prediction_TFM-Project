from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import TypedDict

from src.config import DEEP_LEARNING_OUTPUTS_DIR
from src.deep_learning_comparisons import (
    COMPARISON_PATH,
    FIGURES_DIR as COMPARISON_FIGURES_DIR,
    INDEX_PATH as COMPARISON_INDEX_PATH,
    main as generate_comparison_figures,
)
from src.deep_learning_figures import generate_temporal_model_figures
from src.output_contracts import deep_predictions_filename, deep_summary_filename
from src.real_policies import normalize_real_policy, real_policy_keys


MODEL_PREFIXES = {
    "transformer": "transformer_24h",
    "lstm": "lstm_24h",
    "rnn": "rnn_24h",
}
class FigureRun(TypedDict):
    base_prefix: str
    output_prefix: str
    output_dir: Path


def main() -> None:
    """CLI entry point for regenerating deep-learning figures."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deep-learning figures for Transformer, LSTM, and RNN "
            "from already saved predictions."
        )
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=("all", *MODEL_PREFIXES.keys()),
        help="Temporal model to process. By default, figures are generated for all models.",
    )
    parser.add_argument(
        "--policy",
        default="all",
        help=(
            "Real policy to process. Use 'all' for all policies or a value such as "
            "real_readmitted_2026, real_new_2026, or real_all_2026."
        ),
    )
    args = parser.parse_args()

    paths = generate_deep_learning_figures(model=args.model, policy=args.policy)
    print("Deep-learning figures generated")
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def generate_deep_learning_figures(model: str = "all", policy: str = "all") -> dict[str, dict[str, str]]:
    """Generate individual and comparison figures from saved deep-learning outputs.

    `model` and `policy` can target one saved run or all available runs.
    """
    paths: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for run in _requested_runs(model=model, policy=policy):
        if not _run_outputs_exist(run["output_dir"], run["base_prefix"]):
            missing.append(run["output_prefix"])
            continue
        paths[run["output_prefix"]] = generate_temporal_model_figures(run["output_prefix"])

    if not paths:
        raise FileNotFoundError(
            "No deep-learning outputs were found for figure generation. "
            "Run scripts/06_deep_learning.py first."
        )

    generate_comparison_figures()
    paths["_comparison"] = {
        "figures_dir": str(COMPARISON_FIGURES_DIR),
        "comparison_csv": str(COMPARISON_PATH),
        "index": str(COMPARISON_INDEX_PATH),
    }

    index_path = _index_path(model=model, policy=policy)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "figures": paths,
                "missing_outputs": missing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    paths["_index"] = {"path": str(index_path)}
    return paths


def _requested_runs(model: str, policy: str) -> list[FigureRun]:
    """Build the list of model/policy runs requested by the CLI."""
    model_keys = tuple(MODEL_PREFIXES) if model == "all" else (model,)
    policy_keys = real_policy_keys() if policy == "all" else (normalize_real_policy(policy),)
    runs: list[FigureRun] = []
    for model_key in model_keys:
        base_prefix = MODEL_PREFIXES[model_key]
        for policy_key in policy_keys:
            output_dir = DEEP_LEARNING_OUTPUTS_DIR / model_key / policy_key
            runs.append(
                {
                    "base_prefix": base_prefix,
                    "output_prefix": f"{base_prefix}_{policy_key}",
                    "output_dir": output_dir,
                }
            )
    return runs


def _run_outputs_exist(output_dir: Path, base_prefix: str) -> bool:
    """Return True when the summary and prediction files are both available."""
    summary_path = output_dir / deep_summary_filename(base_prefix)
    predictions_path = output_dir / deep_predictions_filename(base_prefix)
    return summary_path.exists() and predictions_path.exists()


def _index_path(model: str, policy: str) -> Path:
    safe_model = _safe_name(model)
    safe_policy = _safe_name(policy)
    return DEEP_LEARNING_OUTPUTS_DIR / f"deep_learning_24h_figures_{safe_model}_{safe_policy}.json"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


if __name__ == "__main__":
    main()


