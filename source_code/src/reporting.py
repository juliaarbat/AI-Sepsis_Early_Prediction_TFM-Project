"""Shared console-reporting helpers for the main scripts."""

from __future__ import annotations

import json
import math
import importlib.util
import sys


def print_model_environment(packages: tuple[str, ...]) -> None:
    """Show the Python executable and availability of optional packages."""
    print("Python executable:", sys.executable)
    for package in packages:
        spec = importlib.util.find_spec(package)
        status = "found" if spec else "NOT found"
        origin = f" ({spec.origin})" if spec and spec.origin else ""
        print(f"{package}: {status}{origin}")


def print_temporal_metrics(
    summary: dict[str, object],
    *,
    show_thresholds: bool = False,
) -> None:
    """Show the main metrics for 24h temporal models."""
    if show_thresholds:
        thresholds = {
            "youden": summary["threshold_youden_valid"],
            "sensitivity_80": summary["threshold_sensitivity_80_valid"],
        }
        print("Thresholds selected only with validation:")
        print(json.dumps(thresholds, ensure_ascii=False, indent=2))

    metrics = summary["metrics"]
    _print_metrics_table(
        "Next-day level: each row predicts whether day D+1 will have sepsis",
        metrics,
        suffix="",
    )
    _print_metrics_table(
        "Episode level: predicts whether the episode will ever have sepsis",
        metrics,
        suffix="_episode",
    )

    if show_thresholds:
        print("The JSON also stores the 80% sensitivity threshold.")


def _print_metrics_table(
    title: str,
    metrics: dict[str, dict[str, object]],
    suffix: str,
) -> None:
    print(title)
    print("split              n      pos    prev     auroc    auprc   lift     sens     ppv      f1")

    for split in ("train", "valid", "test", "real"):
        key = f"{split}{suffix}"
        if key not in metrics:
            print(f"{split:<18} not available")
            continue
        values = metrics[key]
        print(
            f"{split:<18}"
            f"{values['n']:>6} "
            f"{values['positives']:>6} "
            f"{_fmt(values['prevalence']):>7} "
            f"{_fmt(values['auroc']):>8} "
            f"{_fmt(values['auprc']):>8} "
            f"{_fmt(values.get('auprc_lift')):>7} "
            f"{_fmt(values['sensitivity']):>8} "
            f"{_fmt(values['ppv']):>8} "
            f"{_fmt(values['f1']):>8}"
        )


def _fmt(value: object) -> str:
    if value is None:
        return "nan"
    value_float = float(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.3f}"



