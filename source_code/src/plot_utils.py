from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.figure_style import apply_report_style


def setup_report_style() -> None:
    """Apply the shared report style to Matplotlib figures."""
    apply_report_style()


def clear_pngs(directory: Path) -> None:
    """Remove stale generated PNGs from one figure directory."""
    if not directory.exists():
        return
    for png_path in directory.glob("*.png"):
        png_path.unlink()


def annotate_bars(
    ax,
    bars,
    fmt: str,
    offset: float | None = None,
    color: str = "#24292F",
    *,
    dy: float | None = None,
) -> None:
    """Write numeric labels above or below bar plots.

    `dy` is kept as a backwards-compatible alias for `offset`.
    """
    if offset is None:
        offset = 0.0 if dy is None else dy
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=8,
            color=color,
        )


def metric_label(metric: str) -> str:
    """Return a readable metric label for figure titles and axes."""
    labels = {
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "auprc_lift": "AUPRC lift",
        "episode_auprc": "episode AUPRC",
    }
    return labels.get(metric, metric.upper())


def metric_ymax(values, unit_default: float = 1.02) -> float:
    """Choose a sensible y-axis maximum for bounded and unbounded metrics."""
    numeric = pd.to_numeric(np.asarray(values).ravel(), errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return unit_default
    max_value = float(np.max(finite))
    if max_value <= 1.0:
        return unit_default
    return max_value * 1.18


def padded_ymax(values, min_top: float = 0.12, unit_top: float = 1.0, pad: float = 1.22) -> float:
    """Choose a padded positive y-axis maximum."""
    numeric = pd.to_numeric(np.asarray(values).ravel(), errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return unit_top
    top = float(np.nanmax(finite))
    if top <= 1.0:
        return min(unit_top, max(min_top, top * pad))
    return top * 1.18



