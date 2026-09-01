"""Shared styling for thesis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    # Okabe-Ito colorblind-safe palette.
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "teal": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    # Semantic aliases used by existing figures.
    "green": "#009E73",
    "rose": "#CC79A7",
    "gold": "#E69F00",
    # Neutral structure colors.
    "ink": "#202124",
    "muted": "#666666",
    "grid": "#E6E6E6",
    "border": "#BDBDBD",
    "panel": "#F7F7F7",
    "neutral": "#999999",
    "neutral_light": "#E5E5E5",
}


def apply_report_style() -> None:
    """Apply one clean Matplotlib style for all report-ready figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["border"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.75,
        }
    )


def save_report_figure(fig: plt.Figure, path: Path) -> str:
    """Save one figure with consistent resolution, spacing, and background."""
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)
