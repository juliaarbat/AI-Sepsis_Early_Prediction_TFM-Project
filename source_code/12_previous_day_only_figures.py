from __future__ import annotations

import _bootstrap  # noqa: F401

# The figure-generation logic lives in the source module; this script is only
# the command-line entry point for the previous-day-only ablation.
from src.previous_day_only_ablation_figures import main

if __name__ == "__main__":
    main()
