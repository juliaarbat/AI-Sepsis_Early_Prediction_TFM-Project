from __future__ import annotations

import _bootstrap  # noqa: F401

# The reporting module contains the figure-generation logic; this script is
# only the command-line entry point used to launch it.
from src.post_optuna_final_reporting import main


if __name__ == "__main__":
    main()


