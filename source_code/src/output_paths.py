from __future__ import annotations

from pathlib import Path

def deep_metrics_path(
    deep_dir: Path,
    _flat_dir: Path,
    model_key: str,
    policy: str,
    filename: str,
) -> Path:
    """Return the canonical deep-learning comparable metrics path."""
    return deep_dir / model_key / policy / filename


def deep_optuna_dirs(output_base: Path, policy: str, model_key: str) -> list[Path]:
    """Return current deep-learning Optuna directory candidates."""
    return [
        output_base / policy / model_key,
        output_base / model_key / policy,
    ]
