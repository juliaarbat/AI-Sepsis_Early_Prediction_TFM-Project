from __future__ import annotations

# Importing _bootstrap registers the project root on sys.path before src imports.
import _bootstrap

assert _bootstrap.PROJECT_ROOT

from src.classic_models_24h import CLASSIC_MODEL_FILES, generate_classic_model_figures_from_outputs
from src.classic_model_comparisons import main as generate_comparison_figures
from src.config import MODELS_CLASSICS_OUTPUTS_DIR
from src.real_policies import REAL_POLICIES
from src.progress import log_end, log_start, step


def main() -> None:
    """Regenerate classic-model figures from already saved model outputs."""
    title = "classic-model figures from saved outputs"
    log_start(title)

    with step("Generate individual figures by real-cohort policy", number=1, total=2):
        for policy in REAL_POLICIES:
            policy_key = str(policy["key"])
            output_dir = MODELS_CLASSICS_OUTPUTS_DIR / policy_key
            figures = generate_classic_model_figures_from_outputs(output_dir)
            print(f"Figures {policy_key}: {output_dir / CLASSIC_MODEL_FILES['figures_index']}")
            print(f"  n_figures: {len(figures)}")

    with step("Generate comparison figures across policies", number=2, total=2):
        generate_comparison_figures()

    log_end(title)


if __name__ == "__main__":
    main()
