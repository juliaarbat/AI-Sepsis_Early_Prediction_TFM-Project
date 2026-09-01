import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _resolve_data_dir() -> Path:
    """Find the data directory, prioritizing the environment variable."""
    env_path = os.environ.get("TFM_SEPSIS_DATA_DIR")
    if env_path:
        return Path(env_path)

    candidates = [
        PROJECT_ROOT / "Data",
        WORKSPACE_ROOT / "Data",
        Path.home() / "PycharmProjects" / "TFM_Sepsis" / "Data",
    ]
    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


# Project-level input and output locations used by all analysis stages.
DATA_DIR = _resolve_data_dir()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODELS_CLASSICS_OUTPUTS_DIR = OUTPUTS_DIR / "models_classics_24h"
DEEP_LEARNING_OUTPUTS_DIR = OUTPUTS_DIR / "deep_learning_24h"

EDA_GENERAL_DIR = OUTPUTS_DIR / "eda_general"
SOFA_OUTPUTS_DIR = OUTPUTS_DIR / "sofa"
SOFA_DATASETS_DIR = SOFA_OUTPUTS_DIR / "datasets"
SOFA_PRE_DIR = SOFA_OUTPUTS_DIR / "pre_sofa"
SOFA_REPORTS_DIR = SOFA_OUTPUTS_DIR / "reports"

# Limits used when preparing measurements for SOFA computation.
SOFA_LAB_FFILL_LIMIT_DAYS = 14
SOFA_VITALS_FFILL_LIMIT_DAYS = 3
SOFA_CRITICAL_CARE_RETURN_MIN_HOURS = 72

# Set to None to keep episodes with large unexplained temporal gaps.
SOFA_MAX_UNEXPLAINED_GAP_DAYS = 30

# Shared modelling settings used by classic and deep-learning pipelines.
MODEL_EPISODE_MISSINGNESS_THRESHOLD = 0.50
PRE_SOFA_MAX_ANALYSIS_DATE = "2026-06-30"
MODEL_CV_FOLDS = 5
CLASSIC_OPTUNA_TRIALS = 30
SHAP_SAMPLE_N_DEFAULT = 512

# Date and identifier columns treated specially during preprocessing and splits.
DATE_COLUMNS = [
    "data_index",
    "DataIngres",
    "DataIniciUrgencies",
    "DataAlta",
    "data_hora_alta_critics",
    "data_plaquetes_pre_retorn_critics_3d",
    "data_creatinina_pre_retorn_critics_3d",
    "data_bilirubina_total_pre_retorn_critics_3d",
    "hemocultiu_positiu_data_extraccio",
]

ID_COLUMNS = ["Episodi", "Nhc"]


