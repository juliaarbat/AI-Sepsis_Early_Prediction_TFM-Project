"""Run the first project step: exploratory analysis of the sepsis dataset."""

# Importing _bootstrap registers the project root on sys.path before src imports.
import _bootstrap

assert _bootstrap.PROJECT_ROOT

import pandas as pd

from src.data_loading import load_sepsis_model
from src.config import EDA_GENERAL_DIR
from src.general_eda import run_general_eda
from src.progress import log_end, log_start, step


TOTAL_STEPS = 3
EDA_OUTPUT_SUBFOLDER = EDA_GENERAL_DIR.name
DATASET_SUMMARY_FIELDS = (
    ("Episodes", "Episodi", "nunique"),
    ("Patients", "Nhc", "nunique"),
    ("Minimum date", "data_index", "min"),
    ("Maximum date", "data_index", "max"),
)


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a quick sanity check before generating the EDA outputs.

    The goal is to catch obvious loading issues before running the full EDA:
    an empty dataset, missing identifiers, or an unexpected time range.
    """
    print("\nDATASET SUMMARY")
    print(f"- Rows: {len(df):,}")
    print(f"- Columns: {df.shape[1]:,}")

    for label, column, operation in DATASET_SUMMARY_FIELDS:
        if column not in df.columns:
            continue
        value = getattr(df[column], operation)()
        if operation == "nunique":
            value = f"{value:,}"
        print(f"- {label}: {value}")

    print_future_date_warning(df)


def print_future_date_warning(df: pd.DataFrame) -> None:
    """Warn when the analysis date column contains values after today."""
    if "data_index" not in df.columns:
        return

    today = pd.Timestamp.today().normalize()
    dates = pd.to_datetime(df["data_index"], errors="coerce")
    future_mask = dates > today
    future_rows = int(future_mask.sum())
    if future_rows == 0:
        return

    future_episodes = (
        int(df.loc[future_mask, "Episodi"].nunique()) if "Episodi" in df.columns else None
    )
    future_max = dates.loc[future_mask].max()
    print(
        f"- WARNING: {future_rows:,} rows have data_index after today "
        f"({today:%Y-%m-%d}); maximum future date is {future_max:%Y-%m-%d}."
    )
    if future_episodes is not None:
        print(f"- WARNING: Future-dated episodes: {future_episodes:,}")


def print_eda_outputs() -> None:
    """Show where the first step writes tables, reports, and figures."""
    print("\nEDA OUTPUTS")
    print(f"- tables and reports: {EDA_GENERAL_DIR}")
    print(f"- figures: {EDA_GENERAL_DIR / 'figures'}")


def main() -> None:
    """Run the first project step: data loading and general EDA."""
    title = "general sepsis EDA"
    log_start(title)

    # Load the pinned CSV, or the latest daily_sepsis_model_*.csv if none is pinned.
    with step("Load the daily_sepsis_model dataset", number=1, total=TOTAL_STEPS):
        df = load_sepsis_model()

    # Print a compact summary before spending time on figures and reports.
    with step("Print a basic dataset summary", number=2, total=TOTAL_STEPS):
        print_dataset_summary(df)

    # Generate CSV tables, TXT reports, and PNG figures under outputs/eda_general/.
    with step("Generate the general EDA and main figures", number=3, total=TOTAL_STEPS):
        run_general_eda(
            df,
            output_subfolder=EDA_OUTPUT_SUBFOLDER,
            title_label="eda-general",
        )
        print_eda_outputs()

    log_end(title)


if __name__ == "__main__":
    main()


