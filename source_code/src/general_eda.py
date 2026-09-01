"""General EDA tables, text reports, and figures for the sepsis dataset."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, OUTPUTS_DIR
from .figure_style import PALETTE, apply_report_style, save_report_figure


apply_report_style()

COLORS = {
    "ink": PALETTE["ink"],
    "muted": PALETTE["muted"],
    "line": PALETTE["border"],
    "grid": PALETTE["grid"],
    "neutral": PALETTE["neutral"],
    "neutral_light": PALETTE["neutral_light"],
    "missing": PALETTE["neutral"],
    "coverage": PALETTE["blue"],
    "cohort": PALETTE["blue"],
    "vitals": PALETTE["sky_blue"],
    "lab": PALETTE["green"],
    "microbiology": PALETTE["purple"],
    "medication": PALETTE["blue"],
    "surgery": PALETTE["orange"],
    "no_exposure": PALETTE["neutral"],
    "sofa": PALETTE["blue"],
    "sepsis": PALETTE["vermilion"],
    "no_sepsis": PALETTE["teal"],
    "predictive": PALETTE["gold"],
    "negative": PALETTE["muted"],
    "positive": PALETTE["vermilion"],
    "purple": PALETTE["purple"],
}

NICE_LABELS = {
    "Edat": "Age",
    "edat": "Age",
    "SBP": "SBP",
    "DBP": "DBP",
    "TAM": "MAP",
    "HR": "Heart rate",
    "RESP": "Respiratory rate",
    "O2SAT": "O2 saturation",
    "TEMP": "Temperature",
    "FIO2": "FiO2",
    "GLASGOW": "Glasgow",
    "DIURESIS": "Urine output",
    "porta_o2": "Oxygen support",
    "dispositius_invasius_previs": "Previous invasive devices",
    "creatinina": "Creatinine",
    "bilirubina_total": "Total bilirubin",
    "plaquetes": "Platelets",
    "leucocits": "Leukocytes",
    "hematocrit": "Hematocrit",
    "hemoglobina": "Hemoglobin",
    "pct_neutrofils": "Neutrophils",
    "granulocits_immadurs": "Immature granulocytes",
    "temps_protrombina_pct": "Prothrombin time",
    "pcr": "C-reactive protein",
    "glucosa": "Glucose",
    "urea": "Urea",
    "got_ast": "AST",
    "ph_arterial": "Arterial pH",
    "pao2_arterial": "Arterial PaO2",
    "paco2_arterial": "Arterial PaCO2",
    "bicarbonat_arterial": "Arterial bicarbonate",
    "exc_base_arterial": "Arterial base excess",
    "ph_venos": "Venous pH",
    "pao2_venos": "Venous PO2",
    "paco2_venos": "Venous PCO2",
    "bicarbonat_venos": "Venous bicarbonate",
    "exc_base_venos": "Venous base excess",
    "fibrinogen": "Fibrinogen",
    "albumina": "Albumin",
    "proteines_totals": "Total proteins",
    "troponina": "Troponin",
    "lactat_arterial": "Arterial lactate",
    "lactat_venos": "Venous lactate",
    "temps_cirurgia": "Surgery time",
    "passa_per_critics": "Critical care stay",
    "en_critics_dia": "Critical care on day",
    "temps_critics": "Critical care time",
    "temps_critics_dia": "Critical care time on day",
    "temps_cirurgia_disponible": "Surgery time available",
    "codi_servei_admissor": "Admitting service code",
    "font_admissio": "Admission source",
    "centre_origen": "Origin center",
    "diagnostic_ingres": "Admission diagnosis",
    "data_index": "Analysis date",
    "DataIngres": "Admission date",
    "DataIniciUrgencies": "Emergency department start date",
    "DataAlta": "Discharge date",
    "data_hora_alta_critics": "Critical care discharge datetime",
    "sexe": "Sex",
    "antibiotic": "Active antibiotic",
    "urgencia_cirurgia": "Urgent surgery",
    "hospitalitzacio_recent_90d": "Recent hospitalization 90d",
    "reingres_30d": "Readmission 30d",
    "cultiu_positiu_previ_90d": "Previous positive culture 90d",
    "vasopressor_qualsevol": "Vasopressor",
    "vasopressor_multiple": "Multiple vasopressors",
    "hemocultiu_positiu": "Positive blood culture",
    "ag_pneumococ": "Pneumococcal antigen",
    "ag_legionella": "Legionella antigen",
    "cirurgia": "Surgery",
    "sofa_respiratory": "Respiratory",
    "sofa_coagulation": "Coagulation",
    "sofa_hepatic": "Hepatic",
    "sofa_cardiovascular": "Cardiovascular",
    "sofa_neurologic": "Neurologic",
    "sofa_renal": "Renal",
    "COMORB_DIABETES_MELLITUS": "Diabetes mellitus",
    "COMORB_NEOPLASIA_SOLIDA": "Solid neoplasm",
    "COMORB_NEOPLASIA_HEMATOLOGICA": "Haematological neoplasm",
    "COMORB_ENOLISME_SEVER": "Severe alcohol use",
    "COMORB_CIRROSI_HEPATICA": "Liver cirrhosis",
    "COMORB_VIH_SIDA": "HIV/AIDS",
    "COMORB_TRASPLANT_ORGAN_SOLID": "Solid organ transplant",
    "COMORB_TRASPLANT_MOLL_OS": "Bone marrow transplant",
    "COMORB_AGAMMAGLOBULINEMIA": "Agammaglobulinemia",
    "COMORB_HIPOGAMMAGLOBULINEMIA": "Hypogammaglobulinemia",
    "COMORB_MALABSORTIVES": "Malabsorptive disorder",
    "COMORB_MALNUTRICIO_SEVERA": "Severe malnutrition",
    "COMORB_ASPLENIA": "Asplenia",
    "COMORB_ESPLENECTOMIA": "Splenectomy",
    "COMORB_IRC_DIALISI": "Chronic kidney disease on dialysis",
    "COMORB_NEUTROPENIA_GREU": "Severe neutropenia",
}

NUMERIC_VARIABLES = [
    "edat", "TAM", "RESP", "O2SAT", "TEMP", "GLASGOW", "DIURESIS",
    "creatinina", "bilirubina_total", "plaquetes", "leucocits",
    "lactat_arterial", "lactat_venos", "temps_cirurgia",
]

CORRELATION_VARIABLES = [
    "edat", "SBP", "DBP", "TAM", "HR", "RESP", "O2SAT", "TEMP", "FIO2",
    "GLASGOW", "creatinina", "bilirubina_total", "plaquetes", "leucocits",
    "pcr", "procalcitonina", "lactat_arterial", "lactat_venos",
]

BINARY_VARIABLES_OF_INTEREST = [
    "passa_per_critics", "antibiotic", "urgencia_cirurgia",
    "hospitalitzacio_recent_90d", "reingres_30d", "cultiu_positiu_previ_90d",
    "vasopressor_qualsevol", "hemocultiu_positiu", "vasopressor_multiple",
    "ag_pneumococ", "cirurgia", "ag_legionella",
]

COMORBIDITY_PREFIX = "COMORB_"

VARIABLE_BLOCKS = {
    "Vital signs": ["SBP", "DBP", "TAM", "HR", "RESP", "O2SAT", "TEMP", "FIO2", "DIURESIS", "GLASGOW", "porta_o2"],
    "Laboratory": ["creatinina", "bilirubina_total", "plaquetes", "leucocits", "lactat_arterial", "lactat_venos", "pcr", "procalcitonina", "albumina"],
    "Medication": ["vasopressor_qualsevol", "vasopressor_multiple", "antibiotic", "antibiotics_previs_90d"],
    "Microbiology": ["hemocultiu_positiu", "urocultiu_resultat", "ag_pneumococ", "ag_legionella", "cultiu_positiu_previ_90d"],
}

LABORATORY_VARIABLES_ALL = [
    "ph_arterial",
    "pao2_arterial",
    "paco2_arterial",
    "bicarbonat_arterial",
    "exc_base_arterial",
    "lactat_arterial",
    "ph_venos",
    "pao2_venos",
    "paco2_venos",
    "bicarbonat_venos",
    "exc_base_venos",
    "lactat_venos",
    "hematocrit",
    "hemoglobina",
    "leucocits",
    "pct_neutrofils",
    "granulocits_immadurs",
    "plaquetes",
    "fibrinogen",
    "temps_protrombina_pct",
    "pcr",
    "procalcitonina",
    "glucosa",
    "urea",
    "creatinina",
    "bilirubina_total",
    "got_ast",
    "albumina",
    "proteines_totals",
    "troponina",
]

VITAL_MONITORING_VARIABLES_ALL = [
    "SBP",
    "DBP",
    "TAM",
    "HR",
    "RESP",
    "O2SAT",
    "TEMP",
    "FIO2",
    "DIURESIS",
    "GLASGOW",
    "porta_o2",
    "dispositius_invasius_previs",
]

ICD_DIAGNOSIS_LABELS = {
    "I21.4": "Non-ST elevation myocardial infarction",
    "J18.9": "Pneumonia, unspecified",
    "I21.19": "Inferior ST elevation myocardial infarction",
    "Z51.11": "Antineoplastic chemotherapy admission",
    "E11.51": "Type 2 diabetes with peripheral angiopathy",
    "I21.09": "Anterior ST elevation myocardial infarction",
    "N17.9": "Acute kidney failure, unspecified",
    "C20": "Rectal cancer",
    "I44.2": "Complete atrioventricular block",
    "C78.7": "Secondary malignant neoplasm of liver",
    "I26.99": "Pulmonary embolism without acute cor pulmonale",
    "N10": "Acute pyelonephritis",
    "J44.0": "COPD with acute lower respiratory infection",
    "I35.0": "Aortic valve stenosis",
    "I11.0": "Hypertensive heart disease with heart failure",
}

KEY_SOFA_VARIABLES = [
    "pao2_arterial",
    "O2SAT",
    "FIO2",
    "PaFi",
    "SaFi",
    "respiratory_ratio",
    "plaquetes",
    "bilirubina_total",
    "TAM",
    "vasopressor_qualsevol",
    "vasopressor_dobutamina",
    "vasopressor_dopamina",
    "vasopressor_noradrenalina",
    "GLASGOW",
    "creatinina",
    "DIURESIS",
]

MIN_TTEST_VALUES_PER_GROUP = 100
MIN_TTEST_COVERAGE = 20.0
MIN_NUMERIC_FIGURE_OBSERVATIONS = 1_000
MIN_NUMERIC_FIGURE_COVERAGE = 20.0
MAX_VARIABLES_MISSINGNESS_HEATMAP = 14
CONTEXTUAL_MISSINGNESS_PATTERNS = (
    "_pre_retorn_critics_3d",
)

CLINICAL_REFERENCE_RANGES = {
    "edat": (0, 110),
    "SBP": (50, 260),
    "DBP": (20, 160),
    "TAM": (30, 180),
    "HR": (20, 250),
    "RESP": (4, 60),
    "O2SAT": (50, 100),
    "TEMP": (30, 43),
    "FIO2": (21, 100),
    "GLASGOW": (3, 15),
    "DIURESIS": (0, 5000),
    "creatinina": (0.1, 20),
    "bilirubina_total": (0, 50),
    "plaquetes": (1, 1500),
    "leucocits": (0.1, 200),
    "lactat_arterial": (0.1, 30),
    "lactat_venos": (0.1, 30),
    "temps_cirurgia": (0, 1440),
}

SEPSIS_TRAJECTORY_VARIABLES = [
    "plaquetes",
    "TAM",
    "creatinina",
    "O2SAT",
    "TEMP",
    "RESP",
    "leucocits",
    "sofa_total",
]

SUBGROUP_COLUMNS = [
    "sexe",
    "codi_servei_admissor",
    "font_admissio",
    "centre_origen",
    "diagnostic_ingres",
]


def run_general_eda(
    df: pd.DataFrame,
    output_subfolder: str | None = None,
    title_label: str | None = None,
) -> None:
    """Create the main EDA outputs for one dataframe.

    The workflow keeps three views separate: data quality, clinical cohort
    description, and figures that make the dataset easier to inspect before
    SOFA calculation or model training.
    """
    outputs_dir, figures_dir = _resolve_output_directories(output_subfolder)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_eda_tables(outputs_dir)
    _clear_pngs(figures_dir)
    label = title_label or "cohort"

    missingness = summarize_missingness(df)
    numeric_summary = summarize_numeric_variables(df)
    binaries = summarize_binary_variables(df)
    block_coverage = summarize_block_coverage(df)
    diagnostics = summarize_admission_diagnosis(df)
    diagnosis_groups = summarize_admission_diagnosis_groups(df)
    sofa_missingness = summarize_sofa_variables(df)
    temporal_activity = summarize_temporal_activity(df)
    correlations = summarize_clinical_correlations(df)
    cohort_flow = summarize_cohort_flow(df)
    outliers = summarize_clinical_outliers(df)

    _save_csv_excel(missingness, outputs_dir / "eda_all_variable_missingness.csv")
    _save_csv_excel(numeric_summary, outputs_dir / "eda_numeric_summary.csv")
    _save_csv_excel(binaries, outputs_dir / "eda_binary_summary.csv")
    _save_csv_excel(block_coverage, outputs_dir / "eda_block_coverage.csv")
    _save_csv_excel(diagnostics, outputs_dir / "eda_admission_diagnosis_frequencies.csv")
    _save_csv_excel(diagnosis_groups, outputs_dir / "eda_admission_diagnosis_groups.csv")
    _save_csv_excel(sofa_missingness, outputs_dir / "eda_sofa_variable_missingness.csv")
    _save_csv_excel(outliers, outputs_dir / "eda_clinical_range_outliers.csv")
    if not temporal_activity.empty:
        _save_csv_excel(temporal_activity, outputs_dir / "eda_temporal_activity.csv")
    if not correlations.empty:
        _save_csv_excel(correlations, outputs_dir / "eda_clinical_correlations.csv", index=True)

    write_text_report(df, missingness, binaries, block_coverage, diagnostics, diagnosis_groups, outputs_dir)
    save_dataset_overview_figure(
        df,
        cohort_flow,
        block_coverage,
        temporal_activity,
        missingness,
        figures_dir,
        label,
    )
    save_temporal_activity_figure(temporal_activity, figures_dir, label)
    save_top_missing_figure(df, missingness, figures_dir, label)
    save_block_coverage_figure(block_coverage, figures_dir, label)
    save_descriptive_cohort_figures(df, figures_dir, label)

    if "sofa_total" in df.columns:
        sofa_summary = summarize_sofa_results(df)
        _save_csv_excel(sofa_summary, outputs_dir / "eda_sofa_results_summary.csv")
        save_sofa_distribution_figure(df, figures_dir, label)
        save_sofa_component_figure(df, figures_dir, label)
        write_sofa_report(df, outputs_dir)

    if "sepsis" in df.columns:
        sepsis_summary = summarize_general_sepsis(df)
        sepsis_numeric_variables = numeric_summary_by_sepsis(df)
        sepsis_binaries = binary_summary_by_sepsis(df)
        sepsis_diagnosis_groups = diagnosis_group_summary_by_sepsis(df)
        ttest_sepsis = ttest_summary_by_sepsis(df)
        temporal_sepsis_summary = temporal_prevalence_summary_by_label(
            df,
            "sepsis",
            "Sepsis",
        )
        sepsis_coverage = coverage_summary_by_label(
            df,
            "sepsis",
            "No sepsis",
            "Sepsis",
        )
        sepsis_subgroups = subgroup_summary_by_label(
            df,
            "sepsis",
            "No sepsis",
            "Sepsis",
        )
        sepsis_trajectory = summarize_pre_sepsis_trajectory(df)

        _save_csv_excel(sepsis_summary, outputs_dir / "eda_sepsis_general_summary.csv")
        _save_csv_excel(sepsis_numeric_variables, outputs_dir / "eda_sepsis_numeric_comparison.csv")
        _save_csv_excel(sepsis_binaries, outputs_dir / "eda_sepsis_binary_comparison.csv")
        _save_csv_excel(
            sepsis_diagnosis_groups,
            outputs_dir / "eda_sepsis_admission_diagnosis_group_comparison.csv",
        )
        _save_csv_excel(ttest_sepsis, outputs_dir / "eda_sepsis_numeric_ttest.csv")
        if not temporal_sepsis_summary.empty:
            _save_csv_excel(temporal_sepsis_summary, outputs_dir / "eda_sepsis_temporal_prevalence.csv")
        if not sepsis_coverage.empty:
            _save_csv_excel(sepsis_coverage, outputs_dir / "eda_sepsis_coverage_by_label.csv")
        if not sepsis_subgroups.empty:
            _save_csv_excel(sepsis_subgroups, outputs_dir / "eda_sepsis_subgroup_comparison.csv")
        if not sepsis_trajectory.empty:
            _save_csv_excel(sepsis_trajectory, outputs_dir / "eda_sepsis_pre_sepsis_trajectory.csv")

        save_label_prevalence_figure(
            sepsis_summary,
            figures_dir,
            label,
            "Sepsis prevalence",
            "sepsis_prevalence.png",
            ["No sepsis", "Sepsis"],
        )
        save_ttest_figure_by_label(
            ttest_sepsis,
            figures_dir,
            label,
            "Standardized difference by sepsis status",
            "sepsis_ttest_differences.png",
        )
        save_pre_sepsis_trajectory_figure(
            sepsis_trajectory,
            figures_dir,
            label,
        )
        write_sepsis_report(df, sepsis_summary, ttest_sepsis, outputs_dir)

    if {"next_day_sepsis", "eligible_next_day_model_row"}.issubset(df.columns):
        target_predictive = "next_day_sepsis"
        predictive_eligibility = "eligible_next_day_model_row"
        label_0 = "No sepsis tomorrow"
        label_1 = "Sepsis tomorrow"
        output_prefix = "eda_next_day_sepsis"
        figure_prefix = "next_day_sepsis"
        predictive_title = "sepsis on the following day"
    else:
        target_predictive = None

    if target_predictive:
        eligible = pd.to_numeric(df[predictive_eligibility], errors="coerce") == 1
        predictive_24h_df = df.loc[eligible].copy()
        if not predictive_24h_df.empty:
            predictive_24h_summary = general_label_summary(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )
            predictive_24h_numeric_variables = numeric_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )
            predictive_24h_binary_variables = binary_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )
            predictive_24h_diagnosis_groups = diagnosis_group_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )
            predictive_24h_ttest = ttest_summary_by_label(
                predictive_24h_df,
                target_predictive,
            )
            predictive_24h_temporal_summary = temporal_prevalence_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_1,
            )
            predictive_24h_coverage = coverage_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )
            predictive_24h_subgroups = subgroup_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
            )

            _save_csv_excel(
                predictive_24h_summary,
                outputs_dir / f"{output_prefix}_general_summary.csv",
            )
            _save_csv_excel(
                predictive_24h_numeric_variables,
                outputs_dir / f"{output_prefix}_numeric_comparison.csv",
            )
            _save_csv_excel(
                predictive_24h_binary_variables,
                outputs_dir / f"{output_prefix}_binary_comparison.csv",
            )
            predictive_24h_comorbidities = binary_summary_by_label(
                predictive_24h_df,
                target_predictive,
                label_0,
                label_1,
                variables=get_comorbidity_variables(predictive_24h_df),
            )
            _save_csv_excel(
                predictive_24h_comorbidities,
                outputs_dir / f"{output_prefix}_comorbidity_comparison.csv",
            )
            _save_csv_excel(
                predictive_24h_diagnosis_groups,
                outputs_dir / f"{output_prefix}_admission_diagnosis_group_comparison.csv",
            )
            _save_csv_excel(
                predictive_24h_ttest,
                outputs_dir / f"{output_prefix}_numeric_ttest.csv",
            )
            if not predictive_24h_temporal_summary.empty:
                _save_csv_excel(
                    predictive_24h_temporal_summary,
                    outputs_dir / f"{output_prefix}_temporal_prevalence.csv",
                )
            if not predictive_24h_coverage.empty:
                _save_csv_excel(
                    predictive_24h_coverage,
                    outputs_dir / f"{output_prefix}_coverage_by_label.csv",
                )
            if not predictive_24h_subgroups.empty:
                _save_csv_excel(
                    predictive_24h_subgroups,
                    outputs_dir / f"{output_prefix}_subgroup_comparison.csv",
                )

            save_label_prevalence_figure(
                predictive_24h_summary,
                figures_dir,
                label,
                f"Prevalence of {predictive_title}",
                f"{figure_prefix}_prevalence.png",
                [label_0, label_1],
            )
            save_episode_label_prevalence_figure(
                predictive_24h_summary,
                figures_dir,
                label,
                "Episode-level prevalence of next-day sepsis",
                f"{figure_prefix}_episode_prevalence.png",
                label_1,
            )
            save_ttest_figure_by_label(
                predictive_24h_ttest,
                figures_dir,
                label,
                f"Standardized difference for {predictive_title} (t-test)",
                f"{figure_prefix}_ttest_differences.png",
            )
            write_label_report(
                predictive_24h_df,
                predictive_24h_summary,
                predictive_24h_ttest,
                outputs_dir,
                f"PREDICTIVE SUMMARY: {predictive_title.upper()}",
                label_0,
                label_1,
                f"{output_prefix}_report.txt",
            )


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values for every column."""
    total = len(df)
    rows = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        rows.append({"variable": col, "dtype": str(df[col].dtype), "n_null": n_null, "pct_null": round(100 * n_null / total, 2) if total else 0.0})
    return pd.DataFrame(rows).sort_values(["pct_null", "variable"], ascending=[False, True])


def summarize_numeric_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for key numeric variables."""
    rows = []
    for col in [c for c in NUMERIC_VARIABLES if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": col,
            "pct_null": round(100 * series.isna().sum() / len(df), 2) if len(df) else 0.0,
            "min": _round_if_value(series.min()),
            "p01": _round_if_value(series.quantile(0.01) if series.notna().any() else None),
            "median": _round_if_value(series.median()),
            "mean": _round_if_value(series.mean()),
            "p99": _round_if_value(series.quantile(0.99) if series.notna().any() else None),
            "max": _round_if_value(series.max()),
        })
    return pd.DataFrame(rows)


def summarize_binary_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize prevalence for clinical and administrative binary variables."""
    rows = []
    total = len(df)
    for col in get_binary_model_variables(df):
        series = pd.to_numeric(df[col], errors="coerce")
        n_ones = int((series == 1).sum())
        n_no_null = int(series.notna().sum())
        n_null = total - n_no_null
        rows.append(
            {
                "variable": col,
                "n_total": total,
                "n_no_null": n_no_null,
                "n_null": n_null,
                "pct_null": round(100 * n_null / total, 2) if total else 0.0,
                "n_ones": n_ones,
                "pct_ones_among_total": round(100 * n_ones / total, 2) if total else 0.0,
                "pct_ones_among_non_null": round(100 * n_ones / n_no_null, 2) if n_no_null else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "variable",
                "n_total",
                "n_no_null",
                "n_null",
                "pct_null",
                "n_ones",
                "pct_ones_among_total",
                "pct_ones_among_non_null",
            ]
        )
    return result.sort_values("pct_ones_among_total", ascending=False)


def summarize_block_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate variable coverage by broad domain using all dataset columns."""
    total = len(df)
    rows = []
    for col in df.columns:
        rows.append(
            {
                "block": _domain_for_overall_coverage_variable(col),
                "variable": col,
                "pct_no_null": (100 * df[col].notna().sum() / total) if total else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["block", "n_variables", "mean_coverage_pct"])
    detail = pd.DataFrame(rows)
    result = (
        detail.groupby("block", as_index=False)
        .agg(
            n_variables=("variable", "nunique"),
            mean_coverage_pct=("pct_no_null", "mean"),
        )
        .sort_values("mean_coverage_pct", ascending=False)
    )
    result["mean_coverage_pct"] = result["mean_coverage_pct"].round(2)
    return result


def summarize_sofa_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage for variables required by the SOFA calculation."""
    total = len(df)
    rows = []
    for col in KEY_SOFA_VARIABLES:
        if col not in df.columns:
            rows.append({"variable": col, "present_in_table": 0, "n_no_null": 0, "pct_no_null": 0.0, "pct_null": 100.0})
            continue
        n_no_null = int(df[col].notna().sum())
        pct_no_null = round(100 * n_no_null / total, 2) if total else 0.0
        rows.append(
            {
                "variable": col,
                "present_in_table": 1,
                "n_no_null": n_no_null,
                "pct_no_null": pct_no_null,
                "pct_null": round(100 - pct_no_null, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_null", ascending=False)


def summarize_temporal_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows, episodes, and patients by month."""
    if "data_index" not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df["data_index"], errors="coerce")
    tmp = df.loc[dates.notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["month"] = dates.loc[dates.notna()].dt.to_period("M").dt.to_timestamp()
    aggregations = {"rows": ("month", "size")}
    if "Episodi" in tmp.columns:
        aggregations["episodes"] = ("Episodi", "nunique")
    if "Nhc" in tmp.columns:
        aggregations["patients"] = ("Nhc", "nunique")
    return tmp.groupby("month", as_index=False).agg(**aggregations)


def summarize_clinical_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlations between clinical variables with enough data."""
    variables = [col for col in CORRELATION_VARIABLES if col in df.columns]
    if len(variables) < 2:
        return pd.DataFrame()

    numeric = df[variables].apply(pd.to_numeric, errors="coerce")
    coverage_values = numeric.notna().mean()
    numeric = numeric.loc[:, coverage_values >= 0.20]
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method="spearman")


def summarize_cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize available units and key subsets in the current dataframe."""
    rows = [_cohort_flow_row(df, "Analyzed dataset", pd.Series(True, index=df.index))]

    if "sofa_total" in df.columns:
        sofa_valid = pd.to_numeric(df["sofa_total"], errors="coerce").notna()
        rows.append(_cohort_flow_row(df, "Rows with computed SOFA", sofa_valid))

    if "sepsis" in df.columns:
        sepsis = pd.to_numeric(df["sepsis"], errors="coerce") == 1
        rows.append(_cohort_flow_row(df, "Rows classified as sepsis", sepsis))

    if "eligible_next_day_model_row" in df.columns:
        eligible = pd.to_numeric(df["eligible_next_day_model_row"], errors="coerce") == 1
        rows.append(_cohort_flow_row(df, "Rows eligible for D+1 prediction", eligible))

    if "next_day_sepsis" in df.columns:
        tomorrow_sepsis = pd.to_numeric(df["next_day_sepsis"], errors="coerce") == 1
        rows.append(_cohort_flow_row(df, "Positive rows for D+1 prediction", tomorrow_sepsis))

    return pd.DataFrame(rows)


def summarize_clinical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Audit values outside reference clinical ranges and distribution extremes."""
    rows = []
    total = len(df)
    for col, (expected_min, expected_max) in CLINICAL_REFERENCE_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        observed = series.dropna()
        if observed.empty:
            continue
        below_range = int((observed < expected_min).sum())
        above_range = int((observed > expected_max).sum())
        outside_range = below_range + above_range
        rows.append(
            {
                "variable": col,
                "n_observed": int(observed.shape[0]),
                "coverage_pct": round(100 * observed.shape[0] / total, 2) if total else 0.0,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "n_below_range": below_range,
                "n_above_range": above_range,
                "n_out_of_range": outside_range,
                "pct_out_of_range_among_observed": round(100 * outside_range / observed.shape[0], 2),
                "min": _round_if_value(observed.min()),
                "p01": _round_if_value(observed.quantile(0.01)),
                "median": _round_if_value(observed.median()),
                "p99": _round_if_value(observed.quantile(0.99)),
                "max": _round_if_value(observed.max()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["pct_out_of_range_among_observed", "n_out_of_range", "variable"],
        ascending=[False, False, True],
    )


def temporal_prevalence_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    positive_label: str,
) -> pd.DataFrame:
    """Calculate monthly label prevalence at row, episode, and patient level."""
    if "data_index" not in df.columns or label_col not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df["data_index"], errors="coerce")
    tmp = df.loc[dates.notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["month"] = dates.loc[dates.notna()].dt.to_period("M").dt.to_timestamp()
    tmp["_label"] = pd.to_numeric(tmp[label_col], errors="coerce")
    rows = []
    for month, group in tmp.groupby("month"):
        label = group["_label"]
        n_rows = len(group)
        n_pos = int((label == 1).sum())
        row = {
            "month": month,
            "label": positive_label,
            "rows": n_rows,
            "rows_positives": n_pos,
            "pct_rows_positives": round(100 * n_pos / n_rows, 2) if n_rows else 0.0,
        }
        if "Episodi" in group.columns:
            episodes = group.groupby("Episodi")["_label"].max()
            row["episodes"] = int(episodes.shape[0])
            row["positive_episodes"] = int((episodes == 1).sum())
            row["pct_positive_episodes"] = round(
                100 * row["positive_episodes"] / row["episodes"], 2
            ) if row["episodes"] else 0.0
        if "Nhc" in group.columns:
            patients = group.groupby("Nhc")["_label"].max()
            row["patients"] = int(patients.shape[0])
            row["positive_patients"] = int((patients == 1).sum())
            row["pct_positive_patients"] = round(
                100 * row["positive_patients"] / row["patients"], 2
            ) if row["patients"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def coverage_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
) -> pd.DataFrame:
    """Compare candidate-variable coverage between label groups."""
    if label_col not in df.columns:
        return pd.DataFrame()

    variables = []
    for col in NUMERIC_VARIABLES + KEY_SOFA_VARIABLES + get_binary_model_variables(df):
        if col in df.columns and col not in variables:
            variables.append(col)

    rows = []
    label = pd.to_numeric(df[label_col], errors="coerce")
    for col in variables:
        for value_label, group_label in [(0, label_0), (1, label_1)]:
            subset = df.loc[label == value_label, col]
            total = len(subset)
            n_no_null = int(subset.notna().sum())
            rows.append(
                {
                    "variable": col,
                    "group": group_label,
                    "group_n_total": total,
                    "n_no_null": n_no_null,
                    "pct_no_null": round(100 * n_no_null / total, 2) if total else 0.0,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    pivot = summary.pivot(index="variable", columns="group", values="pct_no_null")
    differences = (pivot.get(label_1, 0) - pivot.get(label_0, 0)).abs()
    summary["absolute_non_null_pct_difference"] = summary["variable"].map(differences).round(2)
    return summary.sort_values(["absolute_non_null_pct_difference", "variable", "group"], ascending=[False, True, True])


def subgroup_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
    top_n_per_column: int = 12,
) -> pd.DataFrame:
    """Compare label prevalence across the main categorical subgroups."""
    if label_col not in df.columns:
        return pd.DataFrame()

    rows = []
    label = pd.to_numeric(df[label_col], errors="coerce")
    total_positive = int((label == 1).sum())
    for col in [c for c in SUBGROUP_COLUMNS if c in df.columns]:
        series = df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        base = pd.DataFrame({"subgroup": series, "label": label}).dropna()
        if base.empty:
            continue
        top_values = base["subgroup"].value_counts().head(top_n_per_column).index
        base = base.loc[base["subgroup"].isin(top_values)]
        for subgroup, group in base.groupby("subgroup"):
            n_total = len(group)
            n_pos = int((group["label"] == 1).sum())
            rows.append(
                {
                    "subgroup_variable": col,
                    "subgroup": subgroup,
                    "n_rows": n_total,
                    "n_positive": n_pos,
                    "pct_positive_within_subgroup": round(100 * n_pos / n_total, 2) if n_total else 0.0,
                    "pct_subgroup_among_positives": round(100 * n_pos / total_positive, 2)
                    if total_positive
                    else 0.0,
                    "negative_label": label_0,
                    "positive_label": label_1,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["pct_positive_within_subgroup", "n_rows"],
        ascending=[False, False],
    )


def summarize_pre_sepsis_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize key-variable trajectories around the first sepsis day."""
    required = {"Episodi", "data_index", "first_sepsis_date"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    variables = [col for col in SEPSIS_TRAJECTORY_VARIABLES if col in df.columns]
    if not variables:
        return pd.DataFrame()

    base = df[["Episodi", "data_index", "first_sepsis_date"] + variables].copy()
    base["data_index"] = pd.to_datetime(base["data_index"], errors="coerce")
    base["first_sepsis_date"] = pd.to_datetime(base["first_sepsis_date"], errors="coerce")
    base = base.dropna(subset=["data_index", "first_sepsis_date"])
    if base.empty:
        return pd.DataFrame()

    base["relative_days_to_sepsis"] = (base["data_index"] - base["first_sepsis_date"]).dt.days
    base = base.loc[base["relative_days_to_sepsis"].between(-7, 1)]
    if base.empty:
        return pd.DataFrame()

    plot_df = base.melt(
        id_vars=["Episodi", "relative_days_to_sepsis"],
        value_vars=variables,
        var_name="variable",
        value_name="value",
    )
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["value"])
    if plot_df.empty:
        return pd.DataFrame()

    return (
        plot_df.groupby(["variable", "relative_days_to_sepsis"], as_index=False)
        .agg(
            n_rows=("value", "size"),
            episodes=("Episodi", "nunique"),
            mean=("value", "mean"),
            median=("value", "median"),
            p25=("value", lambda x: x.quantile(0.25)),
            p75=("value", lambda x: x.quantile(0.75)),
        )
        .assign(
            mean=lambda x: x["mean"].round(3),
            median=lambda x: x["median"].round(3),
            p25=lambda x: x["p25"].round(3),
            p75=lambda x: x["p75"].round(3),
        )
        .sort_values(["variable", "relative_days_to_sepsis"])
    )


def summarize_sofa_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize final SOFA variables after they have been calculated."""
    columns = [
        "sofa_respiratory",
        "sofa_coagulation",
        "sofa_hepatic",
        "sofa_cardiovascular",
        "sofa_neurologic",
        "sofa_renal",
        "sofa_total",
        "first_day_baseline_sofa",
        "first_day_delta_sofa",
    ]
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if not series.notna().any():
            continue
        rows.append(
            {
                "variable": col,
                "n_no_null": int(series.notna().sum()),
                "pct_no_null": round(100 * series.notna().mean(), 2) if len(df) else 0.0,
                "min": _round_if_value(series.min()),
                "median": _round_if_value(series.median()),
                "p99": _round_if_value(series.quantile(0.99) if series.notna().any() else None),
                "max": _round_if_value(series.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_general_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize sepsis prevalence at row and episode level."""
    return general_label_summary(df, "sepsis", "No sepsis", "Sepsis")


def numeric_summary_by_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare numeric distributions between rows with and without sepsis."""
    return numeric_summary_by_label(df, "sepsis", "No sepsis", "Sepsis")


def binary_summary_by_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare binary prevalences between rows with and without sepsis."""
    return binary_summary_by_label(df, "sepsis", "No sepsis", "Sepsis")


def diagnosis_group_summary_by_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare ICD diagnosis groups between rows with and without sepsis."""
    return diagnosis_group_summary_by_label(df, "sepsis", "No sepsis", "Sepsis")


def ttest_summary_by_sepsis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate approximate Welch t-tests for numeric variables and sepsis."""
    rows = []
    sepsis = pd.to_numeric(df["sepsis"], errors="coerce")
    for col in [c for c in NUMERIC_VARIABLES if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        group_0 = series[sepsis == 0].dropna()
        group_1 = series[sepsis == 1].dropna()
        n0 = len(group_0)
        n1 = len(group_1)
        pct0 = round(100 * n0 / int((sepsis == 0).sum()), 2) if int((sepsis == 0).sum()) else 0.0
        pct1 = round(100 * n1 / int((sepsis == 1).sum()), 2) if int((sepsis == 1).sum()) else 0.0
        include = (
            n0 >= MIN_TTEST_VALUES_PER_GROUP
            and n1 >= MIN_TTEST_VALUES_PER_GROUP
            and pct0 >= MIN_TTEST_COVERAGE
            and pct1 >= MIN_TTEST_COVERAGE
        )

        if n0 < 2 or n1 < 2:
            rows.append(
                {
                    "variable": col,
                    "n_no_sepsis": int(n0),
                    "n_sepsis": int(n1),
                    "pct_no_null_no_sepsis": pct0,
                    "pct_no_null_sepsis": pct1,
                    "no_sepsis_mean": _round_if_value(group_0.mean()),
                    "sepsis_mean": _round_if_value(group_1.mean()),
                    "mean_difference": _round_if_value(group_1.mean() - group_0.mean()),
                    "cohen_d": None,
                    "t_stat": None,
                    "approximate_p_value": None,
                    "included_in_main_ttest": 0,
                }
            )
            continue

        t_stat, p_value = _approximate_welch_ttest(group_0, group_1)
        cohen_d = _cohen_d(group_0, group_1)
        rows.append(
            {
                "variable": col,
                "n_no_sepsis": int(n0),
                "n_sepsis": int(n1),
                "pct_no_null_no_sepsis": pct0,
                "pct_no_null_sepsis": pct1,
                "no_sepsis_mean": _round_if_value(group_0.mean()),
                "sepsis_mean": _round_if_value(group_1.mean()),
                "mean_difference": _round_if_value(group_1.mean() - group_0.mean()),
                "cohen_d": _round_if_value(cohen_d),
                "t_stat": _round_if_value(t_stat),
                "approximate_p_value": p_value,
                "included_in_main_ttest": int(include),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["cohen_d_abs"] = pd.to_numeric(result["cohen_d"], errors="coerce").abs()
    return result.sort_values(
        ["included_in_main_ttest", "cohen_d_abs", "approximate_p_value", "variable"],
        ascending=[False, False, True, True],
        na_position="last",
    )


def general_label_summary(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
) -> pd.DataFrame:
    """Summarize binary-label prevalence at row and episode level."""
    total_rows = len(df)
    label = pd.to_numeric(df[label_col], errors="coerce")
    n_positive_rows = int((label == 1).sum())
    n_negative_rows = int((label == 0).sum())
    rows = [
        {
            "group": label_0,
            "n_rows": n_negative_rows,
            "pct_rows": round(100 * n_negative_rows / total_rows, 2) if total_rows else 0.0,
        },
        {
            "group": label_1,
            "n_rows": n_positive_rows,
            "pct_rows": round(100 * n_positive_rows / total_rows, 2) if total_rows else 0.0,
        },
    ]
    if "Episodi" in df.columns:
        episode_max = df.groupby("Episodi")[label_col].max()
        total_episodes = int(episode_max.shape[0])
        n_positive_episodes = int((episode_max == 1).sum())
        n_negative_episodes = int((episode_max == 0).sum())
        rows.extend(
            [
                {
                    "group": f"Episodes without {label_1.lower()}",
                    "n_rows": n_negative_episodes,
                    "pct_rows": round(100 * n_negative_episodes / total_episodes, 2)
                    if total_episodes
                    else 0.0,
                },
                {
                    "group": f"Episodes with {label_1.lower()}",
                    "n_rows": n_positive_episodes,
                    "pct_rows": round(100 * n_positive_episodes / total_episodes, 2)
                    if total_episodes
                    else 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def numeric_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
) -> pd.DataFrame:
    """Compare numeric variables between the two groups of a binary label."""
    rows = []
    label = pd.to_numeric(df[label_col], errors="coerce")
    for col in [c for c in NUMERIC_VARIABLES if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        for value_label, group_label in [(0, label_0), (1, label_1)]:
            subset = series[label == value_label]
            rows.append(
                {
                    "variable": col,
                    "group": group_label,
                    "n_no_null": int(subset.notna().sum()),
                    "pct_null": round(100 * subset.isna().sum() / len(subset), 2)
                    if len(subset)
                    else 0.0,
                    "median": _round_if_value(subset.median()),
                    "p25": _round_if_value(
                        subset.quantile(0.25) if subset.notna().any() else None
                    ),
                    "p75": _round_if_value(
                        subset.quantile(0.75) if subset.notna().any() else None
                    ),
                    "mean": _round_if_value(subset.mean()),
                }
            )
    return pd.DataFrame(rows)


def binary_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Compare binary prevalences between the two groups of a binary label."""
    rows = []
    label = pd.to_numeric(df[label_col], errors="coerce")
    variables_model = variables or get_binary_model_variables(df)
    for col in variables_model:
        series = pd.to_numeric(df[col], errors="coerce")
        for value_label, group_label in [(0, label_0), (1, label_1)]:
            subset = series[label == value_label]
            total = len(subset)
            n_no_null = int(subset.notna().sum())
            n_ones = int((subset == 1).sum())
            rows.append(
                {
                    "variable": col,
                    "group": group_label,
                    "group_n_total": total,
                    "n_no_null": n_no_null,
                    "n_null": total - n_no_null,
                    "n_ones": n_ones,
                    "pct_null": round(100 * (total - n_no_null) / total, 2) if total else 0.0,
                    "pct_ones_within_group": round(100 * n_ones / total, 2) if total else 0.0,
                    "pct_ones_among_non_null": round(100 * n_ones / n_no_null, 2) if n_no_null else 0.0,
                }
            )
    return pd.DataFrame(rows)


def diagnosis_group_summary_by_label(
    df: pd.DataFrame,
    label_col: str,
    label_0: str,
    label_1: str,
) -> pd.DataFrame:
    """Compare diagnosis groups between the two groups of a binary label."""
    if "diagnostic_ingres" not in df.columns:
        return pd.DataFrame(columns=["admission_diagnosis_group", "group", "n_rows", "pct_rows_group"])

    base = df.copy()
    base["diagnostic_ingres"] = (
        base["diagnostic_ingres"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    base = base.dropna(subset=["diagnostic_ingres", label_col])
    if base.empty:
        return pd.DataFrame(columns=["admission_diagnosis_group", "group", "n_rows", "pct_rows_group"])

    base["admission_diagnosis_group"] = base["diagnostic_ingres"].map(classify_icd_diagnosis)
    base["group"] = pd.to_numeric(base[label_col], errors="coerce").map({0: label_0, 1: label_1})

    summary = (
        base.groupby(["group", "admission_diagnosis_group"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    totals = summary.groupby("group")["n_rows"].transform("sum")
    summary["pct_rows_group"] = (100 * summary["n_rows"] / totals).round(2)
    return summary.sort_values(["group", "pct_rows_group"], ascending=[True, False])


def ttest_summary_by_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Calculate approximate Welch t-tests for a generic binary label."""
    rows = []
    label = pd.to_numeric(df[label_col], errors="coerce")
    total_0 = int((label == 0).sum())
    total_1 = int((label == 1).sum())
    for col in [c for c in NUMERIC_VARIABLES if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        group_0 = series[label == 0].dropna()
        group_1 = series[label == 1].dropna()
        n0 = len(group_0)
        n1 = len(group_1)
        pct0 = round(100 * n0 / total_0, 2) if total_0 else 0.0
        pct1 = round(100 * n1 / total_1, 2) if total_1 else 0.0
        include = (
            n0 >= MIN_TTEST_VALUES_PER_GROUP
            and n1 >= MIN_TTEST_VALUES_PER_GROUP
            and pct0 >= MIN_TTEST_COVERAGE
            and pct1 >= MIN_TTEST_COVERAGE
        )

        row = {
            "variable": col,
            "group_0_n": int(n0),
            "group_1_n": int(n1),
            "group_0_non_null_pct": pct0,
            "group_1_non_null_pct": pct1,
            "group_0_mean": _round_if_value(group_0.mean()),
            "group_1_mean": _round_if_value(group_1.mean()),
            "mean_difference": _round_if_value(group_1.mean() - group_0.mean()),
            "cohen_d": None,
            "t_stat": None,
            "approximate_p_value": None,
            "included_in_main_ttest": int(include),
        }

        if n0 >= 2 and n1 >= 2:
            t_stat, p_value = _approximate_welch_ttest(group_0, group_1)
            row["cohen_d"] = _round_if_value(_cohen_d(group_0, group_1))
            row["t_stat"] = _round_if_value(t_stat)
            row["approximate_p_value"] = p_value

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["cohen_d_abs"] = pd.to_numeric(result["cohen_d"], errors="coerce").abs()
    return result.sort_values(
        ["included_in_main_ttest", "cohen_d_abs", "approximate_p_value", "variable"],
        ascending=[False, False, True, True],
        na_position="last",
    )


def _approximate_welch_ttest(group_0: pd.Series, group_1: pd.Series) -> tuple[float | None, float | None]:
    """Calculate Welch's t-statistic with a normal-approximation p-value for large samples."""
    n0 = len(group_0)
    n1 = len(group_1)
    if n0 < 2 or n1 < 2:
        return None, None

    mean0 = group_0.mean()
    mean1 = group_1.mean()
    var0 = group_0.var(ddof=1)
    var1 = group_1.var(ddof=1)
    se = math.sqrt((var0 / n0) + (var1 / n1))
    if se == 0 or pd.isna(se):
        return None, None

    t_stat = (mean1 - mean0) / se
    # With very large samples, the two-sided normal approximation is close to Welch's t-test.
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return float(t_stat), float(p_value)


def _cohen_d(group_0: pd.Series, group_1: pd.Series) -> float | None:
    """Calculate Cohen's d effect size between two numeric groups."""
    n0 = len(group_0)
    n1 = len(group_1)
    if n0 < 2 or n1 < 2:
        return None

    var0 = group_0.var(ddof=1)
    var1 = group_1.var(ddof=1)
    pooled_num = ((n0 - 1) * var0) + ((n1 - 1) * var1)
    pooled_den = n0 + n1 - 2
    if pooled_den <= 0:
        return None

    pooled_sd = math.sqrt(pooled_num / pooled_den)
    if pooled_sd == 0 or pd.isna(pooled_sd):
        return None

    return float((group_1.mean() - group_0.mean()) / pooled_sd)


def summarize_admission_diagnosis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the most frequent admission diagnoses."""
    if "diagnostic_ingres" not in df.columns:
        return pd.DataFrame(columns=["diagnostic_ingres", "admission_diagnosis_group", "n_rows", "pct_rows"])
    series = _serie_diagnostic_ingres(df)
    if series.empty:
        return pd.DataFrame(columns=["diagnostic_ingres", "admission_diagnosis_group", "n_rows", "pct_rows"])
    summary = series.value_counts().head(top_n).rename_axis("diagnostic_ingres").reset_index(name="n_rows")
    summary["admission_diagnosis_group"] = summary["diagnostic_ingres"].map(classify_icd_diagnosis)
    summary["pct_rows"] = (100 * summary["n_rows"] / len(df)).round(2)
    return summary[["diagnostic_ingres", "admission_diagnosis_group", "n_rows", "pct_rows"]]


def summarize_admission_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Group admission diagnoses by ICD family and compute frequencies."""
    if "diagnostic_ingres" not in df.columns:
        return pd.DataFrame(columns=["admission_diagnosis_group", "n_rows", "pct_rows"])
    series = _serie_diagnostic_ingres(df)
    if series.empty:
        return pd.DataFrame(columns=["admission_diagnosis_group", "n_rows", "pct_rows"])
    groups = series.map(classify_icd_diagnosis)
    summary = groups.value_counts().rename_axis("admission_diagnosis_group").reset_index(name="n_rows")
    summary["pct_rows"] = (100 * summary["n_rows"] / len(df)).round(2)
    return summary


def classify_icd_diagnosis(code: str) -> str:
    """Classify an ICD code into a broad clinical family."""
    prefix = code[:1].upper()
    if prefix in {"A", "B"}:
        return "Infectious diseases"
    if prefix == "C" or code.startswith(("D0", "D1", "D2", "D3", "D4")):
        return "Oncology"
    if prefix == "D":
        return "Hematology and immunology"
    if prefix == "E":
        return "Endocrine and metabolic"
    if prefix == "F":
        return "Mental health"
    if prefix == "G":
        return "Neurology"
    if prefix == "H":
        return "Ophthalmology and ENT"
    if prefix == "I":
        return "Cardiovascular"
    if prefix == "J":
        return "Respiratory"
    if prefix == "K":
        return "Digestive"
    if prefix == "L":
        return "Dermatology"
    if prefix == "M":
        return "Musculoskeletal"
    if prefix == "N":
        return "Genitourinary"
    if prefix == "O":
        return "Obstetric"
    if prefix == "P":
        return "Perinatal"
    if prefix == "Q":
        return "Congenital"
    if prefix == "R":
        return "Symptoms and signs"
    if prefix in {"S", "T"}:
        return "Injury and trauma"
    if prefix in {"V", "W", "X", "Y"}:
        return "External causes"
    if prefix == "Z":
        return "Healthcare contact and follow-up"
    return "Other"


def get_comorbidity_variables(df: pd.DataFrame) -> list[str]:
    """Find comorbidity columns using the project prefix."""
    return sorted([c for c in df.columns if c.startswith(COMORBIDITY_PREFIX)])


def get_binary_model_variables(df: pd.DataFrame) -> list[str]:
    """Combine available binary variables of interest and comorbidities."""
    variables_base = [c for c in BINARY_VARIABLES_OF_INTEREST if c in df.columns]
    comorbidities = get_comorbidity_variables(df)
    return variables_base + [c for c in comorbidities if c not in variables_base]


def write_text_report(df, missingness, binaries, block_coverage, diagnostics, diagnosis_groups, outputs_dir: Path) -> None:
    """Write the base text report for the general EDA."""
    lines = [
        "GENERAL COHORT EDA",
        f"Total rows: {len(df)}",
        f"Total columns: {df.shape[1]}",
        "",
        "TOP VARIABLES WITH MOST MISSING VALUES:",
    ]
    for _, row in missingness.head(10).iterrows():
        lines.append(f"- {_label_variable(row['variable'])}: {row['pct_null']}%")
    lines.extend(["", "AVERAGE COVERAGE BY BLOCK:"])
    for _, row in block_coverage.iterrows():
        lines.append(f"- {row['block']}: {row['mean_coverage_pct']}%")
    lines.extend(["", "MOST PREVALENT BINARY VARIABLES:"])
    for _, row in binaries.head(8).iterrows():
        lines.append(f"- {_label_variable(row['variable'])}: {row['pct_ones_among_total']}%")
    if not diagnosis_groups.empty:
        lines.extend(["", "MOST FREQUENT ADMISSION DIAGNOSIS GROUPS:"])
        for _, row in diagnosis_groups.head(10).iterrows():
            lines.append(f"- {row['admission_diagnosis_group']}: {row['pct_rows']}%")
    (outputs_dir / "eda_summary_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_sofa_report(df: pd.DataFrame, outputs_dir: Path) -> None:
    """Write the text report for total SOFA, delta SOFA, and sepsis."""
    if "sofa_total" not in df.columns:
        return

    sofa_ge_2_rows = int((pd.to_numeric(df["sofa_total"], errors="coerce") >= 2).sum())
    sofa_ge_2_pct = (
        pd.to_numeric(df["sofa_total"], errors="coerce").ge(2).mean() * 100
        if "sofa_total" in df.columns
        else 0.0
    )
    sofa_ge_2_episodes = (
        df.loc[pd.to_numeric(df["sofa_total"], errors="coerce") >= 2, "Episodi"].nunique()
        if {"sofa_total", "Episodi"}.issubset(df.columns)
        else 0
    )

    delta_rows = 0
    delta_pct = 0.0
    delta_episodes = 0
    if "first_day_delta_sofa" in df.columns:
        delta_mask = pd.to_numeric(df["first_day_delta_sofa"], errors="coerce") >= 2
        delta_rows = int(delta_mask.sum())
        delta_pct = float(delta_mask.mean() * 100) if len(df) else 0.0
        if "Episodi" in df.columns:
            delta_episodes = int(df.loc[delta_mask, "Episodi"].nunique())

    sepsis_rows = int((pd.to_numeric(df["sepsis"], errors="coerce") == 1).sum()) if "sepsis" in df.columns else delta_rows
    sepsis_episodes = (
        int(df.groupby("Episodi")["sepsis"].max().sum())
        if {"Episodi", "sepsis"}.issubset(df.columns)
        else delta_episodes
    )

    text = [
        "SOFA SUMMARY",
        f"Total rows: {len(df)}",
        "",
        "Severity by total SOFA:",
        f"Rows with SOFA total >= 2: {sofa_ge_2_rows}",
        f"Percentage of rows with SOFA total >= 2: {sofa_ge_2_pct:.2f}%",
        f"Episodes with at least one SOFA total >= 2 day: {sofa_ge_2_episodes}",
    ]
    if "first_day_delta_sofa" in df.columns:
        text.extend(
            [
                "",
                "Deterioration versus the operational baseline:",
                f"Rows with delta SOFA >= 2 versus the active baseline: {delta_rows}",
                f"Percentage of rows with delta SOFA >= 2 versus the active baseline: {delta_pct:.2f}%",
                f"Episodes with at least one delta SOFA >= 2 day versus the active baseline: {delta_episodes}",
            ]
        )
    if "sepsis" in df.columns:
        text.extend(
            [
                "",
                "Current sepsis definition:",
                "Sepsis = 1 when delta SOFA >= 2 versus the active operational SOFA baseline.",
                f"Rows classified as sepsis: {sepsis_rows}",
                f"Episodes classified as sepsis: {sepsis_episodes}",
            ]
        )
    if {"first_day_baseline_sofa", "first_day_delta_sofa"}.issubset(df.columns):
        basal = pd.to_numeric(df["first_day_baseline_sofa"], errors="coerce")
        delta = pd.to_numeric(df["first_day_delta_sofa"], errors="coerce")
        text.extend(
            [
                "",
                "Baseline and delta distribution:",
                f"Operational SOFA baseline median: {_round_if_value(basal.median())}",
                f"Operational delta SOFA median: {_round_if_value(delta.median())}",
                f"Operational delta SOFA p99: {_round_if_value(delta.quantile(0.99) if delta.notna().any() else None)}",
            ]
        )
    (outputs_dir / "eda_sofa_report.txt").write_text("\n".join(text), encoding="utf-8")


def write_sepsis_report(
    df: pd.DataFrame,
    sepsis_summary: pd.DataFrame,
    ttest_sepsis: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """Write the text report for the sepsis label and main differences."""
    sepsis_row_row = sepsis_summary.loc[sepsis_summary["group"] == "Sepsis"]
    sepsis_episodes_line = sepsis_summary.loc[sepsis_summary["group"] == "Episodes with sepsis"]

    text = [
        "SEPSIS SUMMARY",
        f"Total rows: {len(df)}",
    ]

    if not sepsis_row_row.empty:
        text.append(
            f"Rows classified as sepsis: {int(sepsis_row_row['n_rows'].iloc[0])} "
            f"({float(sepsis_row_row['pct_rows'].iloc[0]):.2f}%)"
        )

    if not sepsis_episodes_line.empty:
        text.append(
            f"Episodes classified as sepsis: {int(sepsis_episodes_line['n_rows'].iloc[0])} "
            f"({float(sepsis_episodes_line['pct_rows'].iloc[0]):.2f}%)"
        )

    if not ttest_sepsis.empty:
        text.extend(["", "NUMERIC VARIABLES WITH THE LARGEST BETWEEN-GROUP DIFFERENCE:"])
        top = (
            ttest_sepsis.loc[ttest_sepsis["included_in_main_ttest"] == 1]
            .sort_values("cohen_d_abs", ascending=False, na_position="last")
            .head(8)
        )
        for _, row in top.iterrows():
            text.append(
                f"- {_label_variable(row['variable'])}: mean no sepsis={row['no_sepsis_mean']}, "
                f"mean sepsis={row['sepsis_mean']}, Cohen d={row['cohen_d']}, "
                f"p={row['approximate_p_value']}"
            )

    (outputs_dir / "eda_sepsis_report.txt").write_text("\n".join(text), encoding="utf-8")


def write_label_report(
    df: pd.DataFrame,
    label_summary: pd.DataFrame,
    ttest_label: pd.DataFrame,
    outputs_dir: Path,
    title: str,
    label_0: str,
    label_1: str,
    file_name: str,
) -> None:
    """Write a generic text report for any binary label."""
    positive_row_row = label_summary.loc[label_summary["group"] == label_1]
    positive_episodes_line = label_summary.loc[
        label_summary["group"] == f"Episodes with {label_1.lower()}"
    ]
    text = [title, f"Total rows: {len(df)}"]
    if not positive_row_row.empty:
        text.append(
            f"Rows classified as {label_1.lower()}: {int(positive_row_row['n_rows'].iloc[0])} "
            f"({float(positive_row_row['pct_rows'].iloc[0]):.2f}%)"
        )
    if not positive_episodes_line.empty:
        text.append(
            f"Episodes classified as {label_1.lower()}: {int(positive_episodes_line['n_rows'].iloc[0])} "
            f"({float(positive_episodes_line['pct_rows'].iloc[0]):.2f}%)"
        )
    if not ttest_label.empty:
        text.extend(["", "NUMERIC VARIABLES WITH THE LARGEST BETWEEN-GROUP DIFFERENCE:"])
        top = (
            ttest_label.loc[ttest_label["included_in_main_ttest"] == 1]
            .sort_values("cohen_d_abs", ascending=False, na_position="last")
            .head(8)
        )
        for _, row in top.iterrows():
            text.append(
                f"- {_label_variable(row['variable'])}: mean {label_0.lower()}={row['group_0_mean']}, "
                f"mean {label_1.lower()}={row['group_1_mean']}, "
                f"Cohen d={row['cohen_d']}, p={row['approximate_p_value']}"
            )
    (outputs_dir / file_name).write_text("\n".join(text), encoding="utf-8")


def save_dataset_overview_figure(
    df: pd.DataFrame,
    cohort_flow: pd.DataFrame,
    block_coverage: pd.DataFrame,
    temporal_activity: pd.DataFrame,
    missingness: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save a compact visual overview of size, coverage, and monthly volume."""
    fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], width_ratios=[0.95, 1.05])

    ax_cards = fig.add_subplot(grid[0, 0])
    ax_coverage = fig.add_subplot(grid[0, 1])
    ax_temporal = fig.add_subplot(grid[1, 0])
    ax_missing = fig.add_subplot(grid[1, 1])

    _draw_overview_cards(df, cohort_flow, ax_cards)
    _draw_overview_coverage(block_coverage, ax_coverage)
    _draw_overview_temporal(temporal_activity, ax_temporal)
    _draw_overview_missingness(missingness, ax_missing)

    fig.suptitle(
        "Dataset overview",
        fontsize=23,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.945,
        "Final patient-day cohort size, data availability, temporal coverage, and main missingness hotspots.",
        fontsize=12,
        color=COLORS["muted"],
    )
    save_report_figure(fig, figures_dir / "dataset_overview.png")


def _draw_overview_cards(df: pd.DataFrame, cohort_flow: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw the count cards used in the EDA overview figure."""
    ax.axis("off")
    summary = [("Patient-days", _format_int(len(df)))]
    if "Episodi" in df.columns:
        summary.append(("Episodes", _format_int(int(df["Episodi"].nunique()))))
    if "Nhc" in df.columns:
        summary.append(("Patients", _format_int(int(df["Nhc"].nunique()))))
    summary.append(("Variables", _format_int(df.shape[1])))
    if "data_index" in df.columns:
        dates = pd.to_datetime(df["data_index"], errors="coerce").dropna()
        if not dates.empty:
            valid_dates = dates.loc[dates <= pd.Timestamp.today()]
            date_series = valid_dates if not valid_dates.empty else dates
            years = sorted(date_series.dt.year.unique())
            if years:
                year_label = f"{int(years[0])}-{int(years[-1])}"
                if int(years[-1]) == pd.Timestamp.today().year:
                    year_label += " (partial)"
                summary.append(("Calendar years", year_label))

    ax.text(
        0,
        1.02,
        "Final analytical cohort",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=COLORS["ink"],
        va="bottom",
    )
    panel = FancyBboxPatch(
        (0.02, 0.08),
        0.92,
        0.72,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor=PALETTE["panel"],
        edgecolor=COLORS["line"],
        linewidth=1.0,
    )
    ax.add_patch(panel)

    row_y = [0.68, 0.55, 0.42, 0.29, 0.16]
    for index, (card_label, value) in enumerate(summary[:5]):
        y = row_y[index]
        if index > 0:
            ax.plot(
                [0.06, 0.90],
                [y + 0.07, y + 0.07],
                transform=ax.transAxes,
                color=COLORS["line"],
                linewidth=0.8,
            )
        ax.text(
            0.08,
            y,
            card_label,
            transform=ax.transAxes,
            fontsize=11.5,
            color=COLORS["muted"],
            va="center",
        )
        ax.text(
            0.88,
            y,
            value,
            transform=ax.transAxes,
            fontsize=16 if index < 4 else 14,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
            ha="right",
        )


def _draw_overview_coverage(block_coverage: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw average coverage by variable block in the overview figure."""
    if block_coverage.empty:
        _show_empty_axis(ax, "No coverage block data available")
        return
    plot_df = block_coverage.sort_values("mean_coverage_pct").copy()
    color_map = {
        "Vital signs": COLORS["vitals"],
        "Vital signs and monitoring": COLORS["vitals"],
        "Laboratory": COLORS["lab"],
        "Medication": COLORS["medication"],
        "Medication and treatment": COLORS["medication"],
        "Microbiology": COLORS["microbiology"],
        "Microbiology and colonisation": COLORS["microbiology"],
        "Surgery and critical care": COLORS["surgery"],
        "Admission and demographics": COLORS["cohort"],
        "Comorbidities": COLORS["cohort"],
    }
    colors = [color_map.get(block, COLORS["cohort"]) for block in plot_df["block"]]
    ax.barh(plot_df["block"], plot_df["mean_coverage_pct"], color=colors, edgecolor="white", linewidth=1.0)
    _set_title(ax.figure, ax, "Data availability", "Mean non-missing patient-day values by domain.")
    ax.set_xlabel("Mean non-missing values (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.axvline(50, color=COLORS["line"], linestyle=":", linewidth=1.0, zorder=0)
    _annotate_percent_bars_h(ax, plot_df["mean_coverage_pct"], offset=0.4, fontsize=9)


def _draw_overview_temporal(temporal_activity: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw monthly row and episode volume in the overview figure."""
    if temporal_activity.empty or "month" not in temporal_activity.columns:
        _show_empty_axis(ax, "No valid data_index values available")
        return
    plot_df = temporal_activity.sort_values("month").copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df["rows"] = pd.to_numeric(plot_df["rows"], errors="coerce")
    plot_df = plot_df.loc[plot_df["month"].notna()].copy()
    current_month = pd.Timestamp.today().normalize().replace(day=1)
    plot_df = plot_df.loc[plot_df["month"] <= current_month].copy()
    positive_months = plot_df.loc[plot_df["rows"].fillna(0) > 0, "month"]
    if positive_months.empty:
        _show_empty_axis(ax, "No monthly patient-day counts available")
        return
    plot_df = (
        plot_df.set_index("month")
        .reindex(pd.date_range(positive_months.min(), positive_months.max(), freq="MS"))
        .rename_axis("month")
        .reset_index()
    )
    plot_df["rows"] = pd.to_numeric(plot_df["rows"], errors="coerce")
    reference_volume = plot_df.loc[plot_df["rows"] > 0, "rows"].median()
    low_threshold = 0.5 * reference_volume if pd.notna(reference_volume) else 0
    regular_months = plot_df.loc[plot_df["rows"] >= low_threshold, "month"]
    if not regular_months.empty:
        display_start = regular_months.min()
        display_end = regular_months.max()
        for month, rows_value in plot_df.loc[plot_df["month"] > display_end, ["month", "rows"]].itertuples(index=False):
            if pd.notna(rows_value) and rows_value > 0 and month <= display_end + pd.DateOffset(months=1):
                display_end = month
        plot_df = plot_df.loc[(plot_df["month"] >= display_start) & (plot_df["month"] <= display_end)].copy()
    plot_df["rows_plot"] = plot_df["rows"].fillna(0)
    partial_mask = plot_df["rows_plot"] < low_threshold
    bar_colors = np.where(partial_mask, COLORS["neutral_light"], COLORS["cohort"])
    ax.bar(plot_df["month"], plot_df["rows_plot"], width=24, color=bar_colors, edgecolor="white", linewidth=0.7)
    if pd.notna(reference_volume):
        ax.axhline(reference_volume, color=COLORS["muted"], linestyle="--", linewidth=1.3)
    _set_title(ax.figure, ax, "Monthly patient-day volume", "Final low-volume month reflects partial 2026 extraction.")
    ax.set_xlabel("")
    ax.set_ylabel("Patient-days")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: _format_int(int(value))))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=0)


def _draw_overview_missingness(missingness: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw the top missing variables in the overview figure."""
    plot_df = (
        _missingness_for_main_figures(missingness)
        .loc[lambda data: data["pct_null"] > 0, ["variable", "pct_null"]]
        .head(8)
        .iloc[::-1]
        .copy()
    )
    if plot_df.empty:
        _show_empty_axis(ax, "No missing values found")
        return
    plot_df["variable_label"] = plot_df["variable"].map(_label_variable)
    plot_df["domain"] = plot_df["variable"].map(_domain_for_missingness_variable)
    domain_colors = {
        "Vital signs": COLORS["vitals"],
        "Laboratory": COLORS["lab"],
        "Microbiology": COLORS["microbiology"],
        "Medication": COLORS["medication"],
        "Other": COLORS["neutral"],
    }
    colors = [domain_colors.get(domain, COLORS["neutral"]) for domain in plot_df["domain"]]
    ax.barh(plot_df["variable_label"], plot_df["pct_null"], color=colors, edgecolor="white", linewidth=1.0)
    _set_title(ax.figure, ax, "Most incomplete variables", "Context-specific helper variables excluded.")
    ax.set_xlabel("% missing")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    _annotate_percent_bars_h(ax, plot_df["pct_null"], offset=0.4, fontsize=9)


def save_temporal_activity_figure(temporal_activity: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save a monthly patient-day, episode, and patient activity figure."""
    if temporal_activity.empty or "month" not in temporal_activity.columns:
        return

    plot_df = temporal_activity.sort_values("month").copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df = plot_df.loc[plot_df["month"].notna()].copy()
    if plot_df.empty:
        return

    current_month = pd.Timestamp.today().normalize().replace(day=1)
    future_rows = int(plot_df.loc[plot_df["month"] > current_month, "rows"].sum())
    plot_df = plot_df.loc[plot_df["month"] <= current_month].copy()
    if plot_df.empty:
        return

    positive_months = plot_df.loc[pd.to_numeric(plot_df["rows"], errors="coerce").fillna(0) > 0, "month"]
    if positive_months.empty:
        return

    full_months = pd.date_range(positive_months.min(), positive_months.max(), freq="MS")
    plot_df = (
        plot_df.set_index("month")
        .reindex(full_months)
        .rename_axis("month")
        .reset_index()
    )
    for count_col in ["rows", "episodes", "patients"]:
        if count_col in plot_df.columns:
            plot_df[count_col] = pd.to_numeric(plot_df[count_col], errors="coerce")

    reference_volume = plot_df.loc[plot_df["rows"] > 0, "rows"].median()
    low_volume_threshold = 0.5 * reference_volume if pd.notna(reference_volume) else 0
    regular_months = plot_df.loc[plot_df["rows"] >= low_volume_threshold, "month"]
    omitted_start_months = 0
    omitted_start_rows = 0
    if not regular_months.empty:
        display_start = regular_months.min()
        omitted_mask = plot_df["month"] < display_start
        omitted_start_months = int(omitted_mask.sum())
        omitted_start_rows = int(plot_df.loc[omitted_mask, "rows"].fillna(0).sum())
        plot_df = plot_df.loc[plot_df["month"] >= display_start].copy()
    if plot_df.empty:
        return

    trailing_omitted_rows = 0
    trailing_omitted_months = 0
    regular_months = plot_df.loc[plot_df["rows"] >= low_volume_threshold, "month"]
    if not regular_months.empty:
        display_end = regular_months.max()
        for month, rows_value in plot_df.loc[plot_df["month"] > display_end, ["month", "rows"]].itertuples(index=False):
            if pd.notna(rows_value) and rows_value > 0 and month <= display_end + pd.DateOffset(months=1):
                display_end = month
        trailing_mask = plot_df["month"] > display_end
        trailing_omitted_rows += int(plot_df.loc[trailing_mask, "rows"].fillna(0).sum())
        trailing_omitted_months += int((plot_df.loc[trailing_mask, "rows"].fillna(0) > 0).sum())
        plot_df = plot_df.loc[plot_df["month"] <= display_end].copy()
    if plot_df.empty:
        return

    plot_df["rows_plot"] = plot_df["rows"].fillna(0)
    plot_df["month_status"] = plot_df["rows_plot"].apply(
        lambda value: "Low or partial volume" if value < low_volume_threshold else "Usual volume"
    )
    status_colors = {
        "Usual volume": COLORS["coverage"],
        "Low or partial volume": COLORS["neutral_light"],
    }
    bar_colors = plot_df["month_status"].map(status_colors).fillna(COLORS["coverage"])

    fig, ax = plt.subplots(figsize=(15.5, 7.6))

    ax.bar(
        plot_df["month"],
        plot_df["rows_plot"],
        width=24,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.0,
    )
    if pd.notna(reference_volume):
        ax.axhline(
            reference_volume,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=1.7,
            label=f"Median monthly patient-days ({_format_int(int(reference_volume))})",
        )

    _set_title(
        fig,
        ax,
        "Temporal cohort activity",
        "Patient-days per month in the final analytical cohort; the last low-volume month reflects partial data extraction.",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Patient-days")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: _format_int(int(value))))

    handles = [
        Patch(facecolor=status_colors["Usual volume"], edgecolor="white", label="Regular monthly activity"),
        Patch(facecolor=status_colors["Low or partial volume"], edgecolor="white", label="Partial month"),
    ]
    if pd.notna(reference_volume):
        handles.append(Line2D([0], [0], color=COLORS["muted"], linestyle="--", linewidth=1.5, label="Median patient-days"))
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        fontsize=10,
        borderaxespad=0.0,
    )

    _style_axis(ax)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.9)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(float(plot_df["rows_plot"].max()) * 1.18, 1.0))

    partial_months = plot_df.loc[plot_df["month_status"] == "Low or partial volume", "month"]
    for month in partial_months:
        ax.axvspan(
            month - pd.Timedelta(days=14),
            month + pd.Timedelta(days=14),
            color=COLORS["neutral_light"],
            alpha=0.16,
            zorder=0,
        )

    years = sorted(plot_df["month"].dt.year.unique())
    for year in years[1:]:
        boundary = pd.Timestamp(year=int(year), month=1, day=1)
        ax.axvline(boundary, color=COLORS["line"], linewidth=0.9, alpha=0.8)

    if not partial_months.empty:
        last_partial = partial_months.max()
        partial_value = float(plot_df.loc[plot_df["month"] == last_partial, "rows_plot"].iloc[0])
        ax.annotate(
            "Partial\nextraction",
            xy=(last_partial, partial_value),
            xytext=(-58, 48),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": COLORS["ink"], "linewidth": 1.4},
            fontsize=11,
            color=COLORS["ink"],
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["line"], "alpha": 0.95},
        )

    xmin = plot_df["month"].min() - pd.Timedelta(days=18)
    xmax = plot_df["month"].max() + pd.Timedelta(days=18)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", rotation=0)

    _save_fig(fig, figures_dir / "temporal_cohort_activity.png")


def save_pre_sepsis_trajectory_figure(
    trajectory: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save median trajectories around the first sepsis day."""
    if trajectory.empty:
        return
    variables = [v for v in SEPSIS_TRAJECTORY_VARIABLES if v in set(trajectory["variable"])]
    variables = variables[:6]
    if not variables:
        return
    plot_df = trajectory.loc[trajectory["variable"].isin(variables)].copy()
    plot_df["variable_label"] = plot_df["variable"].map(_label_variable)
    g = sns.relplot(
        data=plot_df,
        x="relative_days_to_sepsis",
        y="median",
        col="variable_label",
        col_wrap=3,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.6,
        aspect=1.25,
        color=COLORS["sepsis"],
    )
    for ax in g.axes.flatten():
        ax.axvline(0, color=COLORS["ink"], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Days from first sepsis")
        ax.set_ylabel("Median")
        _style_axis(ax)
    g.set_titles("{col_name}")
    g.figure.suptitle(
        "Clinical trajectory around the first sepsis day",
        y=1.04,
        fontsize=18,
        fontweight="bold",
    )
    save_report_figure(g.figure, figures_dir / "sepsis_pre_sepsis_trajectory.png")


def save_top_missing_figure(df, missingness, figures_dir: Path, label: str) -> None:
    """Save the variables with the highest missingness percentage."""
    plot_df = (
        _missingness_for_main_figures(missingness)
        .loc[lambda data: data["pct_null"] > 0, ["variable", "pct_null"]]
        .head(20)
        .copy()
    )
    if plot_df.empty:
        return
    plot_df["variable_label"] = plot_df["variable"].map(_label_variable)
    plot_df["domain"] = plot_df["variable"].map(_domain_for_missingness_variable)
    plot_df = plot_df.iloc[::-1].copy()

    domain_colors = {
        "Vital signs": COLORS["vitals"],
        "Laboratory": COLORS["lab"],
        "Microbiology": COLORS["microbiology"],
        "Medication": COLORS["medication"],
        "Other": COLORS["neutral"],
    }
    colors = [domain_colors.get(domain, COLORS["neutral"]) for domain in plot_df["domain"]]

    fig, ax = plt.subplots(figsize=(16, 11))
    bars = ax.barh(
        plot_df["variable_label"],
        plot_df["pct_null"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        height=0.76,
    )
    _set_title(
        fig,
        ax,
        "Variables with most missing data",
        (
            f"Analyzed daily cohort: n = {_format_int(len(df))} rows. "
            "Context-specific date and critical-care return variables excluded."
        ),
    )
    ax.set_xlabel("% missing")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    for bar, value in zip(bars, plot_df["pct_null"]):
        ax.text(
            min(float(value) + 0.8, 99.2),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1f}%",
            va="center",
            ha="left" if float(value) < 98 else "right",
            fontsize=10,
            color=COLORS["ink"],
        )

    present_domains = [domain for domain in domain_colors if domain in set(plot_df["domain"])]
    legend_handles = [
        Patch(facecolor=domain_colors[domain], edgecolor="none", label=domain)
        for domain in present_domains
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.08),
            ncol=len(legend_handles),
            title="Clinical domain",
        )

    fig.text(
        0.02,
        0.015,
        "Note: date variables and critical-care return helper variables are excluded from this ranking.",
        fontsize=10,
        color=COLORS["muted"],
    )
    _save_fig(fig, figures_dir / "top_missing_variables.png")


def save_block_coverage_figure(block_coverage, figures_dir: Path, label: str) -> None:
    """Save mean coverage by broad variable domain."""
    if block_coverage.empty:
        return
    block_coverage = block_coverage.sort_values("mean_coverage_pct", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(13, max(7, 0.62 * len(block_coverage) + 2)))
    domain_colors = {
        "Admission and demographics": COLORS["cohort"],
        "Surgery and critical care": COLORS["surgery"],
        "Comorbidities": COLORS["cohort"],
        "Vital signs and monitoring": COLORS["vitals"],
        "Laboratory": COLORS["lab"],
        "Microbiology and colonisation": COLORS["microbiology"],
        "Medication and treatment": COLORS["medication"],
    }
    if "n_variables" in block_coverage.columns:
        block_coverage["block_label"] = block_coverage.apply(
            lambda row: f"{row['block']} ({int(row['n_variables'])} variables)",
            axis=1,
        )
    else:
        block_coverage["block_label"] = block_coverage["block"]
    colors = [domain_colors.get(block, COLORS["coverage"]) for block in block_coverage["block"]]
    bars = ax.barh(
        block_coverage["block_label"],
        block_coverage["mean_coverage_pct"],
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        height=0.68,
    )
    _set_title(
        fig,
        ax,
        "Data availability by clinical domain",
        f"Mean percentage of non-missing patient-day values across all {int(block_coverage['n_variables'].sum())} variables, grouped by broad domain.",
    )
    ax.set_xlabel("Mean non-missing values (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.axvline(50, color=COLORS["line"], linestyle=":", linewidth=1.0, zorder=0)
    for bar, value in zip(bars, block_coverage["mean_coverage_pct"]):
        ax.text(
            min(float(value) + 1.0, 99.2),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.1f}%",
            va="center",
            ha="left" if float(value) < 98 else "right",
            fontsize=10,
            color=COLORS["ink"],
        )
    _save_fig(fig, figures_dir / "coverage_by_domain.png")


def save_descriptive_cohort_figures(df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save descriptive EDA figures for cohort characteristics."""
    episode_df = _episode_level_dataframe(df)
    save_age_distribution_figure(episode_df, figures_dir, label)
    save_sex_distribution_figure(episode_df, figures_dir, label)
    save_age_by_sex_figure(episode_df, figures_dir, label)
    save_patient_days_per_episode_figure(df, figures_dir, label)
    save_episode_length_distribution_figure(df, episode_df, figures_dir, label)
    save_descriptive_admission_diagnosis_group_figure(episode_df, figures_dir, label)
    save_descriptive_admission_diagnosis_figure(episode_df, figures_dir, label)
    save_admission_type_distribution_figure(episode_df, figures_dir, label)
    save_surgery_exposure_distribution_figure(df, episode_df, figures_dir, label)
    save_comorbidity_prevalence_figure(df, episode_df, figures_dir, label)
    save_recent_healthcare_exposure_figure(df, episode_df, figures_dir, label)
    save_laboratory_variable_availability_figure(df, figures_dir, label)
    save_vital_signs_variable_availability_figure(df, figures_dir, label)
    save_domain_numeric_distribution_figure(
        df,
        _vital_sign_variables(df),
        figures_dir,
        label,
        "Vital signs distributions",
        "Observed patient-day values for selected routinely recorded vital signs.",
        "vital_signs_distributions.png",
    )
    save_domain_numeric_boxplot_figure(
        df,
        _vital_sign_variables(df),
        figures_dir,
        label,
        "Vital signs central ranges",
        "Selected routinely recorded vital signs; values are visually clipped to p1-p99 to keep the clinical range readable.",
        "vital_signs_boxplots.png",
    )
    save_domain_numeric_distribution_figure(
        df,
        _laboratory_variables_for_descriptive_figures(df),
        figures_dir,
        label,
        "Laboratory distributions",
        "Observed patient-day values for laboratory variables with sufficient coverage.",
        "laboratory_distributions.png",
    )
    save_domain_numeric_boxplot_figure(
        df,
        _laboratory_variables_for_descriptive_figures(df),
        figures_dir,
        label,
        "Laboratory central ranges",
        "Values are visually clipped to p1-p99 to reduce the influence of extreme values.",
        "laboratory_boxplots.png",
    )
    save_medication_indicator_prevalence_figure(df, episode_df, figures_dir, label)
    save_microbiology_indicator_prevalence_figure(df, episode_df, figures_dir, label)


def _episode_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hospitalization episode for baseline cohort description."""
    if "Episodi" not in df.columns:
        return df.copy()
    sort_columns = [col for col in ["Episodi", "data_index", "DataIngres"] if col in df.columns]
    base = df.sort_values(sort_columns).drop_duplicates("Episodi", keep="first").copy()
    return base


def _episode_binary_any(df: pd.DataFrame, episode_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summarize binary indicators as whether they ever occurred during the episode."""
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.DataFrame(columns=["variable", "pct_positive", "n_positive"])

    source = df if "Episodi" in df.columns else episode_df
    rows = []
    for col in present:
        series = pd.to_numeric(source[col], errors="coerce").fillna(0)
        if "Episodi" in source.columns:
            values = series.groupby(source["Episodi"]).max()
        else:
            values = series
        positives = int((values > 0).sum())
        total = int(values.notna().sum())
        rows.append(
            {
                "variable": col,
                "variable_label": _label_variable(col),
                "n_positive": positives,
                "pct_positive": round(100 * positives / total, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_positive", ascending=False)


def _plot_percent_bars(
    plot_df: pd.DataFrame,
    figures_dir: Path,
    file_name: str,
    label: str,
    title: str,
    subtitle: str,
    color: str,
    max_items: int | None = 14,
    include_zero: bool = False,
) -> None:
    """Save a simple horizontal percentage barplot."""
    if plot_df.empty or "pct_positive" not in plot_df.columns:
        return
    data = plot_df.copy() if include_zero else plot_df.loc[plot_df["pct_positive"] > 0].copy()
    if max_items is not None:
        data = data.head(max_items)
    data = data.iloc[::-1].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(6, 0.52 * len(data) + 2)))
    ax.barh(data["variable_label"], data["pct_positive"], color=color, edgecolor="white", linewidth=1.0)
    _set_title(fig, ax, title, subtitle)
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    ax.set_xlim(0, max(5, min(100, float(data["pct_positive"].max()) * 1.25)))
    _annotate_percent_bars_h(ax, data["pct_positive"], offset=0.25)
    _save_fig(fig, figures_dir / file_name)


def save_age_distribution_figure(episode_df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save episode-level age distribution."""
    age_col = "edat" if "edat" in episode_df.columns else "Edat" if "Edat" in episode_df.columns else None
    if age_col is None:
        return
    ages = pd.to_numeric(episode_df[age_col], errors="coerce").dropna()
    ages = ages.loc[(ages >= 0) & (ages <= 110)]
    if ages.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.histplot(ages, bins=30, kde=True, color=COLORS["cohort"], edgecolor="white", linewidth=0.6, ax=ax)
    median = ages.median()
    ax.axvline(median, color=COLORS["surgery"], linestyle="--", linewidth=2.0, label=f"Median: {median:.0f} years")
    _set_title(fig, ax, "Age distribution", f"Episode-level distribution; n = {_format_int(len(ages))} hospitalisation episodes.")
    ax.set_xlabel("Age, years")
    ax.set_ylabel("Episodes")
    ax.legend(frameon=False)
    _save_fig(fig, figures_dir / "cohort_age_distribution.png")


def save_sex_distribution_figure(episode_df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save episode-level sex distribution."""
    if "sexe" not in episode_df.columns:
        return
    series = episode_df["sexe"].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna()
    if series.empty:
        return
    counts = series.value_counts().reset_index()
    counts.columns = ["sex", "n"]
    counts["sex"] = counts["sex"].map(_label_sex_code)
    counts["pct"] = 100 * counts["n"] / counts["n"].sum()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(counts["sex"], counts["pct"], color=COLORS["cohort"], edgecolor="white", linewidth=1.0)
    _set_title(fig, ax, "Sex distribution", f"Episode-level distribution; n = {_format_int(int(counts['n'].sum()))} episodes.")
    ax.set_xlabel("")
    ax.set_ylabel("% of episodes")
    ax.set_ylim(0, max(float(counts["pct"].max()) * 1.2, 5))
    for bar, pct, n in zip(bars, counts["pct"], counts["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.0, f"{pct:.1f}%\n(n={_format_int(int(n))})", ha="center", va="bottom", fontsize=10)
    _save_fig(fig, figures_dir / "cohort_sex_distribution.png")


def save_age_by_sex_figure(episode_df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save age distribution stratified by sex."""
    age_col = "edat" if "edat" in episode_df.columns else "Edat" if "Edat" in episode_df.columns else None
    if age_col is None or "sexe" not in episode_df.columns:
        return
    plot_df = episode_df[[age_col, "sexe"]].copy()
    plot_df[age_col] = pd.to_numeric(plot_df[age_col], errors="coerce")
    plot_df["sexe"] = plot_df["sexe"].astype("string").str.strip()
    plot_df = plot_df.dropna(subset=[age_col, "sexe"])
    plot_df = plot_df.loc[(plot_df[age_col] >= 0) & (plot_df[age_col] <= 110)]
    if plot_df.empty or plot_df["sexe"].nunique() < 2:
        return
    plot_df["sex_label"] = plot_df["sexe"].map(_label_sex_code)
    order = plot_df.groupby("sex_label")[age_col].median().sort_values().index.tolist()
    palette = {
        "Female": COLORS["vitals"],
        "Male": COLORS["cohort"],
    }
    palette = {sex: palette.get(sex, COLORS["cohort"]) for sex in order}
    summary = (
        plot_df.groupby("sex_label")[age_col]
        .agg(n="count", median="median", mean="mean")
        .reindex(order)
    )
    point_df = pd.concat(
        [
            group.sample(n=min(len(group), 450), random_state=11)
            for _, group in plot_df.groupby("sex_label")
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.violinplot(
        data=plot_df,
        x="sex_label",
        y=age_col,
        hue="sex_label",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0,
        saturation=0.85,
        alpha=0.26,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x="sex_label",
        y=age_col,
        hue="sex_label",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        width=0.34,
        linewidth=1.4,
        whis=(5, 95),
        showfliers=False,
        boxprops={"alpha": 0.78, "edgecolor": COLORS["ink"]},
        whiskerprops={"color": COLORS["ink"], "linewidth": 1.2},
        capprops={"color": COLORS["ink"], "linewidth": 1.2},
        medianprops={"color": "white", "linewidth": 2.2},
        ax=ax,
    )
    sns.stripplot(
        data=point_df,
        x="sex_label",
        y=age_col,
        order=order,
        hue="sex_label",
        palette=palette,
        jitter=0.16,
        size=2.2,
        alpha=0.16,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    ax.scatter(
        range(len(summary)),
        summary["mean"],
        marker="D",
        s=54,
        color=COLORS["surgery"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Mean",
    )
    _set_title(
        fig,
        ax,
        "Age by sex",
        f"Episode-level age distribution by recorded sex; n = {_format_int(len(plot_df))} episodes.",
    )
    ax.set_xlabel("Sex")
    ax.set_ylabel("Age, years")
    y_min = max(0, float(plot_df[age_col].min()) - 5)
    y_max = min(116, max(112, float(plot_df[age_col].max()) + 10))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=COLORS["grid"], alpha=0.7)
    for i, (sex_label, row) in enumerate(summary.iterrows()):
        ax.text(
            i,
            y_max - 4.0,
            f"n={_format_int(int(row['n']))}\nmedian={row['median']:.0f} y",
            ha="center",
            va="top",
            fontsize=9.5,
            color=COLORS["ink"],
        )
    ax.legend(loc="upper right", frameon=False)
    _save_fig(fig, figures_dir / "cohort_age_by_sex.png")


def save_patient_days_per_episode_figure(df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save distribution of patient-days contributed by each episode."""
    if "Episodi" not in df.columns:
        return
    days = df.groupby("Episodi").size()
    if days.empty:
        return
    clipped = days.clip(upper=days.quantile(0.99))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.histplot(clipped, bins=30, color=COLORS["cohort"], edgecolor="white", linewidth=0.6, ax=ax)
    median = days.median()
    ax.axvline(median, color=COLORS["surgery"], linestyle="--", linewidth=2.0, label=f"Median: {median:.0f} patient-days")
    _set_title(fig, ax, "Patient-days per hospitalisation episode", "Distribution clipped at p99 for readability.")
    ax.set_xlabel("Patient-days per episode")
    ax.set_ylabel("Episodes")
    ax.legend(frameon=False)
    _save_fig(fig, figures_dir / "patient_days_per_episode_distribution.png")


def save_episode_length_distribution_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save observed hospital episode length distribution."""
    lengths = pd.Series(dtype="float64")
    if {"DataIngres", "DataAlta"}.issubset(episode_df.columns):
        start = pd.to_datetime(episode_df["DataIngres"], errors="coerce")
        end = pd.to_datetime(episode_df["DataAlta"], errors="coerce")
        lengths = (end - start).dt.total_seconds() / 86400
        lengths = lengths.dropna()
    if lengths.empty and {"Episodi", "dia_relatiu"}.issubset(df.columns):
        lengths = pd.to_numeric(df["dia_relatiu"], errors="coerce").groupby(df["Episodi"]).max().dropna() + 1
    lengths = lengths.loc[lengths >= 0]
    if lengths.empty:
        return
    clipped = lengths.clip(upper=lengths.quantile(0.99))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.histplot(clipped, bins=30, color=COLORS["cohort"], edgecolor="white", linewidth=0.6, ax=ax)
    median = lengths.median()
    ax.axvline(median, color=COLORS["surgery"], linestyle="--", linewidth=2.0, label=f"Median: {median:.0f} days")
    _set_title(fig, ax, "Episode length distribution", "Length of stay or observed episode duration, clipped at p99 for readability.")
    ax.set_xlabel("Days")
    ax.set_ylabel("Episodes")
    ax.legend(frameon=False)
    _save_fig(fig, figures_dir / "episode_length_distribution.png")


def save_descriptive_admission_diagnosis_group_figure(
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save top admission diagnosis groups at episode level."""
    if "diagnostic_ingres" not in episode_df.columns:
        return
    series = _serie_diagnostic_ingres(episode_df)
    if series.empty:
        return
    groups = series.map(classify_icd_diagnosis).value_counts().head(12)
    plot_df = groups.rename_axis("group").reset_index(name="n")
    plot_df["pct"] = 100 * plot_df["n"] / len(series)
    plot_df = plot_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(plot_df["group"], plot_df["pct"], color=COLORS["cohort"], edgecolor="white", linewidth=1.0)
    _set_title(fig, ax, "Top admission diagnosis groups", "Episode-level ICD diagnosis families at hospital admission.")
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    _annotate_percent_bars_h(ax, plot_df["pct"], offset=0.2)
    _save_fig(fig, figures_dir / "top_admission_diagnosis_groups.png")


def save_descriptive_admission_diagnosis_figure(
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save top concrete admission diagnoses at episode level."""
    if "diagnostic_ingres" not in episode_df.columns:
        return
    series = _serie_diagnostic_ingres(episode_df)
    if series.empty:
        return
    counts = series.value_counts().head(15)
    plot_df = counts.rename_axis("diagnosis").reset_index(name="n")
    plot_df["pct"] = 100 * plot_df["n"] / len(series)
    plot_df["diagnosis_label"] = plot_df["diagnosis"].map(_label_admission_diagnosis)
    plot_df = plot_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.barh(plot_df["diagnosis_label"], plot_df["pct"], color=COLORS["cohort"], edgecolor="white", linewidth=1.0)
    _set_title(fig, ax, "Top 15 admission diagnoses", "Most frequent admission diagnosis codes translated into readable clinical labels.")
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    ax.set_xlim(0, max(2.5, float(plot_df["pct"].max()) * 1.35))
    for i, value in enumerate(plot_df["pct"]):
        ax.text(float(value) + 0.04, i, f"{float(value):.1f}%", va="center", fontsize=11, color=COLORS["ink"])
    _save_fig(fig, figures_dir / "top_admission_diagnoses.png")


def save_admission_type_distribution_figure(
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save distribution of admission source/type."""
    column = "font_admissio" if "font_admissio" in episode_df.columns else "centre_origen" if "centre_origen" in episode_df.columns else None
    if column is None:
        return
    series = episode_df[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna()
    if series.empty:
        return
    counts = series.value_counts().head(12)
    plot_df = counts.rename_axis("category").reset_index(name="n")
    admission_labels = {
        "Urgències": "Emergency admission",
        "Urgencies": "Emergency admission",
        "Urgència": "Emergency admission",
        "Urgencia": "Emergency admission",
        "Programat": "Scheduled admission",
        "Programada": "Scheduled admission",
        "Scheduled": "Scheduled admission",
    }
    plot_df["category_label"] = plot_df["category"].map(lambda value: admission_labels.get(str(value), str(value)))
    plot_df["pct"] = 100 * plot_df["n"] / len(series)
    plot_df = plot_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, max(6, 0.48 * len(plot_df) + 2)))
    ax.barh(plot_df["category_label"], plot_df["pct"], color=COLORS["cohort"], edgecolor="white", linewidth=1.0)
    _set_title(fig, ax, "Admission type distribution", "Episode-level distribution of emergency versus scheduled hospital admissions.")
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    _annotate_percent_bars_h(ax, plot_df["pct"], offset=0.2)
    _save_fig(fig, figures_dir / "admission_type_distribution.png")


def save_surgery_exposure_distribution_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save mutually exclusive surgery exposure categories."""
    if "cirurgia" not in df.columns:
        return
    source = df if "Episodi" in df.columns else episode_df
    surgery = pd.to_numeric(source["cirurgia"], errors="coerce").fillna(0)
    urgent = (
        pd.to_numeric(source["urgencia_cirurgia"], errors="coerce").fillna(0)
        if "urgencia_cirurgia" in source.columns
        else pd.Series(0, index=source.index)
    )
    if "Episodi" in source.columns:
        episode_flags = pd.DataFrame({"surgery": surgery, "urgent": urgent}).groupby(source["Episodi"]).max()
    else:
        episode_flags = pd.DataFrame({"surgery": surgery, "urgent": urgent})
    if episode_flags.empty:
        return

    categories = pd.Series("No surgery", index=episode_flags.index, dtype="object")
    categories.loc[(episode_flags["surgery"] > 0) & ~(episode_flags["urgent"] > 0)] = "Non-urgent surgery"
    categories.loc[episode_flags["urgent"] > 0] = "Urgent surgery"
    order = ["No surgery", "Non-urgent surgery", "Urgent surgery"]
    counts = categories.value_counts().reindex(order, fill_value=0)
    plot_df = counts.rename_axis("category").reset_index(name="n")
    plot_df["pct"] = 100 * plot_df["n"] / len(categories)
    color_map = {
        "No surgery": COLORS["no_exposure"],
        "Non-urgent surgery": COLORS["surgery"],
        "Urgent surgery": COLORS["sepsis"],
    }
    fig, ax = plt.subplots(figsize=(13, 4.2))
    left = 0.0
    for _, row in plot_df.iterrows():
        pct = float(row["pct"])
        ax.barh(
            ["All episodes"],
            [pct],
            left=left,
            color=color_map[row["category"]],
            edgecolor="white",
            linewidth=1.2,
            height=0.56,
            label=row["category"],
        )
        label_x = left + pct / 2
        text = f"{row['category']}\n{pct:.1f}%\n(n={_format_int(int(row['n']))})"
        text_color = "white" if pct >= 12 and row["category"] != "No surgery" else COLORS["ink"]
        if pct >= 8:
            ax.text(label_x, 0, text, ha="center", va="center", fontsize=10.5, color=text_color)
        else:
            ax.text(left + pct + 0.8, 0, f"{row['category']} {pct:.1f}%", ha="left", va="center", fontsize=10.5, color=COLORS["ink"])
        left += pct
    _set_title(
        fig,
        ax,
        "Surgery exposure",
        "Episode-level split: no surgery versus non-urgent and urgent surgery.",
    )
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    _save_fig(fig, figures_dir / "surgery_exposure_distribution.png")


def save_comorbidity_prevalence_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save prevalence of all recorded comorbidity indicators."""
    variables = get_comorbidity_variables(df)
    rows = _episode_binary_any(df, episode_df, variables)
    _plot_percent_bars(
        rows,
        figures_dir,
        "comorbidity_prevalence.png",
        label,
        "Comorbidity prevalence",
        "Episode-level prevalence of all recorded comorbidity indicators.",
        COLORS["cohort"],
        max_items=None,
        include_zero=True,
    )


def save_recent_healthcare_exposure_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save recent healthcare exposure indicators."""
    variables = [
        "hospitalitzacio_recent_90d",
        "reingres_30d",
        "antibiotics_previs_90d",
        "cultiu_positiu_previ_90d",
        "colonitzacio_previa_blee",
        "colonitzacio_previa_cre",
        "colonitzacio_previa_mrsa",
        "colonitzacio_previa_vre",
    ]
    rows = _episode_binary_any(df, episode_df, variables)
    _plot_percent_bars(
        rows,
        figures_dir,
        "recent_healthcare_exposure.png",
        label,
        "Recent healthcare exposure",
        "Episode-level prevalence of recent admission, antibiotic, culture and colonisation indicators.",
        COLORS["cohort"],
        max_items=10,
    )


def _vital_sign_variables(df: pd.DataFrame) -> list[str]:
    """Select routinely measured vital sign variables for descriptive distributions."""
    variables = ["SBP", "DBP", "TAM", "HR", "RESP", "O2SAT", "TEMP"]
    return [col for col in variables if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= MIN_NUMERIC_FIGURE_OBSERVATIONS]


def _all_vital_monitoring_variables(df: pd.DataFrame) -> list[str]:
    """Return all vital-sign and monitoring variables present in the dataset."""
    return [col for col in VITAL_MONITORING_VARIABLES_ALL if col in df.columns]


def _laboratory_variables_for_descriptive_figures(df: pd.DataFrame) -> list[str]:
    """Select laboratory variables with enough observations for readable descriptive plots."""
    candidates = [
        "hematocrit",
        "hemoglobina",
        "leucocits",
        "pct_neutrofils",
        "granulocits_immadurs",
        "plaquetes",
        "temps_protrombina_pct",
        "pcr",
        "glucosa",
        "urea",
        "creatinina",
        "bilirubina_total",
        "got_ast",
    ]
    selected = []
    for col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        coverage = 100 * series.notna().mean() if len(series) else 0
        if series.notna().sum() >= MIN_NUMERIC_FIGURE_OBSERVATIONS and coverage >= MIN_NUMERIC_FIGURE_COVERAGE:
            selected.append(col)
    return selected[:10]


def _all_laboratory_variables(df: pd.DataFrame) -> list[str]:
    """Return all interpretable laboratory measurement variables present in the dataset."""
    return [col for col in LABORATORY_VARIABLES_ALL if col in df.columns]


def save_vital_signs_variable_availability_figure(
    df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save availability for all vital-sign and monitoring variables."""
    variables = _all_vital_monitoring_variables(df)
    if not variables:
        return
    rows = []
    total = len(df)
    for col in variables:
        series = df[col]
        n_observed = int(series.notna().sum())
        rows.append(
            {
                "variable": col,
                "variable_label": _label_variable(col),
                "n_observed": n_observed,
                "pct_observed": round(100 * n_observed / total, 2) if total else 0.0,
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("pct_observed", ascending=True)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, max(7, 0.46 * len(plot_df) + 2.4)))
    ax.barh(
        plot_df["variable_label"],
        plot_df["pct_observed"],
        color=COLORS["vitals"],
        edgecolor="white",
        linewidth=1.0,
    )
    _set_title(
        fig,
        ax,
        "Vital signs and monitoring availability",
        f"Non-missing patient-day values for all vital-sign and monitoring variables shown in the analytical dataset (n = {len(plot_df)} variables).",
    )
    ax.set_xlabel("Non-missing patient-day values (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        value = float(row["pct_observed"])
        text = f"{value:.1f}%"
        if value >= 92:
            ax.text(value - 1.0, i, text, va="center", ha="right", fontsize=10, color=COLORS["ink"])
        else:
            ax.text(value + 0.8, i, text, va="center", ha="left", fontsize=10, color=COLORS["ink"])
    _save_fig(fig, figures_dir / "vital_signs_variable_availability.png")


def save_laboratory_variable_availability_figure(
    df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save availability for all laboratory measurement variables."""
    variables = _all_laboratory_variables(df)
    if not variables:
        return
    rows = []
    total = len(df)
    for col in variables:
        series = pd.to_numeric(df[col], errors="coerce")
        n_observed = int(series.notna().sum())
        rows.append(
            {
                "variable": col,
                "variable_label": _label_variable(col),
                "n_observed": n_observed,
                "pct_observed": round(100 * n_observed / total, 2) if total else 0.0,
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("pct_observed", ascending=True)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, max(8, 0.38 * len(plot_df) + 2.4)))
    ax.barh(
        plot_df["variable_label"],
        plot_df["pct_observed"],
        color=COLORS["lab"],
        edgecolor="white",
        linewidth=1.0,
    )
    _set_title(
        fig,
        ax,
        "Laboratory variable availability",
        f"Non-missing patient-day values for all laboratory measurements shown in the analytical dataset (n = {len(plot_df)} variables).",
    )
    ax.set_xlabel("Non-missing patient-day values (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        value = float(row["pct_observed"])
        text = f"{value:.1f}%"
        if value >= 92:
            ax.text(value - 1.0, i, text, va="center", ha="right", fontsize=9.5, color="white")
        else:
            ax.text(value + 0.8, i, text, va="center", ha="left", fontsize=9.5, color=COLORS["ink"])
    _save_fig(fig, figures_dir / "laboratory_variable_availability.png")


def save_domain_numeric_distribution_figure(
    df: pd.DataFrame,
    variables: list[str],
    figures_dir: Path,
    label: str,
    title: str,
    subtitle: str,
    file_name: str,
) -> None:
    """Save small-multiple histograms for a numeric clinical domain."""
    if not variables:
        return
    ncols = 3
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.2 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, col in zip(axes, variables):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            ax.axis("off")
            continue
        if len(series) > 20:
            series = series.clip(series.quantile(0.01), series.quantile(0.99))
        sns.histplot(series, bins=28, ax=ax, color=_color_variable(col), edgecolor="white", linewidth=0.5)
        ax.axvline(series.median(), color=COLORS["ink"], linestyle="--", linewidth=1.4)
        ax.set_title(f"{_label_variable(col)}", fontsize=13, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Patient-days")
        _style_axis(ax)
    for ax in axes[len(variables):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.02, 0.985, subtitle + " Dashed line = median; displayed range clipped to p1-p99.", fontsize=12, color=COLORS["muted"])
    _save_fig(fig, figures_dir / file_name)


def save_domain_numeric_boxplot_figure(
    df: pd.DataFrame,
    variables: list[str],
    figures_dir: Path,
    label: str,
    title: str,
    subtitle: str,
    file_name: str,
) -> None:
    """Save small-multiple central-range boxplots for a numeric clinical domain."""
    prepared = []
    for col in variables:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        if len(series) > 20:
            series = series.clip(series.quantile(0.01), series.quantile(0.99))
        prepared.append((col, series))
    if not prepared:
        return

    ncols = 2
    nrows = math.ceil(len(prepared) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2.6 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (col, series) in zip(axes, prepared):
        sns.boxplot(
            x=series,
            ax=ax,
            color=_color_variable(col),
            width=0.34,
            linewidth=1.1,
            whis=(5, 95),
            showfliers=False,
        )
        median = float(series.median())
        ax.set_title(f"{_label_variable(col)}", fontsize=13, pad=8, loc="left")
        ax.text(
            0.99,
            0.80,
            f"Median {median:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color=COLORS["muted"],
        )
        ax.set_xlabel("Observed value")
        ax.set_yticks([])
        _style_axis(ax)
    for ax in axes[len(prepared):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.02, 0.985, subtitle + " Whiskers show p5-p95; displayed range clipped to p1-p99.", fontsize=12, color=COLORS["muted"])
    _save_fig(fig, figures_dir / file_name)


def save_medication_indicator_prevalence_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save medication and treatment indicator prevalence."""
    variables = [
        "antibiotic",
        "antibiotics_previs_90d",
        "vasopressor_qualsevol",
        "vasopressor_multiple",
        "vasopressor_dobutamina",
        "vasopressor_dopamina",
        "vasopressor_noradrenalina",
        "vasopressor_adrenalina",
    ]
    rows = _episode_binary_any(df, episode_df, variables)
    _plot_percent_bars(
        rows,
        figures_dir,
        "medication_indicator_prevalence.png",
        label,
        "Medication and treatment indicators",
        "Episode-level prevalence of medication-derived indicators.",
        COLORS["medication"],
        max_items=10,
    )


def save_microbiology_indicator_prevalence_figure(
    df: pd.DataFrame,
    episode_df: pd.DataFrame,
    figures_dir: Path,
    label: str,
) -> None:
    """Save microbiology indicator prevalence."""
    variables = [
        "hemocultiu_positiu",
        "ag_pneumococ",
        "ag_legionella",
        "cultiu_positiu_previ_90d",
        "colonitzacio_previa_blee",
        "colonitzacio_previa_cre",
        "colonitzacio_previa_mrsa",
        "colonitzacio_previa_vre",
    ]
    rows = _episode_binary_any(df, episode_df, variables)
    _plot_percent_bars(
        rows,
        figures_dir,
        "microbiology_indicator_prevalence.png",
        label,
        "Microbiology indicators",
        "Episode-level prevalence of microbiology-derived indicators.",
        COLORS["microbiology"],
        max_items=10,
    )


def save_sofa_distribution_figure(df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save the total SOFA distribution."""
    series = pd.to_numeric(df["sofa_total"], errors="coerce").dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = range(0, int(series.max()) + 2)
    sns.histplot(series, bins=bins, discrete=True, ax=ax, color=COLORS["sofa"], edgecolor="white")
    ax.axvline(2, color=COLORS["sepsis"], linestyle="--", linewidth=2, label="SOFA threshold >= 2")
    _set_title(fig, ax, "Total SOFA distribution", f"n = {_format_int(len(series))} rows with computed SOFA.")
    ax.set_xlabel("SOFA total")
    ax.set_ylabel("Number of rows")
    ax.legend(frameon=False)
    _save_fig(fig, figures_dir / "total_sofa_distribution.png")


def save_sofa_component_figure(df: pd.DataFrame, figures_dir: Path, label: str) -> None:
    """Save the mean weight of each SOFA component."""
    columns = [
        "sofa_respiratory",
        "sofa_coagulation",
        "sofa_hepatic",
        "sofa_cardiovascular",
        "sofa_neurologic",
        "sofa_renal",
    ]
    present_columns = [c for c in columns if c in df.columns]
    if not present_columns:
        return
    rows = []
    for col in present_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if not series.notna().any():
            continue
        rows.append({"component": _label_variable(col), "mean": series.mean()})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x="mean", y="component", ax=ax, color=COLORS["sofa"])
    _set_title(fig, ax, "Mean SOFA component weight", "Helps show which systems contribute most to total SOFA.")
    ax.set_xlabel("Mean score")
    ax.set_ylabel("")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["mean"] + 0.03, i, f"{row['mean']:.2f}", va="center", fontsize=10)
    _save_fig(fig, figures_dir / "mean_sofa_component_scores.png")


def save_label_prevalence_figure(
    label_summary: pd.DataFrame,
    figures_dir: Path,
    label: str,
    title: str,
    file_name: str,
    main_groups: list[str],
) -> None:
    """Save a prevalence figure for a generic binary label."""
    plot_df = label_summary.loc[label_summary["group"].isin(main_groups)].copy()
    if plot_df.empty:
        return
    palette = _palette_for_groups(main_groups)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_df, x="group", y="pct_rows", ax=ax, hue="group", palette=palette, legend=False)
    _set_title(fig, ax, title, "Label distribution in the eligible subset.")
    ax.set_xlabel("")
    ax.set_ylabel("% of rows")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["pct_rows"] + 1, f"{row['pct_rows']:.1f}%", ha="center", fontsize=11)
    _save_fig(fig, figures_dir / file_name)


def save_episode_label_prevalence_figure(
    label_summary: pd.DataFrame,
    figures_dir: Path,
    label: str,
    title: str,
    file_name: str,
    positive_label: str,
) -> None:
    """Save episode-level prevalence for a generic binary label."""
    episode_groups = [
        f"Episodes without {positive_label.lower()}",
        f"Episodes with {positive_label.lower()}",
    ]
    plot_df = label_summary.loc[label_summary["group"].isin(episode_groups)].copy()
    if plot_df.empty:
        return
    plot_df["pct_rows"] = pd.to_numeric(
        plot_df["pct_rows"].astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=["pct_rows"])
    if plot_df.empty:
        return
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=episode_groups, ordered=True)
    plot_df = plot_df.sort_values("group")
    plot_df["plot_group"] = [f"No {positive_label.lower()}", positive_label]
    palette = _palette_for_groups([f"No {positive_label.lower()}", positive_label])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_df, x="plot_group", y="pct_rows", ax=ax, hue="plot_group", palette=palette, legend=False)
    _set_title(fig, ax, title, f"Episodes with at least one positive {positive_label.lower()} label.")
    ax.set_xlabel("")
    ax.set_ylabel("% of episodes")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["pct_rows"] + 1, f"{row['pct_rows']:.1f}%", ha="center", fontsize=11)
    _save_fig(fig, figures_dir / file_name)


def save_ttest_figure_by_label(
    ttest_label: pd.DataFrame,
    figures_dir: Path,
    label: str,
    title: str,
    file_name: str,
) -> None:
    """Save standardized differences for a generic binary label."""
    if ttest_label.empty:
        return
    plot_df = (
        ttest_label.loc[ttest_label["included_in_main_ttest"] == 1]
        .dropna(subset=["cohen_d"])
        .assign(
            cohen_d_abs=lambda x: x["cohen_d"].abs(),
            direccio=lambda x: x["cohen_d"].apply(
                lambda v: "Higher in positives" if v > 0 else "Lower in positives"
            ),
            variable_label=lambda x: x["variable"].map(_label_variable),
        )
        .sort_values("cohen_d_abs", ascending=False)
    )
    if plot_df.empty:
        return
    fig_height = max(7, 0.6 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.barplot(
        data=plot_df,
        x="cohen_d",
        y="variable_label",
        hue="direccio",
        dodge=False,
        ax=ax,
        palette={"Higher in positives": COLORS["positive"], "Lower in positives": COLORS["negative"]},
    )
    _set_title(fig, ax, title, "Positive Cohen d indicates higher values in the positive group.")
    ax.set_xlabel("Cohen d (positive vs negative)")
    ax.set_ylabel("")
    ax.axvline(0, color=COLORS["ink"], linewidth=1.2, linestyle="--")
    min_effect = float(plot_df["cohen_d"].min())
    max_effect = float(plot_df["cohen_d"].max())
    x_padding = max(0.08, 0.15 * (max_effect - min_effect))
    ax.set_xlim(min_effect - x_padding, max_effect + x_padding)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        offset = 0.02 if row["cohen_d"] >= 0 else -0.02
        ha = "left" if row["cohen_d"] >= 0 else "right"
        ax.text(
            row["cohen_d"] + offset,
            i,
            f"{row['cohen_d']:.2f}",
            va="center",
            ha=ha,
            fontsize=10,
        )
    ax.legend(title="")
    _save_fig(fig, figures_dir / file_name)


def _label_variable(variable: str) -> str:
    """Convert raw column names into readable figure labels."""
    variable = str(variable)
    if variable in NICE_LABELS:
        return NICE_LABELS[variable]
    if variable.startswith(COMORBIDITY_PREFIX):
        return variable.replace(COMORBIDITY_PREFIX, "Comorb. ").replace("_", " ").title()
    return _humanize_unknown_variable(variable)


def _humanize_unknown_variable(variable: str) -> str:
    """Best-effort English label for source columns without a curated label."""
    token_labels = {
        "ag": "antigen",
        "alta": "discharge",
        "arterial": "arterial",
        "atb": "antibiotic",
        "bicarbonat": "bicarbonate",
        "bilirubina": "bilirubin",
        "centre": "center",
        "cirurgia": "surgery",
        "codi": "code",
        "colonitzacio": "colonisation",
        "creatinina": "creatinine",
        "critics": "critical care",
        "cultiu": "culture",
        "data": "date",
        "dia": "day",
        "disponible": "available",
        "dispositius": "devices",
        "edat": "age",
        "exc": "excess",
        "font": "source",
        "germen": "organism",
        "hemocultiu": "blood culture",
        "hospitalitzacio": "hospitalization",
        "immadurs": "immature",
        "ingres": "admission",
        "invasius": "invasive",
        "lactat": "lactate",
        "legionella": "legionella",
        "leucocits": "leukocytes",
        "origen": "origin",
        "paco2": "PaCO2",
        "pao2": "PaO2",
        "passa": "critical care stay",
        "plaquetes": "platelets",
        "pneumococ": "pneumococcal",
        "pre": "pre",
        "positiu": "positive",
        "previ": "previous",
        "previs": "previous",
        "protrombina": "prothrombin",
        "qualsevol": "any",
        "recent": "recent",
        "reingres": "readmission",
        "retorn": "return",
        "resultat": "result",
        "sexe": "sex",
        "servei": "service",
        "temps": "time",
        "total": "total",
        "urgencia": "urgent",
        "urocultiu": "urine culture",
        "venos": "venous",
    }
    words = str(variable).split("_")
    return " ".join(token_labels.get(word, word) for word in words)


def _label_admission_diagnosis(code: object) -> str:
    """Return a readable label for frequent admission diagnosis codes."""
    text = str(code).strip()
    return ICD_DIAGNOSIS_LABELS.get(text, text)


def _label_sex_code(value: object) -> str:
    """Return a transparent label for the recorded sex code."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text == "0":
        return "Male"
    if text == "1":
        return "Female"
    return text


def _missingness_for_main_figures(missingness: pd.DataFrame) -> pd.DataFrame:
    """Keep main missingness figures focused on broadly interpretable variables."""
    if missingness.empty or "variable" not in missingness.columns:
        return missingness.copy()

    variables = missingness["variable"].astype(str)
    contextual_mask = variables.apply(_is_contextual_missingness_variable)
    return missingness.loc[~contextual_mask].copy()


def _is_contextual_missingness_variable(variable: str) -> bool:
    """Return True for sparse-by-design date or critical-care return variables."""
    variable = str(variable)
    if variable.startswith("data_"):
        return True
    return any(pattern in variable for pattern in CONTEXTUAL_MISSINGNESS_PATTERNS)


def _domain_for_overall_coverage_variable(variable: str) -> str:
    """Assign every analytical dataset column to one broad availability domain."""
    variable = str(variable)
    if variable.startswith(COMORBIDITY_PREFIX):
        return "Comorbidities"
    if (
        variable in LABORATORY_VARIABLES_ALL
        or variable.endswith("_pre_retorn_critics_3d")
        or variable.startswith("data_plaquetes_pre_retorn_critics")
        or variable.startswith("data_creatinina_pre_retorn_critics")
        or variable.startswith("data_bilirubina_total_pre_retorn_critics")
    ):
        return "Laboratory"
    if (
        variable in VARIABLE_BLOCKS["Vital signs"]
        or variable in {"dispositius_invasius_previs"}
    ):
        return "Vital signs and monitoring"
    if (
        variable.startswith("hemocultiu")
        or "cultiu" in variable
        or "germen" in variable
        or variable.startswith("ag_")
        or variable.startswith("colonitzacio_")
    ):
        return "Microbiology and colonisation"
    if (
        variable.startswith("vasopressor_")
        or variable in {"antibiotic", "atb_duracio", "antibiotics_previs_90d"}
    ):
        return "Medication and treatment"
    if (
        "critics" in variable
        or variable in {"cirurgia", "urgencia_cirurgia", "temps_cirurgia", "temps_cirurgia_disponible"}
    ):
        return "Surgery and critical care"
    return "Admission and demographics"


def _domain_for_missingness_variable(variable: str) -> str:
    """Assign a broad clinical domain for missingness figures."""
    variable = str(variable)
    if variable in VARIABLE_BLOCKS["Vital signs"]:
        return "Vital signs"
    if variable in VARIABLE_BLOCKS["Medication"] or variable.startswith("vasopressor_") or "antibiotic" in variable:
        return "Medication"
    if (
        variable in VARIABLE_BLOCKS["Microbiology"]
        or "cultiu" in variable
        or "germen" in variable
        or variable.startswith("ag_")
        or variable.startswith("hemocultiu")
    ):
        return "Microbiology"
    laboratory_keywords = (
        "arterial",
        "venos",
        "lactat",
        "albumina",
        "fibrinogen",
        "troponina",
        "procalcitonina",
        "proteines",
        "plaquetes",
        "creatinina",
        "bilirubina",
        "leucocits",
        "hematocrit",
        "hemoglobina",
        "pcr",
        "glucosa",
        "urea",
    )
    if variable in VARIABLE_BLOCKS["Laboratory"] or any(keyword in variable for keyword in laboratory_keywords):
        return "Laboratory"
    return "Other"


def _color_variable(variable: str) -> str:
    """Choose a semantic color based on the clinical variable type."""
    if variable in {"TAM", "RESP", "O2SAT", "TEMP", "GLASGOW", "DIURESIS", "SBP", "DBP", "HR"}:
        return COLORS["vitals"]
    if variable in {"creatinina", "bilirubina_total", "plaquetes", "leucocits", "lactat_arterial", "lactat_venos"}:
        return COLORS["lab"]
    return COLORS["coverage"]


def _format_int(value: int) -> str:
    """Format integers with a dot as thousands separator for local reports."""
    return f"{value:,}".replace(",", ".")


def _set_title(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Apply the shared title, subtitle, and axis style."""
    ax.set_title(title, fontsize=20, fontweight="bold", loc="left", pad=34)
    if subtitle:
        ax.text(0, 1.006, subtitle, transform=ax.transAxes, fontsize=12, color=COLORS["muted"], va="bottom")
    _style_axis(ax)


def _style_axis(ax: plt.Axes) -> None:
    """Apply the shared EDA axis style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["line"])
    ax.spines["bottom"].set_color(COLORS["line"])
    ax.tick_params(colors=COLORS["ink"], labelsize=11)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)


def _show_empty_axis(ax: plt.Axes, message: str) -> None:
    """Render a clean placeholder when a figure panel has no data."""
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["muted"],
    )


def _annotate_percent_bars_h(
    ax: plt.Axes,
    values: pd.Series,
    suffix: str = "%",
    offset: float = 0.8,
    fontsize: int = 11,
) -> None:
    """Annotate horizontal bars with percentage labels."""
    max_value = max(float(values.max()), 1.0)
    current_right = ax.get_xlim()[1]
    ax.set_xlim(ax.get_xlim()[0], max(current_right, min(105, max_value + 8)))
    for i, value in enumerate(values):
        ax.text(float(value) + offset, i, f"{float(value):.1f}{suffix}", va="center", fontsize=fontsize, color=COLORS["ink"])


def _palette_for_groups(groups: list[str]) -> dict[str, str]:
    """Return a stable palette for comparison groups."""
    if len(groups) == 2:
        return {groups[0]: COLORS["no_sepsis"], groups[1]: COLORS["sepsis"]}
    okabe_ito = [
        PALETTE["blue"],
        PALETTE["vermilion"],
        PALETTE["teal"],
        PALETTE["purple"],
        PALETTE["orange"],
        PALETTE["sky_blue"],
        PALETTE["neutral"],
        PALETTE["yellow"],
    ]
    return {group: okabe_ito[i % len(okabe_ito)] for i, group in enumerate(groups)}


def _cohort_flow_row(df: pd.DataFrame, etapa: str, mask: pd.Series) -> dict:
    """Build one cohort-count row from a boolean mask."""
    mask = mask.reindex(df.index, fill_value=False).fillna(False)
    subset = df.loc[mask]
    total = len(df)
    row = {
        "etapa": etapa,
        "rows": int(len(subset)),
        "pct_rows_sobre_total": round(100 * len(subset) / total, 2) if total else 0.0,
    }
    if "Episodi" in subset.columns:
        row["episodes"] = int(subset["Episodi"].nunique())
    if "Nhc" in subset.columns:
        row["patients"] = int(subset["Nhc"].nunique())
    return row


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a figure and release its memory."""
    save_report_figure(fig, path)


def _clear_pngs(directory: Path) -> None:
    """Remove stale generated PNGs from one figure directory."""
    for png_path in directory.glob("*.png"):
        png_path.unlink()


def _clear_eda_tables(directory: Path) -> None:
    """Remove stale generated EDA tables and text reports before rewriting them."""
    for pattern in ("eda_*.csv", "eda_*.txt"):
        for path in directory.glob(pattern):
            path.unlink()


def _save_csv_excel(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a CSV that opens cleanly in Excel with European separators."""
    output = _format_eda_table_for_output(df)
    output.to_csv(path, index=index, sep=";", decimal=",", encoding="utf-8-sig")


def _format_eda_table_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Keep source variable names in CSV exports and add readable labels."""
    output = df.copy()
    for column in ("variable", "subgroup_variable"):
        if column in output.columns:
            label_column = f"{column}_label"
            if label_column not in output.columns:
                insert_at = output.columns.get_loc(column) + 1
                output.insert(insert_at, label_column, output[column].map(_label_variable))
    output = output.rename(
        columns={
            "diagnostic_ingres": "admission_diagnosis",
        },
    )
    return output


def _serie_diagnostic_ingres(df: pd.DataFrame) -> pd.Series:
    """Normalize admission diagnosis values and remove blanks."""
    return df["diagnostic_ingres"].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna()


def _round_if_value(value):
    """Round numeric values when they exist."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 3)


def _resolve_output_directories(output_subfolder: str | None) -> tuple[Path, Path]:
    """Resolve output and figure directories, with an optional subfolder."""
    if not output_subfolder:
        return OUTPUTS_DIR, FIGURES_DIR
    outputs_dir = OUTPUTS_DIR / output_subfolder
    figures_dir = outputs_dir / "figures"
    return outputs_dir, figures_dir










