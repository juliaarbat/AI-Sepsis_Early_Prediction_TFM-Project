# TFM Sepsis Early Prediction - Python Analysis

This directory contains the essential Python pipeline for the TFM:

- General dataset EDA.
- SOFA cleaning and calculation.
- Training of classic machine-learning models.
- Optuna only on the best already-compared classic model, with post-Optuna SHAP.
- Training of temporal Transformer, LSTM, and RNN models under the three real-cohort policies.
- Optuna only on the best already-compared deep-learning model, with post-Optuna SHAP.
- Individual and comparative figures for classic and deep-learning models.
- PCA/LASSO analysis to review variables that may behave like noise.
- Previous-day-only ablation for the selected final LightGBM and Transformer
  candidates, plus comparison figures.

Previously generated outputs are kept in `outputs/`. Python caches, training logs,
and temporary folders are ignored through `.gitignore`.
Word/Docx report-construction scripts are not kept here.

## Structure

- `scripts/01_eda.py`: general EDA and main figures.
- `scripts/02_compute_sofa.py`: pre-SOFA cleaning, SOFA calculation, and next-day sepsis labels.
- `scripts/03_models_classics.py`: classic-model comparison with a short parameter grid.
- `scripts/04_models_classics_optuna.py`: Optuna only on the best classic model from previous results, plus SHAP for the optimized model.
- `scripts/05_models_classics_figures.py`: individual and comparative figures for already trained classic models.
- `scripts/06_deep_learning.py`: temporal Transformer, LSTM, and RNN models for the three real-cohort policies.
- `scripts/07_deep_learning_figures.py`: individual and comparative Transformer, LSTM, and RNN figures without retraining.
- `scripts/08_deep_learning_optuna.py`: Optuna only on the best sequential model for `real_all_2026`, with post-Optuna SHAP inside the selected run.
- `scripts/09_post_optuna_final_figures.py`: final tables and figures comparing baseline, Optuna, test, real, D+1/row, episode level, and robustness CV when available.
- `scripts/10_pca_lasso_noise.py`: PCA/LASSO analysis of variables that may behave like noise.
- `scripts/11_previous_day_only_ablation.py`: previous-day-only ablation for the selected final LightGBM and Transformer candidates.
- `scripts/12_previous_day_only_figures.py`: tables, SHAP outputs, and figures comparing full-history and previous-day-only models.

- `scripts/_bootstrap.py`: adjusts the path so scripts can run from outside `Python_analisi`.
- `src/config.py`: project paths.
- `src/data_loading.py`: original CSV loading and SOFA caches.
- `src/feature_utils.py`: shared feature-name, policy-label, and feature-importance helpers.
- `src/general_eda.py`: reusable general EDA.
- `src/sofa_cleaning.py`: dataset preparation before SOFA.
- `src/sofa_calculation.py`: SOFA components, operational baseline, delta, and labels.
- `src/predictive_model_24h.py`: data preparation, splits, and common metrics.
- `src/output_contracts.py`: shared output filenames and metric-column constants.
- `src/output_paths.py`: shared output-path resolution helpers.
- `src/output_schema.py`: shared legacy-to-English output column mapping.
- `src/plot_utils.py`: shared plotting helpers for report figures.
- `src/split_utils.py`: shared split-unit normalization, split validation, and split audit metadata.
- `src/classic_models_24h.py`: preparation, training, validation, and figures for classic models.
- `src/temporal_model_24h.py`: sequential dataset and temporal Transformer, LSTM, and RNN training.
- `src/training_comparisons.py`: immediate comparison between policies after training classic models.
- `src/classic_model_comparisons.py`: comparative tables and figures for already trained classic models.
- `src/deep_learning_comparisons.py`: comparative tables and figures for Transformer, LSTM, and RNN.
- `src/post_optuna_final_reporting.py`: final baseline-vs-Optuna tables and figures.
- `src/deep_learning_figures.py`: individual temporal-model figures without retraining.
- `src/deep_learning_shap_24h.py`: internal SHAP calculation for the best post-Optuna deep-learning model.
- `src/previous_day_only_ablation_figures.py`: comparison tables, SHAP backfill, and figures for the previous-day-only ablation.
- `src/progress.py`: shared progress messages.
- `src/reporting.py`: shared console output.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

1. General EDA:

```bash
python scripts/01_eda.py
```

This step loads the daily CSV, prints a control summary, and generates:

- summary report `outputs/eda_general/eda_summary_report.txt`;
- missingness, numeric-variable, binary-variable, block-coverage, diagnosis,
  SOFA-missingness, clinical-correlation, clinical-range-outlier, and temporal
  activity tables;
- supplementary-material figures for dataset overview, temporal activity,
  missingness, data availability, demographics, admission characteristics,
  comorbidities, recent healthcare exposure, treatment/microbiology indicators,
  and key vital-sign and laboratory distributions.

The outputs from this step are grouped under `outputs/eda_general/`, with figures
inside `outputs/eda_general/figures/`.

EDA CSV files are saved in an Excel-compatible European format: `;` separator,
`UTF-8-SIG` encoding, and comma decimals.

The general EDA deliberately no longer writes internal audit or redundant outputs
that were not needed for the supplementary material:

- `eda_leakage_variable_audit.csv`;
- `eda_cohort_flow.csv`;
- `eda_date_audit.csv`;
- `eda_missingness_temporal.csv`;
- standalone `cohort_flow.png`.

`cohort_flow` is still calculated internally for the dataset overview figure, but
it is not exported as a separate CSV or figure.

Visible figure titles are publication-facing and do not include internal labels
such as `(eda-general)`. File names still keep stable technical names so scripts
and report references remain easy to track.

Distribution figures for numeric variables only show variables with at least
20% coverage and 1,000 observations. Low-coverage variables remain documented in
the missingness and coverage tables, but are excluded from the main histograms
and boxplots to avoid poorly interpretable charts.

The visual criteria follow the provided HDVC notes: avoid pie, donut, and circular
plots; prioritize barplots and lollipop or linear readings; use color deliberately;
apply color-blind-friendly palettes such as Okabe-Ito; mark exceptions with accent
colors; and reduce visual redundancy.

When the input dataset already includes SOFA or sepsis labels, this same EDA code
also creates conditional post-label summaries and figures: SOFA distribution,
SOFA-component contribution, sepsis prevalence, next-day sepsis prevalence,
episode-level prevalence, standardized differences, label-specific coverage,
subgroup comparisons, and pre-sepsis clinical trajectories where available.

2. SOFA cleaning and calculation:

```bash
python scripts/02_compute_sofa.py
```

This step generates grouped outputs under `outputs/sofa/`:

- `outputs/sofa/datasets/daily_sepsis_model_clean_sofa.csv`;
- `outputs/sofa/datasets/daily_sepsis_model_with_sofa.csv`;
- pre-SOFA audits under `outputs/sofa/pre_sofa/`;
- summaries and criteria under `outputs/sofa/reports/`;
- post-SOFA EDA under `outputs/sofa/eda_post_sofa/`.

The pre-SOFA phase is conservative and auditable: it does not remove predictive
variables or clinically plausible outliers. Before calculating SOFA, it only:

- converts dates and removes rows without a valid `data_index` or with a future
  `data_index` relative to the project's reproducible cutoff date;
- removes exact duplicates;
- converts physiologically impossible values to `NaN` using broad ranges;
- normalizes inconsistent pre-SOFA binary flags;
- excludes episodes with too many missing SOFA components;
- audits very large temporal gaps without excluding them by default.

For daily SOFA, missing laboratory values can be carried forward for up to
14 days within the same episode, and vital signs for up to 3 days. These limits
are applied consistently in `scripts/02_compute_sofa.py` and in the model scripts
that load the SOFA dataset. The shared configuration names are
`SOFA_LAB_FFILL_LIMIT_DAYS` and `SOFA_VITALS_FFILL_LIMIT_DAYS`.

The audits are saved to:

- `outputs/sofa/pre_sofa/pre_sofa_auditoria_resum.csv`;
- `outputs/sofa/pre_sofa/pre_sofa_auditoria_dates.csv`;
- `outputs/sofa/pre_sofa/pre_sofa_auditoria_variables.csv`;
- `outputs/sofa/pre_sofa/pre_sofa_auditoria_binaries.csv`;
- `outputs/sofa/pre_sofa/episodis_exclosos_sofa.csv`;
- `outputs/sofa/pre_sofa/resum_neteja_sofa.json`.

The SOFA calculation also generates short audits in:

- `outputs/sofa/reports/auditoria_sofa_variables.csv`, with imputations,
  carry-forwards, and variables without a recent value;
- `outputs/sofa/reports/auditoria_sofa_components.csv`, with available
  components, baseline, labels, and model eligibility.

The post-SOFA EDA extends the base EDA with temporal prevalence of labels,
comparisons by `sepsis` and `sepsis_dia_seguent`, differential coverage by label,
clinical/administrative subgroups, and trajectories of key variables around the
first sepsis day.

Cleaning decisions happen at three points:

- before SOFA: conservative quality cleaning and exclusion of non-computable episodes;
- after SOFA: labels, coverage, and candidate variables are reviewed through the post-SOFA EDA;
- inside the models: variables are removed using train-derived criteria to avoid leakage.

3. Classic models:

```bash
python scripts/03_models_classics.py
```

This step creates four chronological and mutually exclusive groups by `Nhc` by
default: `train`, `valid`, `test`, and `real`. The `real` sample reserves complete
patients with activity from `2026-01-01` onward under three policies: readmitted
patients with previous history, new patients, and all patients with real-period
activity. The `real` set does not participate in preprocessing, variable selection,
tuning, threshold selection, or cross-validation.

Here, chronological means that split units are ordered by their first observed
date. When the split unit is `Nhc`, each patient remains entirely in one
partition, even if some of that patient's records would conceptually fall on
different sides of a row-level date boundary. This prioritizes patient-level
separation over strict row-level temporal cutoffs.

The code uses `patient` and `episode` as the canonical split-unit names. Legacy
aliases such as `pacient` and `episodi` are still accepted for compatibility.

Cross-validation uses 5 patient-grouped folds inside the development set. It is a
robustness analysis and does not replace either the temporal test set or the real
set. The pipeline:

- excludes SOFA availability proxies to reduce leakage;
- keeps microbiology variables available in the dataset;
- creates simple historical variables using information available up to the current day;
- tries a short grid per model and selects by validation AUPRC;
- computes patient-grouped cross-validation to estimate stability;
- evaluates the holdout test set, the real sample, and between-fold robustness;
- finally evaluates the real sample with the model and thresholds already fixed;
- computes AUPRC lift (`AUPRC / prevalence`) to interpret AUPRC under different
  prevalences, both in the results CSV and in comparative figures;
- saves the tables needed to generate figures without retraining models.

Outputs are grouped under `outputs/models_classics_24h/`:

- `outputs/models_classics_24h/real_readmitted_2026/`: results,
  predictions, tuning, cross-validation, importances, and figures for patients
  with previous history and real-period activity.
- `outputs/models_classics_24h/real_new_2026/`: the same outputs for patients
  that start directly in the real period.
- `outputs/models_classics_24h/real_all_2026/`: the same outputs for all
  patients with real-period activity.
- `outputs/models_classics_24h/comparison/`: comparative tables and figures
  between policies.
- `outputs/models_classics_24h/README.md`: folder-organization summary.

4. Optuna only on the best classic model:

```bash
python scripts/04_models_classics_optuna.py
```

This step reads previous results from `scripts/03_models_classics.py`, selects
the best model by validation AUPRC inside each real-cohort strategy, and runs
Optuna only for that model. Outputs are saved to
`outputs/models_classics_24h/optuna_best/`. By default it runs 30 trials, which
is enough for LightGBM given that the best values are tightly grouped. After the
run, it computes a 5-fold patient-grouped robustness CV only for the optimized
final configuration, and saves D+1/row and episode-level metrics in
`classic_models_24h_cv_folds.csv` and `classic_models_24h_cv_summary.csv`.
It also computes SHAP on a sample from the real split of the optimized model and
saves the table, figure, and summary inside the `shap/` subfolder of each Optuna
run. The `optuna_best/` folder includes a `README.md` and a
`classic_models_24h_optuna_best_summary.json` with the main paths. The Optuna
criterion is validation AUPRC at D+1/row level; the tuning CSV also saves
episode-level AUROC/AUPRC to compare both levels.

5. Classic-model figures:

```bash
python scripts/05_models_classics_figures.py
```

This step does not retrain models. It reads the results from the three policies
`real_readmitted_2026`, `real_new_2026`, and `real_all_2026`, regenerates
individual figures inside each policy folder, and creates curated comparative
figures under `outputs/models_classics_24h/comparison/`.

6. Temporal deep-learning models:

```bash
python scripts/06_deep_learning.py
```

This step trains Transformer, LSTM, and RNN models under the same three policies
as the classic models:

- `real_readmitted_2026`: patients with previous history and real-period activity.
- `real_new_2026`: patients that start directly in the real period.
- `real_all_2026`: all patients with real-period activity.

The real-cohort cutoff date is `2026-01-01`. Outputs are grouped under:

- `outputs/deep_learning_24h/transformer/real_readmitted_2026/`
- `outputs/deep_learning_24h/transformer/real_new_2026/`
- `outputs/deep_learning_24h/transformer/real_all_2026/`
- `outputs/deep_learning_24h/lstm/real_readmitted_2026/`
- `outputs/deep_learning_24h/lstm/real_new_2026/`
- `outputs/deep_learning_24h/lstm/real_all_2026/`
- `outputs/deep_learning_24h/rnn/real_readmitted_2026/`
- `outputs/deep_learning_24h/rnn/real_new_2026/`
- `outputs/deep_learning_24h/rnn/real_all_2026/`

Each folder contains a JSON summary, predictions, comparable metrics, variables
excluded due to missingness, the `.pt` model, and figures inside `figures/`. The
`*_metriques_comparables.csv` CSV includes `auprc_lift` to interpret AUPRC under
different prevalences.

7. Deep-learning figures:

```bash
python scripts/07_deep_learning_figures.py
```

This step does not retrain models. It reads already saved summaries and
predictions from Transformer, LSTM, and RNN, regenerates figures inside each
`figures/` folder, and also generates curated comparative figures under
`outputs/deep_learning_24h/comparison/`.

Figures can also be generated by model or policy:

```bash
python scripts/07_deep_learning_figures.py --model transformer
python scripts/07_deep_learning_figures.py --model lstm --policy real_readmitted_2026
python scripts/07_deep_learning_figures.py --model rnn --policy real_new_2026
```

8. Optuna only on the best deep-learning model:

```bash
python scripts/08_deep_learning_optuna.py
```

This step reads the comparable metrics from `scripts/06_deep_learning.py`,
selects the best model by validation AUPRC for `real_all_2026`, and runs or
reuses the corresponding Optuna execution according to `RUN_NEW_OPTUNA_TRIALS`
inside `scripts/08_deep_learning_optuna.py`. New outputs are saved to
`outputs/deep_learning_24h/optuna_best/<policy>/<model>/`, matching the classic
model structure; older executions in the `<model>/<policy>` order can also be
reused. At the end, it computes SHAP for the optimized selected model and
saves a table, importance file, positive/negative directional figure, and summary
inside the `shap/` subfolder of each Optuna run. The `optuna_best/` folder
includes a `README.md` and a `deep_learning_24h_optuna_best_resum.json` with the
main paths. The Optuna criterion is validation AUPRC at D+1/row level; the tuning
CSV also saves episode-level AUROC/AUPRC.

Legacy policy names such as `real_tots_2026`, `real_nous_2026`,
`real_reingressats_2026`, `pacient_al_real`, and
`pacient_al_desenvolupament` are still accepted when reading existing outputs,
but new code should use `real_readmitted_2026`, `real_new_2026`, and
`real_all_2026`.
Comparison scripts keep their legacy CSV schemas and also write `_english.csv`
companions with translated public column names.

9. Final post-Optuna figures:

```bash
python scripts/09_post_optuna_final_figures.py
```

This step does not retrain models. It reads baseline and post-Optuna results from
classic and deep-learning models, accepts both the new and old deep-learning
Optuna folder structures, and generates a final folder:
`outputs/post_optuna_final/`. It includes a comparison CSV, a model-decision
table, and figures for baseline vs Optuna, test vs real, D+1/row vs episode
level, and post-Optuna robustness CV when available.
The main outputs have English names:
`tables/01_final_model_comparison.csv`, `tables/02_final_model_decision.csv`,
`tables/03_post_optuna_robustness_cv.csv`, and figure folders such as
`figures/01_base_optuna_comparison/`, `figures/03_row_episode_level/`, and
`figures/05_robustness_cv/`.

10. PCA/LASSO analysis of variables that may behave like noise:

```bash
python scripts/10_pca_lasso_noise.py
```

This step does not retrain or modify the final model. It is an exploratory check
of variables with weak signal: it applies PCA to the preprocessed features, fits
a logistic LASSO regression, and combines both readings in
`outputs/pca_lasso_noise/variable_noise_summary.csv`. Flagged variables are
candidates for review, not automatic removals.

11. Previous-day-only ablation:

```bash
python scripts/11_previous_day_only_ablation.py
```

This step tests whether the selected final candidates still perform well when
restricted to the immediately preceding patient-day. It keeps the same
SOFA-labelled dataset, patient-level split, `real_all_2026` policy, and real
cutoff date as the main final comparison, but removes longer temporal-history
information:

- LightGBM is rerun with engineered temporal-window features disabled;
- Transformer is rerun with `lookback_days = 1` and no lookback tuning.

Outputs are saved under `outputs/previous_day_only_ablation/`:

- `classic_lightgbm_optuna/`: LightGBM previous-day-only results, predictions,
  tuning outputs, split audits, saved model, and SHAP outputs when generated;
- `deep_transformer_optuna/`: Transformer previous-day-only summary, comparable
  metrics, predictions, tuning outputs, model file, and SHAP outputs when generated;
- `previous_day_only_ablation_index.json`: compact index consumed by the figure
  script.

By default, this script runs new Optuna trials according to
`RUN_NEW_OPTUNA_TRIALS` in `scripts/11_previous_day_only_ablation.py`. Set it to
`False` only when reusing an existing ablation run.

12. Previous-day-only ablation figures:

```bash
python scripts/12_previous_day_only_figures.py
```

This step does not retrain the ablation models. It reads the full-history
Optuna outputs and the previous-day-only ablation outputs, creates missing SHAP
interpretability files when needed, and generates the final comparison material.
Outputs are saved under `outputs/previous_day_only_ablation/`:

- `tables/01_previous_day_only_comparison.csv`;
- `tables/02_previous_day_only_feature_importance_comparison.csv`;
- comparative figures such as full-history vs previous-day-only AUPRC,
  previous-day-only metrics, and SHAP feature-importance comparisons;
- refreshed individual diagnostic figures inside the LightGBM and Transformer
  ablation folders;
- `previous_day_only_figures_index.json` with the generated table and figure paths.
