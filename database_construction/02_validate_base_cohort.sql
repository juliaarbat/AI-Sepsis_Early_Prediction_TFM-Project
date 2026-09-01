/* =========================================================
   BASIC VALIDATION OF base_hospitalization_cohort

   Purpose:
   - Check that the base cohort has one row per hospital episode.
   - Confirm that main inclusion/exclusion criteria were applied.
   - Review key data quality indicators before building daily feature tables.
   ========================================================= */


/* =========================================================
   1) Executive quality summary
   Compact overview of volume, uniqueness, date ranges, age range, missing key
   fields, exclusion leakage, and negative derived durations.
   ========================================================= */
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT `Episodi`) AS n_unique_episodes,
    COUNT(DISTINCT `Nhc`) AS n_unique_patients,

    COUNT(*) - COUNT(DISTINCT `Episodi`) AS n_duplicate_episode_rows,

    MIN(`DataIngres`) AS min_admission_date,
    MAX(`DataIngres`) AS max_admission_date,
    MIN(`DataIniciUrgencies`) AS min_emergency_start_date,
    MAX(`DataIniciUrgencies`) AS max_emergency_start_date,
    MIN(`DataAlta`) AS min_discharge_date,
    MAX(`DataAlta`) AS max_discharge_date,

    MIN(`Edat`) AS min_age,
    MAX(`Edat`) AS max_age,
    AVG(`Edat`) AS mean_age,

    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS n_missing_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS n_missing_patient_id,
    SUM(CASE WHEN `DataIngres` IS NULL THEN 1 ELSE 0 END) AS n_missing_admission_date,
    SUM(CASE WHEN `DataIniciUrgencies` IS NULL THEN 1 ELSE 0 END) AS n_missing_emergency_start_date,
    SUM(CASE WHEN `DataAlta` IS NULL THEN 1 ELSE 0 END) AS n_missing_discharge_date,
    SUM(CASE WHEN `Edat` IS NULL THEN 1 ELSE 0 END) AS n_missing_age,
    SUM(CASE WHEN `Sexe` IS NULL THEN 1 ELSE 0 END) AS n_missing_sex,

    SUM(CASE WHEN `Edat` < 18 THEN 1 ELSE 0 END) AS n_under_18,
    SUM(CASE WHEN `DataIngres` < '2018-01-01' OR `DataIngres` >= '2027-01-01' THEN 1 ELSE 0 END) AS n_outside_study_period,
    SUM(CASE WHEN `DataIniciUrgencies` > `DataIngres` THEN 1 ELSE 0 END) AS n_emergency_start_after_admission,
    SUM(CASE WHEN `DataAlta` IS NOT NULL AND `DataAlta` < `DataIngres` THEN 1 ELSE 0 END) AS n_discharge_before_admission,

    SUM(CASE WHEN `Temps_critics` < 0 THEN 1 ELSE 0 END) AS n_negative_critical_care_hours,
    SUM(CASE WHEN `Temps_cirurgia` < 0 THEN 1 ELSE 0 END) AS n_negative_surgery_hours,

    SUM(
        CASE
            WHEN `Diagnostic_ingres` IS NOT NULL
             AND (
                    UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A40%'
                 OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A41%'
                 OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'P36%'
                 OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) IN ('R6520', 'R6521', 'A021', 'B377', 'O85', 'T814')
                 )
            THEN 1 ELSE 0
        END
    ) AS n_initial_sepsis_diagnosis_still_present
FROM base_hospitalization_cohort;


/* =========================================================
   2) Duplicate episode details
   Ideally this query should return no rows. Any result means the cohort is no
   longer one row per hospital episode.
   ========================================================= */
SELECT
    `Episodi`,
    COUNT(*) AS n_rows
FROM base_hospitalization_cohort
GROUP BY `Episodi`
HAVING COUNT(*) > 1
ORDER BY n_rows DESC, `Episodi`
LIMIT 100;


/* =========================================================
   3) Records with critical temporal or inclusion problems
   Shows concrete rows when the executive summary reports invalid age, dates,
   negative durations, or an admission diagnosis compatible with sepsis.
   ========================================================= */
SELECT
    `Episodi`,
    `Nhc`,
    `DataIngres`,
    `DataIniciUrgencies`,
    `DataAlta`,
    `Edat`,
    `Diagnostic_ingres`,
    `Temps_critics`,
    `Temps_cirurgia`
FROM base_hospitalization_cohort
WHERE `Edat` IS NULL
   OR `Edat` < 18
   OR `DataIngres` < '2018-01-01'
   OR `DataIngres` >= '2027-01-01'
   OR `DataIniciUrgencies` > `DataIngres`
   OR (`DataAlta` IS NOT NULL AND `DataAlta` < `DataIngres`)
   OR `Temps_critics` < 0
   OR `Temps_cirurgia` < 0
   OR (
        `Diagnostic_ingres` IS NOT NULL
    AND (
           UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A40%'
        OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A41%'
        OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'P36%'
        OR UPPER(REPLACE(REPLACE(TRIM(`Diagnostic_ingres`), '.', ''), ' ', '')) IN ('R6520', 'R6521', 'A021', 'B377', 'O85', 'T814')
        )
      )
ORDER BY `DataIngres`, `Episodi`
LIMIT 100;


/* =========================================================
   4) Admission year distribution
   Used to verify that the 2018-2026 study window is populated as expected and
   that no year has an unexpected drop.
   ========================================================= */
SELECT
    YEAR(`DataIngres`) AS admission_year,
    COUNT(*) AS n_episodes
FROM base_hospitalization_cohort
GROUP BY YEAR(`DataIngres`)
ORDER BY admission_year;


/* =========================================================
   5) Basic categorical distributions
   Quick checks for demographic and admission-source fields.
   ========================================================= */
SELECT
    'Sexe' AS variable_name,
    CAST(`Sexe` AS CHAR) AS category_value,
    COUNT(*) AS n_rows
FROM base_hospitalization_cohort
GROUP BY `Sexe`

UNION ALL

SELECT
    'Font_admissio' AS variable_name,
    CAST(`Font_admissio` AS CHAR) AS category_value,
    COUNT(*) AS n_rows
FROM base_hospitalization_cohort
GROUP BY `Font_admissio`

UNION ALL

SELECT
    'Passa_per_critics' AS variable_name,
    CAST(`Passa_per_critics` AS CHAR) AS category_value,
    COUNT(*) AS n_rows
FROM base_hospitalization_cohort
GROUP BY `Passa_per_critics`

UNION ALL

SELECT
    'Cirurgia' AS variable_name,
    CAST(`Cirurgia` AS CHAR) AS category_value,
    COUNT(*) AS n_rows
FROM base_hospitalization_cohort
GROUP BY `Cirurgia`
ORDER BY variable_name, category_value;


/* =========================================================
   6) Length of stay and derived duration summaries
   Reviews hospital stay, critical care exposure, and surgery duration. Very
   short stays are informative only; they are not excluded in this table.
   ========================================================= */
SELECT
    COUNT(CASE WHEN `DataAlta` IS NOT NULL THEN 1 END) AS n_with_discharge_date,
    SUM(CASE WHEN `DataAlta` IS NOT NULL AND TIMESTAMPDIFF(HOUR, `DataIngres`, `DataAlta`) <= 24 THEN 1 ELSE 0 END) AS n_stay_24h_or_less,

    MIN(TIMESTAMPDIFF(HOUR, `DataIngres`, `DataAlta`)) AS min_stay_hours,
    MAX(TIMESTAMPDIFF(HOUR, `DataIngres`, `DataAlta`)) AS max_stay_hours,
    AVG(TIMESTAMPDIFF(HOUR, `DataIngres`, `DataAlta`)) AS mean_stay_hours,

    MIN(`Temps_critics`) AS min_critical_care_hours,
    MAX(`Temps_critics`) AS max_critical_care_hours,
    AVG(`Temps_critics`) AS mean_critical_care_hours,

    MIN(`Temps_cirurgia`) AS min_surgery_hours,
    MAX(`Temps_cirurgia`) AS max_surgery_hours,
    AVG(`Temps_cirurgia`) AS mean_surgery_hours,
    SUM(CASE WHEN `Urgencia_cirurgia` = 1 THEN 1 ELSE 0 END) AS n_urgent_surgery
FROM base_hospitalization_cohort;


/* =========================================================
   7) Comorbidity prevalence
   Counts episodes with each comorbidity flag. A value of zero may be valid,
   but should be reviewed for rare conditions or newly added definitions.
   ========================================================= */
SELECT
    SUM(`COMORB_DIABETES_MELLITUS`)      AS n_diabetes_mellitus,
    SUM(`COMORB_NEOPLASIA_SOLIDA`)       AS n_solid_tumor,
    SUM(`COMORB_NEOPLASIA_HEMATOLOGICA`) AS n_hematological_malignancy,
    SUM(`COMORB_ENOLISME_SEVER`)         AS n_severe_alcohol_use_disorder,
    SUM(`COMORB_CIRROSI_HEPATICA`)       AS n_liver_cirrhosis,
    SUM(`COMORB_VIH_SIDA`)               AS n_hiv_aids,
    SUM(`COMORB_TRASPLANT_ORGAN_SOLID`)  AS n_solid_organ_transplant,
    SUM(`COMORB_TRASPLANT_MOLL_OS`)      AS n_bone_marrow_transplant,
    SUM(`COMORB_AGAMMAGLOBULINEMIA`)     AS n_agammaglobulinemia,
    SUM(`COMORB_HIPOGAMMAGLOBULINEMIA`)  AS n_hypogammaglobulinemia,
    SUM(`COMORB_MALABSORTIVES`)          AS n_malabsorption_syndromes,
    SUM(`COMORB_MALNUTRICIO_SEVERA`)     AS n_severe_malnutrition,
    SUM(`COMORB_ASPLENIA`)               AS n_asplenia,
    SUM(`COMORB_ESPLENECTOMIA`)          AS n_splenectomy,
    SUM(`COMORB_IRC_DIALISI`)            AS n_chronic_kidney_disease_on_dialysis,
    SUM(`COMORB_NEUTROPENIA_GREU`)       AS n_severe_neutropenia,
    SUM(
        CASE
            WHEN COALESCE(`COMORB_DIABETES_MELLITUS`, 0) = 0
             AND COALESCE(`COMORB_NEOPLASIA_SOLIDA`, 0) = 0
             AND COALESCE(`COMORB_NEOPLASIA_HEMATOLOGICA`, 0) = 0
             AND COALESCE(`COMORB_ENOLISME_SEVER`, 0) = 0
             AND COALESCE(`COMORB_CIRROSI_HEPATICA`, 0) = 0
             AND COALESCE(`COMORB_VIH_SIDA`, 0) = 0
             AND COALESCE(`COMORB_TRASPLANT_ORGAN_SOLID`, 0) = 0
             AND COALESCE(`COMORB_TRASPLANT_MOLL_OS`, 0) = 0
             AND COALESCE(`COMORB_AGAMMAGLOBULINEMIA`, 0) = 0
             AND COALESCE(`COMORB_HIPOGAMMAGLOBULINEMIA`, 0) = 0
             AND COALESCE(`COMORB_MALABSORTIVES`, 0) = 0
             AND COALESCE(`COMORB_MALNUTRICIO_SEVERA`, 0) = 0
             AND COALESCE(`COMORB_ASPLENIA`, 0) = 0
             AND COALESCE(`COMORB_ESPLENECTOMIA`, 0) = 0
             AND COALESCE(`COMORB_IRC_DIALISI`, 0) = 0
             AND COALESCE(`COMORB_NEUTROPENIA_GREU`, 0) = 0
            THEN 1 ELSE 0
        END
    ) AS n_without_defined_comorbidities
FROM base_hospitalization_cohort;


/* =========================================================
   8) Binary flag validation
   All derived binary variables should contain only 0 or 1 and should not be
   NULL after the final COALESCE step in the cohort-building script.
   ========================================================= */
SELECT
    SUM(CASE WHEN `Passa_per_critics` NOT IN (0, 1) OR `Passa_per_critics` IS NULL THEN 1 ELSE 0 END) AS invalid_passa_per_critics,
    SUM(CASE WHEN `Cirurgia` NOT IN (0, 1) OR `Cirurgia` IS NULL THEN 1 ELSE 0 END) AS invalid_cirurgia,
    SUM(CASE WHEN `Urgencia_cirurgia` NOT IN (0, 1) OR `Urgencia_cirurgia` IS NULL THEN 1 ELSE 0 END) AS invalid_urgencia_cirurgia,
    SUM(CASE WHEN `Temps_cirurgia_disponible` NOT IN (0, 1) OR `Temps_cirurgia_disponible` IS NULL THEN 1 ELSE 0 END) AS invalid_temps_cirurgia_disponible,

    SUM(CASE WHEN `COMORB_DIABETES_MELLITUS` NOT IN (0, 1) OR `COMORB_DIABETES_MELLITUS` IS NULL THEN 1 ELSE 0 END) AS invalid_diabetes_mellitus,
    SUM(CASE WHEN `COMORB_NEOPLASIA_SOLIDA` NOT IN (0, 1) OR `COMORB_NEOPLASIA_SOLIDA` IS NULL THEN 1 ELSE 0 END) AS invalid_solid_tumor,
    SUM(CASE WHEN `COMORB_NEOPLASIA_HEMATOLOGICA` NOT IN (0, 1) OR `COMORB_NEOPLASIA_HEMATOLOGICA` IS NULL THEN 1 ELSE 0 END) AS invalid_hematological_malignancy,
    SUM(CASE WHEN `COMORB_ENOLISME_SEVER` NOT IN (0, 1) OR `COMORB_ENOLISME_SEVER` IS NULL THEN 1 ELSE 0 END) AS invalid_severe_alcohol_use_disorder,
    SUM(CASE WHEN `COMORB_CIRROSI_HEPATICA` NOT IN (0, 1) OR `COMORB_CIRROSI_HEPATICA` IS NULL THEN 1 ELSE 0 END) AS invalid_liver_cirrhosis,
    SUM(CASE WHEN `COMORB_VIH_SIDA` NOT IN (0, 1) OR `COMORB_VIH_SIDA` IS NULL THEN 1 ELSE 0 END) AS invalid_hiv_aids,
    SUM(CASE WHEN `COMORB_TRASPLANT_ORGAN_SOLID` NOT IN (0, 1) OR `COMORB_TRASPLANT_ORGAN_SOLID` IS NULL THEN 1 ELSE 0 END) AS invalid_solid_organ_transplant,
    SUM(CASE WHEN `COMORB_TRASPLANT_MOLL_OS` NOT IN (0, 1) OR `COMORB_TRASPLANT_MOLL_OS` IS NULL THEN 1 ELSE 0 END) AS invalid_bone_marrow_transplant,
    SUM(CASE WHEN `COMORB_AGAMMAGLOBULINEMIA` NOT IN (0, 1) OR `COMORB_AGAMMAGLOBULINEMIA` IS NULL THEN 1 ELSE 0 END) AS invalid_agammaglobulinemia,
    SUM(CASE WHEN `COMORB_HIPOGAMMAGLOBULINEMIA` NOT IN (0, 1) OR `COMORB_HIPOGAMMAGLOBULINEMIA` IS NULL THEN 1 ELSE 0 END) AS invalid_hypogammaglobulinemia,
    SUM(CASE WHEN `COMORB_MALABSORTIVES` NOT IN (0, 1) OR `COMORB_MALABSORTIVES` IS NULL THEN 1 ELSE 0 END) AS invalid_malabsorption_syndromes,
    SUM(CASE WHEN `COMORB_MALNUTRICIO_SEVERA` NOT IN (0, 1) OR `COMORB_MALNUTRICIO_SEVERA` IS NULL THEN 1 ELSE 0 END) AS invalid_severe_malnutrition,
    SUM(CASE WHEN `COMORB_ASPLENIA` NOT IN (0, 1) OR `COMORB_ASPLENIA` IS NULL THEN 1 ELSE 0 END) AS invalid_asplenia,
    SUM(CASE WHEN `COMORB_ESPLENECTOMIA` NOT IN (0, 1) OR `COMORB_ESPLENECTOMIA` IS NULL THEN 1 ELSE 0 END) AS invalid_splenectomy,
    SUM(CASE WHEN `COMORB_IRC_DIALISI` NOT IN (0, 1) OR `COMORB_IRC_DIALISI` IS NULL THEN 1 ELSE 0 END) AS invalid_chronic_kidney_disease_on_dialysis,
    SUM(CASE WHEN `COMORB_NEUTROPENIA_GREU` NOT IN (0, 1) OR `COMORB_NEUTROPENIA_GREU` IS NULL THEN 1 ELSE 0 END) AS invalid_severe_neutropenia
FROM base_hospitalization_cohort;


/* =========================================================
   9) Manual inspection sample
   Small sample for visual review of the most relevant columns.
   ========================================================= */
SELECT
    `Episodi`,
    `Nhc`,
    `DataIngres`,
    `DataIniciUrgencies`,
    `DataAlta`,
    `Edat`,
    `Sexe`,
    `Font_admissio`,
    `Diagnostic_ingres`,
    `Passa_per_critics`,
    `Temps_critics`,
    `Cirurgia`,
    `Urgencia_cirurgia`,
    `Temps_cirurgia`
FROM base_hospitalization_cohort
ORDER BY `DataIngres`, `Episodi`
LIMIT 20;


/* =========================================================
   10) Table structure and indexes
   Confirms that expected columns and final indexes exist.
   ========================================================= */
DESCRIBE base_hospitalization_cohort;

SHOW INDEX FROM base_hospitalization_cohort;
