/* ============================================================
   FINAL VALIDATION OF daily_sepsis_model

   Purpose:
   - Validate the final daily modelling table before SOFA calculation,
     imputation, EDA, and downstream modelling.
   - Check row uniqueness, cohort flow, temporal consistency, source coverage,
     binary flags, plausible clinical ranges, and SOFA readiness.

   Note:
   - This script does not create tables. It only reads daily_sepsis_model and
     the upstream tables used to build it.
   ============================================================ */


/* ============================================================
   1) General structure
   Expected: duplicate_episode_patient_days = 0.
   ============================================================ */
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT `Episodi`) AS total_episodes,
    COUNT(DISTINCT `Nhc`) AS total_patients,
    COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`)) AS total_episode_patient_days,
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`)) AS duplicate_episode_patient_days
FROM daily_sepsis_model;


/* ============================================================
   2) Duplicate model rows
   Expected: no rows.
   ============================================================ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    COUNT(*) AS n_rows
FROM daily_sepsis_model
GROUP BY
    `Episodi`,
    `Nhc`,
    `data_index`
HAVING COUNT(*) > 1
ORDER BY n_rows DESC, `Episodi`, `data_index`
LIMIT 100;


/* ============================================================
   3) Cohort flow from base cohort to final model
   ============================================================ */
WITH
base_cohort AS (
    SELECT DISTINCT
        `Episodi`,
        `Nhc`
    FROM base_hospitalization_cohort
    WHERE `Episodi` IS NOT NULL
      AND `Nhc` IS NOT NULL
),
episodes_with_24h_vital_data AS (
    SELECT DISTINCT
        b.`Episodi`,
        b.`Nhc`
    FROM base_hospitalization_cohort b
    INNER JOIN clean_vital_signs_events v
        ON b.`Episodi` = v.`Episodi`
       AND b.`Nhc` = v.`Nhc`
    WHERE b.`DataIngres` IS NOT NULL
      AND v.`event_time` IS NOT NULL
      AND TIMESTAMPDIFF(HOUR, b.`DataIngres`, v.`event_time`) >= 24
),
episodes_with_creatinine_or_platelets AS (
    SELECT DISTINCT
        l.`Episodi`,
        l.`Nhc`
    FROM daily_laboratory_records l
    INNER JOIN episodes_with_24h_vital_data e24
        ON l.`Episodi` = e24.`Episodi`
       AND l.`Nhc` = e24.`Nhc`
    WHERE l.`creatinina` IS NOT NULL
       OR l.`plaquetes` IS NOT NULL
),
flow AS (
    SELECT
        '01_base_hospitalization_cohort' AS step_name,
        `Episodi`,
        `Nhc`,
        CAST(NULL AS DATE) AS `data_index`
    FROM base_cohort

    UNION ALL

    SELECT
        '02_with_vital_sign_days' AS step_name,
        v.`Episodi`,
        v.`Nhc`,
        v.`data_index`
    FROM daily_vital_signs v
    INNER JOIN base_cohort b
        ON v.`Episodi` = b.`Episodi`
       AND v.`Nhc` = b.`Nhc`

    UNION ALL

    SELECT
        '03_with_vital_data_after_24h' AS step_name,
        v.`Episodi`,
        v.`Nhc`,
        v.`data_index`
    FROM daily_vital_signs v
    INNER JOIN episodes_with_24h_vital_data e24
        ON v.`Episodi` = e24.`Episodi`
       AND v.`Nhc` = e24.`Nhc`

    UNION ALL

    SELECT
        '04_with_creatinine_or_platelets' AS step_name,
        l.`Episodi`,
        l.`Nhc`,
        l.`data_index`
    FROM daily_laboratory_records l
    INNER JOIN episodes_with_creatinine_or_platelets lab
        ON l.`Episodi` = lab.`Episodi`
       AND l.`Nhc` = lab.`Nhc`

    UNION ALL

    SELECT
        '05_final_daily_sepsis_model' AS step_name,
        m.`Episodi`,
        m.`Nhc`,
        m.`data_index`
    FROM daily_sepsis_model m
)
SELECT
    step_name,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients,
    COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', COALESCE(CAST(`data_index` AS CHAR), 'episode_level'))) AS n_episode_patient_days
FROM flow
GROUP BY step_name
ORDER BY step_name;


/* ============================================================
   4) Key and temporal integrity
   Expected: all checks should be 0, except null_discharge_date if open or
   missing discharge dates are allowed in the source.
   ============================================================ */
SELECT
    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS null_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS null_patient_id,
    SUM(CASE WHEN `data_index` IS NULL THEN 1 ELSE 0 END) AS null_model_day,
    SUM(CASE WHEN `DataIngres` IS NULL THEN 1 ELSE 0 END) AS null_admission_date,
    SUM(CASE WHEN `DataAlta` IS NULL THEN 1 ELSE 0 END) AS null_discharge_date,
    SUM(CASE WHEN `DataIniciUrgencies` > `DataIngres` THEN 1 ELSE 0 END) AS emergency_start_after_admission,
    SUM(CASE WHEN `dia_relatiu` < 0 THEN 1 ELSE 0 END) AS negative_relative_day,
    SUM(CASE WHEN DATE(`DataIngres`) > `data_index` THEN 1 ELSE 0 END) AS model_day_before_admission,
    SUM(CASE WHEN `DataAlta` IS NOT NULL AND DATE(`DataAlta`) < `data_index` THEN 1 ELSE 0 END) AS model_day_after_discharge,
    SUM(CASE WHEN DATE_ADD(DATE(`DataIngres`), INTERVAL `dia_relatiu` DAY) <> `data_index` THEN 1 ELSE 0 END) AS relative_day_inconsistency
FROM daily_sepsis_model;


/* ============================================================
   5) Upstream key uniqueness before the final join
   Expected: duplicate_keys = 0 in all daily upstream tables.
   ============================================================ */
SELECT
    'daily_vital_signs' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`)) AS unique_keys,
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`)) AS duplicate_keys
FROM daily_vital_signs

UNION ALL

SELECT
    'daily_laboratory_records',
    COUNT(*),
    COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`)),
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `data_index`))
FROM daily_laboratory_records

UNION ALL

SELECT
    'daily_pharmacy_features',
    COUNT(*),
    COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `Data_dia`)),
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '_', `Nhc`, '_', `Data_dia`))
FROM daily_pharmacy_features;


/* ============================================================
   6) Source coverage in the final model
   ============================================================ */
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE
        WHEN `SBP` IS NOT NULL
          OR `DBP` IS NOT NULL
          OR `TAM` IS NOT NULL
          OR `HR` IS NOT NULL
          OR `RESP` IS NOT NULL
          OR `O2SAT` IS NOT NULL
          OR `TEMP` IS NOT NULL
        THEN 1 ELSE 0
    END) AS rows_with_any_vital_sign,
    ROUND(100 * AVG(CASE
        WHEN `SBP` IS NOT NULL
          OR `DBP` IS NOT NULL
          OR `TAM` IS NOT NULL
          OR `HR` IS NOT NULL
          OR `RESP` IS NOT NULL
          OR `O2SAT` IS NOT NULL
          OR `TEMP` IS NOT NULL
        THEN 1 ELSE 0
    END), 2) AS pct_rows_with_any_vital_sign,
    SUM(CASE
        WHEN `creatinina` IS NOT NULL
          OR `plaquetes` IS NOT NULL
          OR `bilirubina_total` IS NOT NULL
          OR `leucocits` IS NOT NULL
          OR `hemoglobina` IS NOT NULL
          OR `pcr` IS NOT NULL
          OR `procalcitonina` IS NOT NULL
          OR `lactat_arterial` IS NOT NULL
          OR `lactat_venos` IS NOT NULL
        THEN 1 ELSE 0
    END) AS rows_with_any_laboratory_value,
    ROUND(100 * AVG(CASE
        WHEN `creatinina` IS NOT NULL
          OR `plaquetes` IS NOT NULL
          OR `bilirubina_total` IS NOT NULL
          OR `leucocits` IS NOT NULL
          OR `hemoglobina` IS NOT NULL
          OR `pcr` IS NOT NULL
          OR `procalcitonina` IS NOT NULL
          OR `lactat_arterial` IS NOT NULL
          OR `lactat_venos` IS NOT NULL
        THEN 1 ELSE 0
    END), 2) AS pct_rows_with_any_laboratory_value,
    SUM(CASE WHEN `antibiotic` = 1 THEN 1 ELSE 0 END) AS rows_with_antibiotic,
    ROUND(100 * AVG(CASE WHEN `antibiotic` = 1 THEN 1 ELSE 0 END), 2) AS pct_rows_with_antibiotic,
    SUM(CASE WHEN `vasopressor_qualsevol` = 1 THEN 1 ELSE 0 END) AS rows_with_any_vasopressor,
    ROUND(100 * AVG(CASE WHEN `vasopressor_qualsevol` = 1 THEN 1 ELSE 0 END), 2) AS pct_rows_with_any_vasopressor
FROM daily_sepsis_model;


/* ============================================================
   7) Coverage of key SOFA variables
   ============================================================ */
SELECT 'respiratory' AS sofa_component,
       SUM(CASE WHEN `pao2_arterial` IS NOT NULL AND `FIO2` IS NOT NULL AND `FIO2` > 0 THEN 1 ELSE 0 END) AS n_calculable_rows,
       COUNT(*) AS total_rows,
       ROUND(100 * AVG(CASE WHEN `pao2_arterial` IS NOT NULL AND `FIO2` IS NOT NULL AND `FIO2` > 0 THEN 1 ELSE 0 END), 2) AS pct_calculable_rows
FROM daily_sepsis_model
UNION ALL SELECT 'coagulation', SUM(CASE WHEN `plaquetes` IS NOT NULL THEN 1 ELSE 0 END), COUNT(*), ROUND(100 * AVG(CASE WHEN `plaquetes` IS NOT NULL THEN 1 ELSE 0 END), 2) FROM daily_sepsis_model
UNION ALL SELECT 'hepatic', SUM(CASE WHEN `bilirubina_total` IS NOT NULL THEN 1 ELSE 0 END), COUNT(*), ROUND(100 * AVG(CASE WHEN `bilirubina_total` IS NOT NULL THEN 1 ELSE 0 END), 2) FROM daily_sepsis_model
UNION ALL SELECT 'cardiovascular', SUM(CASE WHEN `TAM` IS NOT NULL OR `vasopressor_qualsevol` = 1 THEN 1 ELSE 0 END), COUNT(*), ROUND(100 * AVG(CASE WHEN `TAM` IS NOT NULL OR `vasopressor_qualsevol` = 1 THEN 1 ELSE 0 END), 2) FROM daily_sepsis_model
UNION ALL SELECT 'neurological', SUM(CASE WHEN `GLASGOW` IS NOT NULL THEN 1 ELSE 0 END), COUNT(*), ROUND(100 * AVG(CASE WHEN `GLASGOW` IS NOT NULL THEN 1 ELSE 0 END), 2) FROM daily_sepsis_model
UNION ALL SELECT 'renal', SUM(CASE WHEN `creatinina` IS NOT NULL OR `DIURESIS` IS NOT NULL THEN 1 ELSE 0 END), COUNT(*), ROUND(100 * AVG(CASE WHEN `creatinina` IS NOT NULL OR `DIURESIS` IS NOT NULL THEN 1 ELSE 0 END), 2) FROM daily_sepsis_model;


/* ============================================================
   8) Binary flag consistency
   Expected: all checks should be 0.
   ============================================================ */
SELECT
    SUM(CASE WHEN COALESCE(`hemocultiu_positiu`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_blood_culture_flag,
    SUM(CASE WHEN COALESCE(`ag_pneumococ`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_pneumococcal_antigen_flag,
    SUM(CASE WHEN COALESCE(`ag_legionella`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_legionella_antigen_flag,
    SUM(CASE WHEN COALESCE(`cultiu_positiu_previ_90d`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_positive_culture_flag,
    SUM(CASE WHEN COALESCE(`colonitzacio_previa_blee`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_blee_flag,
    SUM(CASE WHEN COALESCE(`colonitzacio_previa_cre`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_cre_flag,
    SUM(CASE WHEN COALESCE(`colonitzacio_previa_mrsa`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_mrsa_flag,
    SUM(CASE WHEN COALESCE(`colonitzacio_previa_vre`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_vre_flag,
    SUM(CASE WHEN COALESCE(`dispositius_invasius_previs`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_invasive_devices_flag,
    SUM(CASE WHEN COALESCE(`antibiotic`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_antibiotic_flag,
    SUM(CASE WHEN COALESCE(`antibiotics_previs_90d`, 0) NOT IN (0, 1) THEN 1 ELSE 0 END) AS non_binary_prior_antibiotics_90d_flag,
    SUM(CASE WHEN COALESCE(`vasopressor_multiple`, 0) = 1 AND COALESCE(`vasopressor_qualsevol`, 0) = 0 THEN 1 ELSE 0 END) AS multiple_vasopressor_without_any_vasopressor
FROM daily_sepsis_model;


/* ============================================================
   9) Temporal consistency of cumulative variables
   Expected: all checks should be 0.
   ============================================================ */
WITH ordered_model AS (
    SELECT
        `Episodi`,
        `Nhc`,
        `data_index`,
        `passa_per_critics`,
        `temps_critics`,
        `cirurgia`,
        `temps_cirurgia`,
        LAG(`passa_per_critics`) OVER (PARTITION BY `Episodi`, `Nhc` ORDER BY `data_index`) AS previous_any_critical_care,
        LAG(`temps_critics`) OVER (PARTITION BY `Episodi`, `Nhc` ORDER BY `data_index`) AS previous_critical_care_hours,
        LAG(`cirurgia`) OVER (PARTITION BY `Episodi`, `Nhc` ORDER BY `data_index`) AS previous_surgery,
        LAG(`temps_cirurgia`) OVER (PARTITION BY `Episodi`, `Nhc` ORDER BY `data_index`) AS previous_surgery_hours
    FROM daily_sepsis_model
)
SELECT
    SUM(CASE WHEN previous_any_critical_care = 1 AND `passa_per_critics` = 0 THEN 1 ELSE 0 END) AS critical_care_flag_returns_to_zero,
    SUM(CASE WHEN `temps_critics` < previous_critical_care_hours THEN 1 ELSE 0 END) AS decreasing_critical_care_hours,
    SUM(CASE WHEN previous_surgery = 1 AND `cirurgia` = 0 THEN 1 ELSE 0 END) AS surgery_flag_returns_to_zero,
    SUM(CASE WHEN `temps_cirurgia` IS NOT NULL AND previous_surgery_hours IS NOT NULL AND `temps_cirurgia` < previous_surgery_hours THEN 1 ELSE 0 END) AS decreasing_surgery_hours
FROM ordered_model;


/* ============================================================
   10) Plausible clinical ranges
   These checks flag values to review in Python; they do not modify data.
   ============================================================ */
SELECT
    SUM(CASE WHEN `SBP` IS NOT NULL AND (`SBP` < 40 OR `SBP` > 300) THEN 1 ELSE 0 END) AS n_sbp_out_of_range,
    SUM(CASE WHEN `DBP` IS NOT NULL AND (`DBP` < 20 OR `DBP` > 200) THEN 1 ELSE 0 END) AS n_dbp_out_of_range,
    SUM(CASE WHEN `TAM` IS NOT NULL AND (`TAM` < 20 OR `TAM` > 220) THEN 1 ELSE 0 END) AS n_map_out_of_range,
    SUM(CASE WHEN `HR` IS NOT NULL AND (`HR` < 20 OR `HR` > 250) THEN 1 ELSE 0 END) AS n_hr_out_of_range,
    SUM(CASE WHEN `RESP` IS NOT NULL AND (`RESP` < 4 OR `RESP` > 80) THEN 1 ELSE 0 END) AS n_resp_out_of_range,
    SUM(CASE WHEN `O2SAT` IS NOT NULL AND (`O2SAT` < 40 OR `O2SAT` > 100) THEN 1 ELSE 0 END) AS n_o2sat_out_of_range,
    SUM(CASE WHEN `TEMP` IS NOT NULL AND (`TEMP` < 30 OR `TEMP` > 43) THEN 1 ELSE 0 END) AS n_temp_out_of_range,
    SUM(CASE WHEN `FIO2` IS NOT NULL AND (`FIO2` < 21 OR `FIO2` > 100) THEN 1 ELSE 0 END) AS n_fio2_out_of_range,
    SUM(CASE WHEN `DIURESIS` IS NOT NULL AND (`DIURESIS` < 0 OR `DIURESIS` > 10000) THEN 1 ELSE 0 END) AS n_diuresis_out_of_range,
    SUM(CASE WHEN `GLASGOW` IS NOT NULL AND (`GLASGOW` < 1 OR `GLASGOW` > 15) THEN 1 ELSE 0 END) AS n_glasgow_out_of_range,
    SUM(CASE WHEN `creatinina` IS NOT NULL AND (`creatinina` < 0 OR `creatinina` > 20) THEN 1 ELSE 0 END) AS n_creatinine_out_of_range,
    SUM(CASE WHEN `plaquetes` IS NOT NULL AND (`plaquetes` < 0 OR `plaquetes` > 2000) THEN 1 ELSE 0 END) AS n_platelets_out_of_range,
    SUM(CASE WHEN `bilirubina_total` IS NOT NULL AND (`bilirubina_total` < 0 OR `bilirubina_total` > 40) THEN 1 ELSE 0 END) AS n_total_bilirubin_out_of_range,
    SUM(CASE WHEN `pao2_arterial` IS NOT NULL AND (`pao2_arterial` < 20 OR `pao2_arterial` > 600) THEN 1 ELSE 0 END) AS n_arterial_pao2_out_of_range,
    SUM(CASE WHEN `lactat_arterial` IS NOT NULL AND (`lactat_arterial` <= 0 OR `lactat_arterial` > 200) THEN 1 ELSE 0 END) AS n_arterial_lactate_out_of_range,
    SUM(CASE WHEN `lactat_venos` IS NOT NULL AND (`lactat_venos` <= 0 OR `lactat_venos` > 200) THEN 1 ELSE 0 END) AS n_venous_lactate_out_of_range,
    SUM(CASE WHEN `ph_arterial` IS NOT NULL AND (`ph_arterial` < 6.5 OR `ph_arterial` > 7.8) THEN 1 ELSE 0 END) AS n_arterial_ph_out_of_range,
    SUM(CASE WHEN `ph_venos` IS NOT NULL AND (`ph_venos` < 6.5 OR `ph_venos` > 7.8) THEN 1 ELSE 0 END) AS n_venous_ph_out_of_range
FROM daily_sepsis_model;


/* ============================================================
   11) Basic distributions for EDA
   ============================================================ */
SELECT
    YEAR(`DataIngres`) AS admission_year,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(*) AS n_day_rows
FROM daily_sepsis_model
GROUP BY YEAR(`DataIngres`)
ORDER BY admission_year;

SELECT
    `sexe`,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(*) AS n_day_rows,
    ROUND(AVG(`edat`), 2) AS mean_age
FROM daily_sepsis_model
GROUP BY `sexe`
ORDER BY n_episodes DESC;

SELECT
    `font_admissio`,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(*) AS n_day_rows
FROM daily_sepsis_model
GROUP BY `font_admissio`
ORDER BY n_episodes DESC;


/* ============================================================
   12) Final sample for manual review
   ============================================================ */
SELECT *
FROM daily_sepsis_model
ORDER BY `Episodi`, `data_index`
LIMIT 50;
