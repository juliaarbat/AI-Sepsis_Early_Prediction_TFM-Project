/* ============================================================
   BASIC VALIDATION OF daily_laboratory_records

   Purpose:
   - Confirm one row per Episodi + Nhc + data_index.
   - Check temporal consistency with the base cohort and daily model days.
   - Review laboratory coverage, numeric ranges, microbiology flags, and
     traceability from raw source records to the final daily table.
   ============================================================ */


/* ============================================================
   1) Executive quality summary
   Compact overview of volume, uniqueness, missing keys, temporal issues,
   invalid flags, and main out-of-range values after cleaning.
   ============================================================ */
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients,
    COUNT(DISTINCT CONCAT(`Episodi`, '|', `Nhc`, '|', `data_index`)) AS n_episode_patient_days,
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '|', `Nhc`, '|', `data_index`)) AS n_duplicate_episode_patient_days,

    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS n_missing_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS n_missing_patient_id,
    SUM(CASE WHEN `data_index` IS NULL THEN 1 ELSE 0 END) AS n_missing_day,

    SUM(CASE WHEN `creatinina` IS NOT NULL AND (`creatinina` <= 0 OR `creatinina` > 20) THEN 1 ELSE 0 END) AS n_creatinine_out_of_range,
    SUM(CASE WHEN `plaquetes` IS NOT NULL AND (`plaquetes` <= 0 OR `plaquetes` > 2000) THEN 1 ELSE 0 END) AS n_platelets_out_of_range,
    SUM(CASE WHEN `bilirubina_total` IS NOT NULL AND (`bilirubina_total` <= 0 OR `bilirubina_total` > 40) THEN 1 ELSE 0 END) AS n_bilirubin_out_of_range,
    SUM(CASE WHEN `leucocits` IS NOT NULL AND (`leucocits` <= 0 OR `leucocits` > 200) THEN 1 ELSE 0 END) AS n_leukocytes_out_of_range,
    SUM(CASE WHEN `hemoglobina` IS NOT NULL AND (`hemoglobina` <= 0 OR `hemoglobina` > 25) THEN 1 ELSE 0 END) AS n_hemoglobin_out_of_range,
    SUM(CASE WHEN `lactat_arterial` IS NOT NULL AND (`lactat_arterial` <= 0 OR `lactat_arterial` > 200) THEN 1 ELSE 0 END) AS n_arterial_lactate_out_of_range,
    SUM(CASE WHEN `lactat_venos` IS NOT NULL AND (`lactat_venos` <= 0 OR `lactat_venos` > 200) THEN 1 ELSE 0 END) AS n_venous_lactate_out_of_range,
    SUM(CASE WHEN `ph_arterial` IS NOT NULL AND (`ph_arterial` < 6.5 OR `ph_arterial` > 7.8) THEN 1 ELSE 0 END) AS n_arterial_ph_out_of_range,
    SUM(CASE WHEN `ph_venos` IS NOT NULL AND (`ph_venos` < 6.5 OR `ph_venos` > 7.8) THEN 1 ELSE 0 END) AS n_venous_ph_out_of_range,

    SUM(CASE WHEN `hemocultiu_positiu` NOT IN (0, 1) OR `hemocultiu_positiu` IS NULL THEN 1 ELSE 0 END) AS n_invalid_positive_blood_culture_flag,
    SUM(CASE WHEN `ag_pneumococ` NOT IN (0, 1) OR `ag_pneumococ` IS NULL THEN 1 ELSE 0 END) AS n_invalid_pneumococcus_antigen_flag,
    SUM(CASE WHEN `ag_legionella` NOT IN (0, 1) OR `ag_legionella` IS NULL THEN 1 ELSE 0 END) AS n_invalid_legionella_antigen_flag,
    SUM(CASE WHEN `cultiu_positiu_previ_90d` NOT IN (0, 1) OR `cultiu_positiu_previ_90d` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_positive_culture_flag,
    SUM(CASE WHEN `colonitzacio_previa_blee` NOT IN (0, 1) OR `colonitzacio_previa_blee` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_blee_flag,
    SUM(CASE WHEN `colonitzacio_previa_cre` NOT IN (0, 1) OR `colonitzacio_previa_cre` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_cre_flag,
    SUM(CASE WHEN `colonitzacio_previa_mrsa` NOT IN (0, 1) OR `colonitzacio_previa_mrsa` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_mrsa_flag,
    SUM(CASE WHEN `colonitzacio_previa_vre` NOT IN (0, 1) OR `colonitzacio_previa_vre` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_vre_flag
FROM daily_laboratory_records;


/* ============================================================
   2) Duplicate episode-patient-days
   Expected result: no rows. The final table should contain one row per
   Episodi + Nhc + data_index.
   ============================================================ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    COUNT(*) AS n_rows
FROM daily_laboratory_records
GROUP BY
    `Episodi`,
    `Nhc`,
    `data_index`
HAVING COUNT(*) > 1
ORDER BY n_rows DESC, `Episodi`, `data_index`
LIMIT 100;


/* ============================================================
   3) Temporal consistency with the base cohort
   Expected result: no rows before DataIniciUrgencies. Days before DataIngres
   can occur when laboratory results belong to the linked emergency episode.
   ============================================================ */
SELECT
    l.`Episodi`,
    l.`Nhc`,
    l.`data_index`,
    b.`DataIniciUrgencies`,
    b.`DataIngres`
FROM daily_laboratory_records l
INNER JOIN base_hospitalization_cohort b
    ON l.`Episodi` = b.`Episodi`
WHERE l.`data_index` < DATE(b.`DataIniciUrgencies`)
ORDER BY l.`Episodi`, l.`data_index`
LIMIT 100;


/* ============================================================
   4) Laboratory days within the hospital admission interval
   Expected: lab_days_after_discharge = 0.
   ============================================================ */
SELECT
    COUNT(*) AS total_lab_rows,
    SUM(CASE WHEN l.`data_index` < DATE(b.`DataIngres`) THEN 1 ELSE 0 END) AS lab_days_before_admission,
    SUM(CASE WHEN b.`DataAlta` IS NOT NULL AND l.`data_index` > DATE(b.`DataAlta`) THEN 1 ELSE 0 END) AS lab_days_after_discharge,
    COUNT(DISTINCT CASE WHEN b.`DataAlta` IS NOT NULL AND l.`data_index` > DATE(b.`DataAlta`) THEN l.`Episodi` END) AS episodes_with_lab_days_after_discharge,
    MIN(CASE WHEN b.`DataAlta` IS NOT NULL AND l.`data_index` > DATE(b.`DataAlta`) THEN DATEDIFF(l.`data_index`, DATE(b.`DataAlta`)) END) AS min_days_after_discharge,
    MAX(CASE WHEN b.`DataAlta` IS NOT NULL AND l.`data_index` > DATE(b.`DataAlta`) THEN DATEDIFF(l.`data_index`, DATE(b.`DataAlta`)) END) AS max_days_after_discharge
FROM daily_laboratory_records l
INNER JOIN base_hospitalization_cohort b
    ON l.`Episodi` = b.`Episodi`
   AND l.`Nhc` = b.`Nhc`;


/* ============================================================
   5) Laboratory variable coverage
   Counts and percentages of available daily values. The first rows focus on
   SOFA-relevant variables; the rest provide broader quality control.
   ============================================================ */
SELECT 'sofa' AS variable_group, 'creatinina' AS variable_name, SUM(`creatinina` IS NOT NULL) AS n_available, COUNT(*) AS n_rows, ROUND(100 * AVG(`creatinina` IS NOT NULL), 2) AS pct_available FROM daily_laboratory_records
UNION ALL SELECT 'sofa', 'plaquetes', SUM(`plaquetes` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`plaquetes` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'sofa', 'bilirubina_total', SUM(`bilirubina_total` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`bilirubina_total` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'ph_arterial', SUM(`ph_arterial` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`ph_arterial` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'pao2_arterial', SUM(`pao2_arterial` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`pao2_arterial` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'paco2_arterial', SUM(`paco2_arterial` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`paco2_arterial` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'lactat_arterial', SUM(`lactat_arterial` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`lactat_arterial` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'ph_venos', SUM(`ph_venos` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`ph_venos` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'pao2_venos', SUM(`pao2_venos` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`pao2_venos` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'paco2_venos', SUM(`paco2_venos` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`paco2_venos` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'gasometry', 'lactat_venos', SUM(`lactat_venos` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`lactat_venos` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'blood_count', 'hemoglobina', SUM(`hemoglobina` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`hemoglobina` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'blood_count', 'leucocits', SUM(`leucocits` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`leucocits` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'inflammation', 'pcr', SUM(`pcr` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`pcr` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'inflammation', 'procalcitonina', SUM(`procalcitonina` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`procalcitonina` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'microbiology', 'hemocultiu_positiu', SUM(`hemocultiu_positiu` = 1), COUNT(*), ROUND(100 * AVG(`hemocultiu_positiu` = 1), 2) FROM daily_laboratory_records
UNION ALL SELECT 'microbiology', 'hemocultiu_germen', SUM(`hemocultiu_germen` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`hemocultiu_germen` IS NOT NULL), 2) FROM daily_laboratory_records
UNION ALL SELECT 'microbiology', 'urocultiu_resultat', SUM(`urocultiu_resultat` IS NOT NULL), COUNT(*), ROUND(100 * AVG(`urocultiu_resultat` IS NOT NULL), 2) FROM daily_laboratory_records
ORDER BY variable_group, variable_name;


/* ============================================================
   6) Observed ranges for main numeric variables
   Reviews minimum, maximum, and mean values after operational cleaning.
   ============================================================ */
SELECT
    MIN(`creatinina`) AS min_creatinine,
    MAX(`creatinina`) AS max_creatinine,
    ROUND(AVG(`creatinina`), 3) AS mean_creatinine,
    MIN(`plaquetes`) AS min_platelets,
    MAX(`plaquetes`) AS max_platelets,
    ROUND(AVG(`plaquetes`), 3) AS mean_platelets,
    MIN(`bilirubina_total`) AS min_bilirubin,
    MAX(`bilirubina_total`) AS max_bilirubin,
    ROUND(AVG(`bilirubina_total`), 3) AS mean_bilirubin,
    MIN(`leucocits`) AS min_leukocytes,
    MAX(`leucocits`) AS max_leukocytes,
    ROUND(AVG(`leucocits`), 3) AS mean_leukocytes,
    MIN(`hemoglobina`) AS min_hemoglobin,
    MAX(`hemoglobina`) AS max_hemoglobin,
    ROUND(AVG(`hemoglobina`), 3) AS mean_hemoglobin,
    MIN(`lactat_arterial`) AS min_arterial_lactate,
    MAX(`lactat_arterial`) AS max_arterial_lactate,
    ROUND(AVG(`lactat_arterial`), 3) AS mean_arterial_lactate,
    MIN(`lactat_venos`) AS min_venous_lactate,
    MAX(`lactat_venos`) AS max_venous_lactate,
    ROUND(AVG(`lactat_venos`), 3) AS mean_venous_lactate,
    MIN(`ph_arterial`) AS min_arterial_ph,
    MAX(`ph_arterial`) AS max_arterial_ph,
    ROUND(AVG(`ph_arterial`), 3) AS mean_arterial_ph,
    MIN(`ph_venos`) AS min_venous_ph,
    MAX(`ph_venos`) AS max_venous_ph,
    ROUND(AVG(`ph_venos`), 3) AS mean_venous_ph
FROM daily_laboratory_records;


/* ============================================================
   7) Cleaning impact in clean_laboratory_events
   Compares original numeric values with cleaned values to detect overly
   restrictive ranges or unexpected unit problems.
   ============================================================ */
SELECT
    `variable_lab`,
    COUNT(*) AS n_rows,
    SUM(`ResultatNumèric` IS NOT NULL) AS n_original_numeric,
    SUM(`ResultatNumericClean` IS NOT NULL) AS n_clean_numeric,
    SUM(`ResultatNumèric` IS NOT NULL AND `ResultatNumericClean` IS NULL) AS n_removed_by_cleaning,
    ROUND(100 * AVG(`ResultatNumèric` IS NOT NULL AND `ResultatNumericClean` IS NULL), 2) AS pct_removed_by_cleaning,
    MIN(`ResultatNumèric`) AS min_original,
    MAX(`ResultatNumèric`) AS max_original,
    MIN(`ResultatNumericClean`) AS min_clean,
    MAX(`ResultatNumericClean`) AS max_clean
FROM clean_laboratory_events
WHERE `variable_lab` IS NOT NULL
GROUP BY `variable_lab`
ORDER BY pct_removed_by_cleaning DESC, `variable_lab`;


/* ============================================================
   8) Microbiology and temporal leakage checks
   Positive blood cultures should not be visible before the calculated
   availability date, and positive blood culture rows should have an anchor.
   ============================================================ */
SELECT
    SUM(`hemocultiu_positiu` = 1) AS n_positive_blood_culture_days,
    SUM(`hemocultiu_germen` IS NOT NULL) AS n_blood_culture_organism_days,
    SUM(`urocultiu_resultat` IS NOT NULL) AS n_urine_culture_result_days,
    SUM(`aspirat_traqueal_germen` IS NOT NULL) AS n_tracheal_aspirate_days,
    SUM(`broncoaspirat_germen` IS NOT NULL) AS n_bronchoaspirate_days,
    SUM(`bal_germen` IS NOT NULL) AS n_bal_days,
    SUM(`ag_pneumococ` = 1) AS n_pneumococcus_antigen_days,
    SUM(`ag_legionella` = 1) AS n_legionella_antigen_days,
    SUM(`cultiu_positiu_previ_90d` = 1) AS n_prior_positive_culture_days,
    SUM(`colonitzacio_previa_blee` = 1) AS n_prior_blee_days,
    SUM(`colonitzacio_previa_cre` = 1) AS n_prior_cre_days,
    SUM(`colonitzacio_previa_mrsa` = 1) AS n_prior_mrsa_days,
    SUM(`colonitzacio_previa_vre` = 1) AS n_prior_vre_days,
    SUM(
        CASE
            WHEN `hemocultiu_positiu` = 1
             AND `hemocultiu_positiu_data_extraccio` IS NOT NULL
             AND `hemocultiu_temps_positivitat_h` IS NOT NULL
             AND `data_index` < DATE(DATE_ADD(`hemocultiu_positiu_data_extraccio`, INTERVAL `hemocultiu_temps_positivitat_h` HOUR))
            THEN 1 ELSE 0
        END
    ) AS n_positive_blood_culture_before_availability,
    SUM(
        CASE
            WHEN `hemocultiu_positiu` = 1
             AND (`hemocultiu_positiu_data_extraccio` IS NULL OR `hemocultiu_temps_positivitat_h` IS NULL)
            THEN 1 ELSE 0
        END
    ) AS n_positive_blood_culture_without_anchor
FROM daily_laboratory_records;


/* ============================================================
   9) SOFA laboratory-variable availability by episode
   Summarizes how often creatinine, bilirubin, and platelets are available
   across episode days.
   ============================================================ */
WITH sofa_lab_per_episode AS (
    SELECT
        `Episodi`,
        COUNT(*) AS n_model_days,
        SUM(`creatinina` IS NOT NULL) AS n_days_with_creatinine,
        SUM(`bilirubina_total` IS NOT NULL) AS n_days_with_bilirubin,
        SUM(`plaquetes` IS NOT NULL) AS n_days_with_platelets
    FROM daily_laboratory_records
    GROUP BY `Episodi`
)
SELECT
    COUNT(*) AS n_episodes,
    ROUND(AVG(`n_model_days`), 2) AS mean_model_days,
    ROUND(100 * AVG(`n_days_with_creatinine` > 0), 2) AS pct_episodes_with_creatinine,
    ROUND(AVG(`n_days_with_creatinine`), 2) AS mean_days_with_creatinine,
    ROUND(100 * AVG(`n_days_with_bilirubin` > 0), 2) AS pct_episodes_with_bilirubin,
    ROUND(AVG(`n_days_with_bilirubin`), 2) AS mean_days_with_bilirubin,
    ROUND(100 * AVG(`n_days_with_platelets` > 0), 2) AS pct_episodes_with_platelets,
    ROUND(AVG(`n_days_with_platelets`), 2) AS mean_days_with_platelets
FROM sofa_lab_per_episode;


/* ============================================================
   10) Yearly traceability from base cohort to final laboratory records
   Separates loss due to missing raw laboratory data, cleaning, or final
   daily alignment.
   ============================================================ */
WITH raw_lab AS (
    SELECT DISTINCT b.`Episodi`
    FROM base_hospitalization_cohort b
    INNER JOIN tab_dt_sepsis_laboratori_001_ano l
        ON b.`Nhc` = l.`PacientSAP`
    WHERE l.`DataPetició` >= COALESCE(b.`DataIniciUrgencies`, b.`DataIngres`)
),
clean_lab AS (
    SELECT DISTINCT `Episodi`
    FROM clean_laboratory_events
),
daily_lab AS (
    SELECT DISTINCT `Episodi`
    FROM daily_laboratory_records
    WHERE `creatinina` IS NOT NULL
       OR `plaquetes` IS NOT NULL
       OR `bilirubina_total` IS NOT NULL
       OR `leucocits` IS NOT NULL
       OR `hemoglobina` IS NOT NULL
       OR `pcr` IS NOT NULL
       OR `procalcitonina` IS NOT NULL
       OR `hemocultiu_positiu` = 1
       OR `hemocultiu_germen` IS NOT NULL
)
SELECT
    YEAR(b.`DataIngres`) AS admission_year,
    COUNT(DISTINCT b.`Episodi`) AS n_base_cohort_episodes,
    COUNT(DISTINCT raw_lab.`Episodi`) AS n_episodes_with_raw_laboratory,
    COUNT(DISTINCT clean_lab.`Episodi`) AS n_episodes_with_clean_laboratory,
    COUNT(DISTINCT daily_lab.`Episodi`) AS n_episodes_with_daily_laboratory
FROM base_hospitalization_cohort b
LEFT JOIN raw_lab
    ON b.`Episodi` = raw_lab.`Episodi`
LEFT JOIN clean_lab
    ON b.`Episodi` = clean_lab.`Episodi`
LEFT JOIN daily_lab
    ON b.`Episodi` = daily_lab.`Episodi`
GROUP BY YEAR(b.`DataIngres`)
ORDER BY admission_year;


/* ============================================================
   11) Raw source coverage by request year
   Checks whether source laboratory volume is stable over the study period.
   A sharp drop in recent years may indicate incomplete source loading.
   ============================================================ */
SELECT
    YEAR(`DataPetició`) AS request_year,
    COUNT(*) AS n_raw_rows,
    COUNT(DISTINCT `PacientSAP`) AS n_patients,
    COUNT(DISTINCT CAST(`ProvaCodi` AS UNSIGNED)) AS n_test_codes
FROM tab_dt_sepsis_laboratori_001_ano
WHERE `DataPetició` IS NOT NULL
GROUP BY YEAR(`DataPetició`)
ORDER BY request_year;


/* ============================================================
   12) Expected numeric test codes by year
   Confirms that the mapped laboratory codes exist over time. Missing or
   disappearing codes may indicate source changes or incomplete loading.
   ============================================================ */
WITH expected_codes AS (
    SELECT 2640 AS test_code, 'ph_arterial' AS variable_lab UNION ALL
    SELECT 2642, 'pao2_arterial' UNION ALL
    SELECT 2641, 'paco2_arterial' UNION ALL
    SELECT 2643, 'bicarbonat_arterial' UNION ALL
    SELECT 2645, 'exc_base_arterial' UNION ALL
    SELECT 7336, 'lactat_arterial' UNION ALL
    SELECT 2653, 'ph_venos' UNION ALL
    SELECT 2655, 'pao2_venos' UNION ALL
    SELECT 2654, 'paco2_venos' UNION ALL
    SELECT 2656, 'bicarbonat_venos' UNION ALL
    SELECT 2658, 'exc_base_venos' UNION ALL
    SELECT 7339, 'lactat_venos' UNION ALL
    SELECT 2418, 'hematocrit' UNION ALL
    SELECT 2419, 'hemoglobina' UNION ALL
    SELECT 2428, 'leucocits' UNION ALL
    SELECT 2429, 'pct_neutrofils' UNION ALL
    SELECT 2439, 'granulocits_immadurs' UNION ALL
    SELECT 2424, 'plaquetes' UNION ALL
    SELECT 2470, 'fibrinogen' UNION ALL
    SELECT 2460, 'temps_protrombina_pct' UNION ALL
    SELECT 2691, 'pcr' UNION ALL
    SELECT 2693, 'procalcitonina' UNION ALL
    SELECT 2515, 'glucosa' UNION ALL
    SELECT 2516, 'urea' UNION ALL
    SELECT 2517, 'creatinina' UNION ALL
    SELECT 2526, 'bilirubina_total' UNION ALL
    SELECT 2529, 'got_ast' UNION ALL
    SELECT 9218, 'albumina' UNION ALL
    SELECT 2519, 'proteines_totals' UNION ALL
    SELECT 2689, 'troponina'
)
SELECT
    YEAR(l.`DataPetició`) AS request_year,
    e.`variable_lab`,
    e.`test_code`,
    COUNT(l.`ProvaCodi`) AS n_rows,
    COUNT(DISTINCT l.`PacientSAP`) AS n_patients,
    MIN(l.`ResultatNumèric`) AS min_result,
    MAX(l.`ResultatNumèric`) AS max_result,
    ROUND(AVG(l.`ResultatNumèric`), 3) AS mean_result
FROM expected_codes e
LEFT JOIN tab_dt_sepsis_laboratori_001_ano l
    ON CAST(l.`ProvaCodi` AS UNSIGNED) = e.`test_code`
   AND l.`ResultatNumèric` IS NOT NULL
GROUP BY
    YEAR(l.`DataPetició`),
    e.`variable_lab`,
    e.`test_code`
ORDER BY request_year, e.`variable_lab`;


/* ============================================================
   13) Manual review sample for extreme values
   Shows concrete rows for values that are valid after cleaning but clinically
   high or low enough to review during EDA.
   ============================================================ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    `creatinina`,
    `plaquetes`,
    `bilirubina_total`,
    `leucocits`,
    `hemoglobina`,
    `lactat_arterial`,
    `lactat_venos`,
    `ph_arterial`,
    `ph_venos`
FROM daily_laboratory_records
WHERE (`creatinina` IS NOT NULL AND (`creatinina` <= 0.05 OR `creatinina` >= 10))
   OR (`plaquetes` IS NOT NULL AND (`plaquetes` <= 10 OR `plaquetes` >= 1000))
   OR (`bilirubina_total` IS NOT NULL AND `bilirubina_total` >= 20)
   OR (`leucocits` IS NOT NULL AND (`leucocits` <= 0.5 OR `leucocits` >= 100))
   OR (`hemoglobina` IS NOT NULL AND (`hemoglobina` <= 5 OR `hemoglobina` >= 20))
   OR (`lactat_arterial` IS NOT NULL AND `lactat_arterial` >= 90)
   OR (`lactat_venos` IS NOT NULL AND `lactat_venos` >= 90)
   OR (`ph_arterial` IS NOT NULL AND (`ph_arterial` <= 6.8 OR `ph_arterial` >= 7.7))
   OR (`ph_venos` IS NOT NULL AND (`ph_venos` <= 6.8 OR `ph_venos` >= 7.7))
ORDER BY `data_index`, `Episodi`
LIMIT 100;
