/* ============================================================
   BASIC VALIDATION OF daily_vital_signs

   Purpose:
   - Confirm one row per episode-patient-day.
   - Check key completeness, date consistency, variable coverage, and ranges.
   - Review oxygen support, Glasgow construction, and selected manual samples.
   ============================================================ */


/* ------------------------------------------------------------
   1) Executive quality summary
   Compact overview of table size, uniqueness, missing keys, temporal issues,
   and impossible values after the cleaning step.
   ------------------------------------------------------------ */
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients,
    COUNT(DISTINCT `data_index`) AS n_calendar_days,
    COUNT(*) - COUNT(DISTINCT CONCAT(`Episodi`, '|', `Nhc`, '|', `data_index`)) AS n_duplicate_episode_patient_days,

    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS n_missing_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS n_missing_patient_id,
    SUM(CASE WHEN `data_index` IS NULL THEN 1 ELSE 0 END) AS n_missing_day,

    SUM(CASE WHEN `porta_o2` IS NULL OR `porta_o2` NOT IN (0, 1) THEN 1 ELSE 0 END) AS n_invalid_oxygen_flag,

    SUM(CASE WHEN `SBP` IS NOT NULL AND (`SBP` < 40 OR `SBP` > 300) THEN 1 ELSE 0 END) AS n_sbp_out_of_range,
    SUM(CASE WHEN `DBP` IS NOT NULL AND (`DBP` < 20 OR `DBP` > 200) THEN 1 ELSE 0 END) AS n_dbp_out_of_range,
    SUM(CASE WHEN `TAM` IS NOT NULL AND (`TAM` < 20 OR `TAM` > 220) THEN 1 ELSE 0 END) AS n_tam_out_of_range,
    SUM(CASE WHEN `HR` IS NOT NULL AND (`HR` < 20 OR `HR` > 250) THEN 1 ELSE 0 END) AS n_hr_out_of_range,
    SUM(CASE WHEN `RESP` IS NOT NULL AND (`RESP` < 4 OR `RESP` > 80) THEN 1 ELSE 0 END) AS n_resp_out_of_range,
    SUM(CASE WHEN `O2SAT` IS NOT NULL AND (`O2SAT` < 40 OR `O2SAT` > 100) THEN 1 ELSE 0 END) AS n_o2sat_out_of_range,
    SUM(CASE WHEN `TEMP` IS NOT NULL AND (`TEMP` < 30 OR `TEMP` > 43) THEN 1 ELSE 0 END) AS n_temp_out_of_range,
    SUM(CASE WHEN `FIO2` IS NOT NULL AND (`FIO2` < 21 OR `FIO2` > 100) THEN 1 ELSE 0 END) AS n_fio2_out_of_range,
    SUM(CASE WHEN `DIURESIS` IS NOT NULL AND (`DIURESIS` < 0 OR `DIURESIS` > 10000) THEN 1 ELSE 0 END) AS n_diuresis_out_of_range,
    SUM(CASE WHEN `GLASGOW` IS NOT NULL AND (`GLASGOW` < 1 OR `GLASGOW` > 15) THEN 1 ELSE 0 END) AS n_glasgow_out_of_range,

    SUM(
        CASE
            WHEN `SBP` IS NULL
             AND `DBP` IS NULL
             AND `TAM` IS NULL
             AND `HR` IS NULL
             AND `RESP` IS NULL
             AND `O2SAT` IS NULL
             AND `TEMP` IS NULL
             AND `FIO2` IS NULL
             AND `DIURESIS` IS NULL
             AND `GLASGOW` IS NULL
             AND `porta_o2` = 0
            THEN 1 ELSE 0
        END
    ) AS n_days_without_useful_vital_signs
FROM daily_vital_signs;


/* ------------------------------------------------------------
   2) Duplicate episode-patient-days
   Expected result: no rows. The final table should contain one row per
   Episodi + Nhc + data_index.
   ------------------------------------------------------------ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    COUNT(*) AS n_rows
FROM daily_vital_signs
GROUP BY
    `Episodi`,
    `Nhc`,
    `data_index`
HAVING COUNT(*) > 1
ORDER BY n_rows DESC, `Episodi`, `data_index`
LIMIT 100;


/* ------------------------------------------------------------
   3) Temporal consistency with the base cohort
   Expected result: no rows. Vital-sign days should not be before the hospital
   admission date. Discharge date is not used here because this table is built
   from observed vital-sign days and does not filter by DataAlta.
   ------------------------------------------------------------ */
SELECT
    v.`Episodi`,
    v.`Nhc`,
    v.`data_index`,
    b.`DataIngres`
FROM daily_vital_signs v
INNER JOIN base_hospitalization_cohort b
    ON v.`Episodi` = b.`Episodi`
WHERE v.`data_index` < DATE(b.`DataIngres`)
ORDER BY v.`Episodi`, v.`data_index`
LIMIT 100;


/* ------------------------------------------------------------
   4) Variable coverage
   Counts and percentages of available daily values. For porta_o2, coverage is
   interpreted as days with active oxygen support.
   ------------------------------------------------------------ */
SELECT 'SBP' AS variable_name, SUM(`SBP` IS NOT NULL) AS n_available, ROUND(100 * AVG(`SBP` IS NOT NULL), 2) AS pct_available FROM daily_vital_signs
UNION ALL SELECT 'DBP', SUM(`DBP` IS NOT NULL), ROUND(100 * AVG(`DBP` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'TAM', SUM(`TAM` IS NOT NULL), ROUND(100 * AVG(`TAM` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'HR', SUM(`HR` IS NOT NULL), ROUND(100 * AVG(`HR` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'RESP', SUM(`RESP` IS NOT NULL), ROUND(100 * AVG(`RESP` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'O2SAT', SUM(`O2SAT` IS NOT NULL), ROUND(100 * AVG(`O2SAT` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'TEMP', SUM(`TEMP` IS NOT NULL), ROUND(100 * AVG(`TEMP` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'FIO2', SUM(`FIO2` IS NOT NULL), ROUND(100 * AVG(`FIO2` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'DIURESIS', SUM(`DIURESIS` IS NOT NULL), ROUND(100 * AVG(`DIURESIS` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'GLASGOW', SUM(`GLASGOW` IS NOT NULL), ROUND(100 * AVG(`GLASGOW` IS NOT NULL), 2) FROM daily_vital_signs
UNION ALL SELECT 'porta_o2', SUM(`porta_o2` = 1), ROUND(100 * AVG(`porta_o2` = 1), 2) FROM daily_vital_signs
ORDER BY variable_name;


/* ------------------------------------------------------------
   5) Observed numeric ranges
   Quick check of minimum and maximum values after operational cleaning.
   ------------------------------------------------------------ */
SELECT
    MIN(`SBP`) AS min_sbp, MAX(`SBP`) AS max_sbp,
    MIN(`DBP`) AS min_dbp, MAX(`DBP`) AS max_dbp,
    MIN(`TAM`) AS min_tam, MAX(`TAM`) AS max_tam,
    MIN(`HR`) AS min_hr, MAX(`HR`) AS max_hr,
    MIN(`RESP`) AS min_resp, MAX(`RESP`) AS max_resp,
    MIN(`O2SAT`) AS min_o2sat, MAX(`O2SAT`) AS max_o2sat,
    MIN(`TEMP`) AS min_temp, MAX(`TEMP`) AS max_temp,
    MIN(`FIO2`) AS min_fio2, MAX(`FIO2`) AS max_fio2,
    MIN(`DIURESIS`) AS min_diuresis, MAX(`DIURESIS`) AS max_diuresis,
    MIN(`GLASGOW`) AS min_glasgow, MAX(`GLASGOW`) AS max_glasgow
FROM daily_vital_signs;


/* ------------------------------------------------------------
   6) Rows with range or empty-day anomalies
   Manual review sample for values that should normally have been removed by
   cleaning, or rows with no useful daily information.
   ------------------------------------------------------------ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    `SBP`, `DBP`, `TAM`, `HR`, `RESP`, `O2SAT`, `TEMP`, `FIO2`, `DIURESIS`, `GLASGOW`, `porta_o2`
FROM daily_vital_signs
WHERE (`SBP` IS NOT NULL AND (`SBP` < 40 OR `SBP` > 300))
   OR (`DBP` IS NOT NULL AND (`DBP` < 20 OR `DBP` > 200))
   OR (`TAM` IS NOT NULL AND (`TAM` < 20 OR `TAM` > 220))
   OR (`HR` IS NOT NULL AND (`HR` < 20 OR `HR` > 250))
   OR (`RESP` IS NOT NULL AND (`RESP` < 4 OR `RESP` > 80))
   OR (`O2SAT` IS NOT NULL AND (`O2SAT` < 40 OR `O2SAT` > 100))
   OR (`TEMP` IS NOT NULL AND (`TEMP` < 30 OR `TEMP` > 43))
   OR (`FIO2` IS NOT NULL AND (`FIO2` < 21 OR `FIO2` > 100))
   OR (`DIURESIS` IS NOT NULL AND (`DIURESIS` < 0 OR `DIURESIS` > 10000))
   OR (`GLASGOW` IS NOT NULL AND (`GLASGOW` < 1 OR `GLASGOW` > 15))
   OR (
        `SBP` IS NULL AND `DBP` IS NULL AND `TAM` IS NULL AND `HR` IS NULL
    AND `RESP` IS NULL AND `O2SAT` IS NULL AND `TEMP` IS NULL AND `FIO2` IS NULL
    AND `DIURESIS` IS NULL AND `GLASGOW` IS NULL AND `porta_o2` = 0
      )
ORDER BY `Episodi`, `data_index`
LIMIT 100;


/* ------------------------------------------------------------
   7) Glasgow component completeness in the cleaned event table
   Quantifies how often the final daily Glasgow score comes from complete vs.
   partial component information.
   ------------------------------------------------------------ */
WITH glasgow_components AS (
    SELECT
        vc.`Episodi`,
        vc.`Nhc`,
        vc.`data_index`,
        vc.`event_time`,
        MAX(CASE WHEN vc.`variable_std` = 'GLASGOW_OCULAR' AND vc.`numeric_value_clean` IS NOT NULL THEN 1 ELSE 0 END) AS has_ocular,
        MAX(CASE WHEN vc.`variable_std` = 'GLASGOW_VERBAL' AND vc.`numeric_value_clean` IS NOT NULL THEN 1 ELSE 0 END) AS has_verbal,
        MAX(CASE WHEN vc.`variable_std` = 'GLASGOW_MOTORA' AND vc.`numeric_value_clean` IS NOT NULL THEN 1 ELSE 0 END) AS has_motor
    FROM clean_vital_signs_events vc
    WHERE vc.`variable_std` IN ('GLASGOW_OCULAR', 'GLASGOW_VERBAL', 'GLASGOW_MOTORA')
    GROUP BY
        vc.`Episodi`,
        vc.`Nhc`,
        vc.`data_index`,
        vc.`event_time`
),
glasgow_days AS (
    SELECT
        `Episodi`,
        `Nhc`,
        `data_index`,
        MAX(CASE WHEN has_ocular + has_verbal + has_motor = 3 THEN 1 ELSE 0 END) AS has_complete_glasgow,
        MAX(CASE WHEN has_ocular + has_verbal + has_motor BETWEEN 1 AND 2 THEN 1 ELSE 0 END) AS has_partial_glasgow
    FROM glasgow_components
    GROUP BY
        `Episodi`,
        `Nhc`,
        `data_index`
)
SELECT
    COUNT(*) AS n_days_with_any_glasgow_component,
    SUM(CASE WHEN has_complete_glasgow = 1 THEN 1 ELSE 0 END) AS n_days_with_complete_glasgow,
    SUM(CASE WHEN has_partial_glasgow = 1 THEN 1 ELSE 0 END) AS n_days_with_any_partial_glasgow,
    SUM(CASE WHEN has_complete_glasgow = 0 AND has_partial_glasgow = 1 THEN 1 ELSE 0 END) AS n_days_with_only_partial_glasgow
FROM glasgow_days;


/* ------------------------------------------------------------
   8) Mean arterial pressure consistency sample
   Manual check comparing observed TAM with the value estimated from
   SBP and DBP. This is not a strict exclusion rule.
   ------------------------------------------------------------ */
SELECT
    `Episodi`,
    `Nhc`,
    `data_index`,
    `SBP`,
    `DBP`,
    `TAM`,
    ROUND((`SBP` + 2 * `DBP`) / 3, 2) AS calculated_tam,
    ROUND(ABS(`TAM` - ((`SBP` + 2 * `DBP`) / 3)), 2) AS tam_difference
FROM daily_vital_signs
WHERE `SBP` IS NOT NULL
  AND `DBP` IS NOT NULL
  AND `TAM` IS NOT NULL
ORDER BY tam_difference DESC
LIMIT 100;


/* ------------------------------------------------------------
   9) Episodes with the longest vital-sign records
   Helps identify very long admissions or possible date anomalies.
   ------------------------------------------------------------ */
SELECT
    v.`Episodi`,
    MIN(v.`data_index`) AS first_vital_sign_day,
    MAX(v.`data_index`) AS last_vital_sign_day,
    COUNT(*) AS n_vital_sign_days,
    DATE(b.`DataIngres`) AS admission_day
FROM daily_vital_signs v
INNER JOIN base_hospitalization_cohort b
    ON v.`Episodi` = b.`Episodi`
GROUP BY
    v.`Episodi`,
    b.`DataIngres`
ORDER BY n_vital_sign_days DESC
LIMIT 100;


/* ------------------------------------------------------------
   10) Manual inspection sample
   Small ordered sample of final rows for visual review.
   ------------------------------------------------------------ */
SELECT *
FROM daily_vital_signs
ORDER BY `Episodi`, `data_index`
LIMIT 100;
