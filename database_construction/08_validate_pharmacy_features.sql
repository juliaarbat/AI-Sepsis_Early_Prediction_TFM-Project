/* ============================================================
   BASIC VALIDATION OF daily_pharmacy_features

   Purpose:
   - Validate the cleaned pharmacy event table and the final daily table.
   - Confirm one row per Episodi + Nhc + Data_dia in daily_pharmacy_features.
   - Check temporal consistency, valid binary flags, vasopressor logic,
     antimicrobial duration, and basic source traceability.
   ============================================================ */


/* ============================================================
   1) Table volume summary
   Quick overview of the two pharmacy tables used downstream.
   ============================================================ */
SELECT
    'clean_pharmacy_events' AS table_name,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients,
    COUNT(DISTINCT `Data_dia`) AS n_days,
    COUNT(DISTINCT `Codi_medicament`) AS n_drugs
FROM clean_pharmacy_events

UNION ALL

SELECT
    'daily_pharmacy_features' AS table_name,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients,
    COUNT(DISTINCT `Data_dia`) AS n_days,
    NULL AS n_drugs
FROM daily_pharmacy_features;


/* ============================================================
   2) Key completeness
   Expected result: all missing-key counts should be 0.
   ============================================================ */
SELECT
    'clean_pharmacy_events' AS table_name,
    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS n_missing_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS n_missing_patient_id,
    SUM(CASE WHEN `Data_dia` IS NULL THEN 1 ELSE 0 END) AS n_missing_day
FROM clean_pharmacy_events

UNION ALL

SELECT
    'daily_pharmacy_features' AS table_name,
    SUM(CASE WHEN `Episodi` IS NULL THEN 1 ELSE 0 END) AS n_missing_episode,
    SUM(CASE WHEN `Nhc` IS NULL THEN 1 ELSE 0 END) AS n_missing_patient_id,
    SUM(CASE WHEN `Data_dia` IS NULL THEN 1 ELSE 0 END) AS n_missing_day
FROM daily_pharmacy_features;


/* ============================================================
   3) Duplicate daily rows
   Expected result: no rows. The final table should contain one row per
   Episodi + Nhc + Data_dia.
   ============================================================ */
SELECT
    `Episodi`,
    `Nhc`,
    `Data_dia`,
    COUNT(*) AS n_rows
FROM daily_pharmacy_features
GROUP BY
    `Episodi`,
    `Nhc`,
    `Data_dia`
HAVING COUNT(*) > 1
ORDER BY n_rows DESC, `Episodi`, `Data_dia`
LIMIT 100;


/* ============================================================
   4) Unexpected records in clean_pharmacy_events
   Expected result: no rows. Clean events should be vasopressors or
   antimicrobial therapeutic groups included in the operational definition.
   ============================================================ */
SELECT *
FROM clean_pharmacy_events
WHERE `Codi_medicament` NOT IN (1968, 25143, 149, 28823, 655)
  AND UPPER(COALESCE(`Descripcio_medicament`, '')) NOT LIKE '%NORADRENALINA%'
  AND UPPER(COALESCE(`Principi_actiu`, '')) NOT LIKE '%NORADRENALINA%'
  AND UPPER(COALESCE(`Descripcio_medicament`, '')) NOT LIKE '%NOREPINEFRINA%'
  AND UPPER(COALESCE(`Principi_actiu`, '')) NOT LIKE '%NOREPINEFRINA%'
  AND UPPER(COALESCE(`Descripcio_medicament`, '')) NOT LIKE '%ADRENALINA%'
  AND UPPER(COALESCE(`Principi_actiu`, '')) NOT LIKE '%ADRENALINA%'
  AND UPPER(COALESCE(`Descripcio_medicament`, '')) NOT LIKE '%EPINEFRINA%'
  AND UPPER(COALESCE(`Principi_actiu`, '')) NOT LIKE '%EPINEFRINA%'
  AND COALESCE(`Codi_grup_terapeutic`, '') NOT LIKE 'J01%'
  AND COALESCE(`Codi_grup_terapeutic`, '') NOT LIKE 'J02%'
  AND COALESCE(`Codi_grup_terapeutic`, '') NOT LIKE 'H02%'
  AND COALESCE(`Codi_grup_terapeutic`, '') NOT LIKE 'L01%'
LIMIT 200;


/* ============================================================
   5) Temporal consistency with the base cohort
   Expected result: no rows. Pharmacy events and daily pharmacy rows should
   not start before DataIngres.
   ============================================================ */
SELECT
    'clean_pharmacy_events' AS source_table,
    f.`Episodi`,
    f.`Nhc`,
    f.`Data_hora_consum` AS event_datetime,
    f.`Data_dia`,
    b.`DataIngres`
FROM clean_pharmacy_events f
INNER JOIN base_hospitalization_cohort b
    ON f.`Episodi` = b.`Episodi`
WHERE f.`Data_hora_consum` < b.`DataIngres`

UNION ALL

SELECT
    'daily_pharmacy_features' AS source_table,
    f.`Episodi`,
    f.`Nhc`,
    NULL AS event_datetime,
    f.`Data_dia`,
    b.`DataIngres`
FROM daily_pharmacy_features f
INNER JOIN base_hospitalization_cohort b
    ON f.`Episodi` = b.`Episodi`
WHERE f.`Data_dia` < DATE(b.`DataIngres`)
ORDER BY source_table, `Episodi`, `Data_dia`
LIMIT 200;


/* ============================================================
   6) Pharmacy exposure coverage
   Shows the main vasopressor and antimicrobial groups found in the cleaned
   event table.
   ============================================================ */
SELECT
    CASE
        WHEN `Codi_medicament` IN (1968, 25143) THEN 'Dobutamine'
        WHEN `Codi_medicament` = 149 THEN 'Dopamine'
        WHEN `Codi_medicament` IN (28823, 655)
          OR UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%NORADRENALINA%'
          OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%NORADRENALINA%'
          OR UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%NOREPINEFRINA%'
          OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%NOREPINEFRINA%' THEN 'Noradrenaline'
        WHEN (
                UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%ADRENALINA%'
             OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%ADRENALINA%'
             OR UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%EPINEFRINA%'
             OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%EPINEFRINA%'
             )
         AND NOT (
                UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%NORADRENALINA%'
             OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%NORADRENALINA%'
             OR UPPER(COALESCE(`Descripcio_medicament`, '')) LIKE '%NOREPINEFRINA%'
             OR UPPER(COALESCE(`Principi_actiu`, '')) LIKE '%NOREPINEFRINA%'
             ) THEN 'Adrenaline'
        WHEN `Codi_grup_terapeutic` LIKE 'J01%' THEN 'Antibiotic J01'
        WHEN `Codi_grup_terapeutic` LIKE 'J02%' THEN 'Antifungal J02'
        WHEN `Codi_grup_terapeutic` LIKE 'H02%' THEN 'H02'
        WHEN `Codi_grup_terapeutic` LIKE 'L01%' THEN 'L01'
        ELSE 'Other'
    END AS exposure_group,
    COUNT(*) AS n_events,
    COUNT(DISTINCT `Episodi`) AS n_episodes,
    COUNT(DISTINCT `Nhc`) AS n_patients
FROM clean_pharmacy_events
GROUP BY exposure_group
ORDER BY n_events DESC;


/* ============================================================
   7) Daily feature coverage
   Counts how many daily rows carry each pharmacy feature and the maximum
   estimated antimicrobial duration.
   ============================================================ */
SELECT
    COUNT(*) AS n_rows,
    SUM(CASE WHEN `vasopressor_dobutamina` = 1 THEN 1 ELSE 0 END) AS n_dobutamine_days,
    SUM(CASE WHEN `vasopressor_dopamina` = 1 THEN 1 ELSE 0 END) AS n_dopamine_days,
    SUM(CASE WHEN `vasopressor_noradrenalina` = 1 THEN 1 ELSE 0 END) AS n_noradrenaline_days,
    SUM(CASE WHEN `vasopressor_adrenalina` = 1 THEN 1 ELSE 0 END) AS n_adrenaline_days,
    SUM(CASE WHEN `vasopressor_qualsevol` = 1 THEN 1 ELSE 0 END) AS n_any_vasopressor_days,
    SUM(CASE WHEN `vasopressor_multiple` = 1 THEN 1 ELSE 0 END) AS n_multiple_vasopressor_days,
    SUM(CASE WHEN `antibiotic` = 1 THEN 1 ELSE 0 END) AS n_antimicrobial_days,
    SUM(CASE WHEN `atb_duracio` > 0 THEN 1 ELSE 0 END) AS n_positive_antimicrobial_duration_days,
    MAX(`atb_duracio`) AS max_antimicrobial_duration_hours,
    SUM(CASE WHEN `antibiotics_previs_90d` = 1 THEN 1 ELSE 0 END) AS n_prior_antimicrobial_90d_days
FROM daily_pharmacy_features;


/* ============================================================
   8) Invalid flags and duration values
   Expected result: all invalid counts should be 0.
   ============================================================ */
SELECT
    SUM(CASE WHEN `vasopressor_dobutamina` NOT IN (0, 1) OR `vasopressor_dobutamina` IS NULL THEN 1 ELSE 0 END) AS n_invalid_dobutamine_flag,
    SUM(CASE WHEN `vasopressor_dopamina` NOT IN (0, 1) OR `vasopressor_dopamina` IS NULL THEN 1 ELSE 0 END) AS n_invalid_dopamine_flag,
    SUM(CASE WHEN `vasopressor_noradrenalina` NOT IN (0, 1) OR `vasopressor_noradrenalina` IS NULL THEN 1 ELSE 0 END) AS n_invalid_noradrenaline_flag,
    SUM(CASE WHEN `vasopressor_adrenalina` NOT IN (0, 1) OR `vasopressor_adrenalina` IS NULL THEN 1 ELSE 0 END) AS n_invalid_adrenaline_flag,
    SUM(CASE WHEN `vasopressor_qualsevol` NOT IN (0, 1) OR `vasopressor_qualsevol` IS NULL THEN 1 ELSE 0 END) AS n_invalid_any_vasopressor_flag,
    SUM(CASE WHEN `vasopressor_multiple` NOT IN (0, 1) OR `vasopressor_multiple` IS NULL THEN 1 ELSE 0 END) AS n_invalid_multiple_vasopressor_flag,
    SUM(CASE WHEN `antibiotic` NOT IN (0, 1) OR `antibiotic` IS NULL THEN 1 ELSE 0 END) AS n_invalid_antimicrobial_flag,
    SUM(CASE WHEN `antibiotics_previs_90d` NOT IN (0, 1) OR `antibiotics_previs_90d` IS NULL THEN 1 ELSE 0 END) AS n_invalid_prior_antimicrobial_90d_flag,
    SUM(CASE WHEN `atb_duracio` IS NULL OR `atb_duracio` < 0 THEN 1 ELSE 0 END) AS n_invalid_antimicrobial_duration
FROM daily_pharmacy_features;


/* ============================================================
   9) Derived-feature consistency
   Expected result: no rows. Derived summary flags and antimicrobial duration
   should match the component features.
   ============================================================ */
SELECT *
FROM daily_pharmacy_features
WHERE `vasopressor_qualsevol` <>
      CASE
          WHEN COALESCE(`vasopressor_dobutamina`, 0)
             + COALESCE(`vasopressor_dopamina`, 0)
             + COALESCE(`vasopressor_noradrenalina`, 0)
             + COALESCE(`vasopressor_adrenalina`, 0) >= 1
          THEN 1 ELSE 0
      END
   OR `vasopressor_multiple` <>
      CASE
          WHEN COALESCE(`vasopressor_dobutamina`, 0)
             + COALESCE(`vasopressor_dopamina`, 0)
             + COALESCE(`vasopressor_noradrenalina`, 0)
             + COALESCE(`vasopressor_adrenalina`, 0) >= 2
          THEN 1 ELSE 0
      END
   OR (`antibiotic` = 0 AND `atb_duracio` <> 0)
   OR (`antibiotic` = 1 AND (`atb_duracio` <= 0 OR MOD(`atb_duracio`, 24) <> 0))
LIMIT 200;


/* ============================================================
   10) Number of vasopressors per day
   Helps identify monotherapy versus combined vasopressor exposure.
   ============================================================ */
SELECT
    (
        COALESCE(`vasopressor_dobutamina`, 0) +
        COALESCE(`vasopressor_dopamina`, 0) +
        COALESCE(`vasopressor_noradrenalina`, 0) +
        COALESCE(`vasopressor_adrenalina`, 0)
    ) AS n_vasopressors_day,
    COUNT(*) AS n_rows
FROM daily_pharmacy_features
GROUP BY n_vasopressors_day
ORDER BY n_vasopressors_day;


/* ============================================================
   11) Records with the longest pharmacy follow-up
   Useful to spot very long stays or unexpected daily-row generation.
   ============================================================ */
SELECT
    f.`Episodi`,
    f.`Nhc`,
    MIN(f.`Data_dia`) AS first_pharmacy_day,
    MAX(f.`Data_dia`) AS last_pharmacy_day,
    COUNT(*) AS n_pharmacy_days,
    DATE(b.`DataIngres`) AS admission_day
FROM daily_pharmacy_features f
INNER JOIN base_hospitalization_cohort b
    ON f.`Episodi` = b.`Episodi`
GROUP BY
    f.`Episodi`,
    f.`Nhc`,
    b.`DataIngres`
ORDER BY n_pharmacy_days DESC
LIMIT 100;


/* ============================================================
   12) Manual review samples
   Small samples for reviewing cleaned events, daily rows, and days with
   active pharmacy features.
   ============================================================ */
SELECT *
FROM clean_pharmacy_events
ORDER BY `Episodi`, `Data_hora_consum`
LIMIT 100;

SELECT *
FROM daily_pharmacy_features
ORDER BY `Episodi`, `Data_dia`
LIMIT 100;

SELECT *
FROM daily_pharmacy_features
WHERE `vasopressor_qualsevol` = 1
   OR `antibiotic` = 1
   OR `antibiotics_previs_90d` = 1
ORDER BY `Episodi`, `Data_dia`
LIMIT 100;
