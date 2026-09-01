/* =========================================================
   DAILY VITAL SIGNS FOR THE BASE COHORT

   Operational summary
   - Unit of analysis: one row per Episodi + Nhc + data_index.
   - Time window: vital signs recorded from DataIngres onward.
   - Daily rule: keep the last valid value of each variable within each day.
   - FIO2 is stored as a percentage: values 0.21-1 are converted to 21-100,
     while values already in the 21-100 range are kept.
   - Glasgow is computed as the sum of available daily components. Missing
     components contribute 0, and the final score is NULL if all are missing.
   - porta_o2 is 1 when any oxygen device is recorded on that day.

   Output table:
   - daily_vital_signs

   Intermediate table:
   - clean_vital_signs_events
   ========================================================= */

/* ---------------------------------------------------------
   RECOMMENDED SOURCE INDEXES
   Run only once if these indexes do not already exist.
   --------------------------------------------------------- */


CREATE INDEX idx_cv_episodi_data_variable
    ON tab_dt_sepsis_constants_vitals_001_ano (`episodi_sap`, `datahora_variable`, `variable_codi`);

CREATE INDEX idx_cv_variable_episodi_data
    ON tab_dt_sepsis_constants_vitals_001_ano (`variable_codi`, `episodi_sap`, `datahora_variable`);




/* ---------------------------------------------------------
   1) CLEAN VITAL SIGN EVENTS

   This materialized table keeps only vital-sign records from cohort episodes,
   normalizes numeric values, applies plausible clinical ranges, and creates
   one oxygen-device flag per raw event. It is kept as an audit table for
   discarded or transformed values.
   --------------------------------------------------------- */

DROP TABLE IF EXISTS clean_vital_signs_events;

CREATE TABLE clean_vital_signs_events AS
WITH

/* Base cohort episodes used to restrict the source vital-sign table. */
cohort_base AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        b.`DataIngres`
    FROM base_hospitalization_cohort b
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND b.`DataIngres` IS NOT NULL
),

/* Maps raw source variable codes to the standardized variables used later. */
map_vitals AS (
    SELECT 'VA30295548' AS `variable_codi`, 'SBP' AS `variable_std`, 'numeric' AS `value_type`
    UNION ALL SELECT 'VA66051657', 'DBP', 'numeric'
    UNION ALL SELECT 'VA16874736', 'TAM', 'numeric'
    UNION ALL SELECT 'VA27157783', 'HR', 'numeric'
    UNION ALL SELECT 'VA43911089', 'RESP', 'numeric'
    UNION ALL SELECT 'VA60459136', 'O2SAT', 'numeric'
    UNION ALL SELECT 'VA83961342', 'TEMP', 'numeric'
    UNION ALL SELECT 'VA22848586', 'FIO2', 'numeric'
    UNION ALL SELECT 'VA06545324', 'DIURESIS', 'numeric'
    UNION ALL SELECT 'VA0000004611', 'DISPOSITIU_O2', 'text'
    UNION ALL SELECT 'VA0000007595', 'GLASGOW_OCULAR', 'numeric'
    UNION ALL SELECT 'VA0000007743', 'GLASGOW_VERBAL', 'numeric'
    UNION ALL SELECT 'VA0000007764', 'GLASGOW_MOTORA', 'numeric'
),

/* Reads raw vital-sign records after hospital admission and parses values. */
vitals_raw AS (
    SELECT
        c.`Episodi`,
        c.`Nhc`,
        cv.`id`,
        cv.`datahora_variable` AS `event_time`,
        DATE(cv.`datahora_variable`) AS `data_index`,
        cv.`variable_codi`,
        m.`variable_std`,
        m.`value_type`,
        cv.`variable_nom_curt`,
        cv.`variable_especifica_codi`,
        cv.`unitat_codi`,
        cv.`unitat_descripcio_curta`,
        CASE
            WHEN cv.`valor_numeric` IS NOT NULL
             AND TRIM(cv.`valor_numeric`) <> ''
             AND REPLACE(TRIM(cv.`valor_numeric`), ',', '.') REGEXP '^-?([0-9]+([.][0-9]+)?|[.][0-9]+)$'
            THEN CAST(REPLACE(TRIM(cv.`valor_numeric`), ',', '.') AS DECIMAL(18,4))
            ELSE NULL
        END AS `numeric_value`,
        NULLIF(TRIM(cv.`valor_no_numeric`), '') AS `text_value`
    FROM tab_dt_sepsis_constants_vitals_001_ano cv
    INNER JOIN cohort_base c
        ON cv.`episodi_sap` = c.`Episodi`
    INNER JOIN map_vitals m
        ON cv.`variable_codi` = m.`variable_codi`
    WHERE cv.`episodi_sap` IS NOT NULL
      AND cv.`datahora_variable` IS NOT NULL
      AND cv.`datahora_variable` >= c.`DataIngres`
)

SELECT
    v.`Episodi`,
    v.`Nhc`,
    v.`id`,
    v.`event_time`,
    v.`data_index`,
    v.`variable_codi`,
    v.`variable_std`,
    v.`value_type`,
    v.`variable_nom_curt`,
    v.`variable_especifica_codi`,
    v.`unitat_codi`,
    v.`unitat_descripcio_curta`,
    v.`numeric_value`,
    v.`text_value`,

    CASE
        WHEN v.`variable_std` = 'SBP'
         AND v.`numeric_value` BETWEEN 40 AND 300
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'DBP'
         AND v.`numeric_value` BETWEEN 20 AND 200
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'TAM'
         AND v.`numeric_value` BETWEEN 20 AND 220
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'HR'
         AND v.`numeric_value` BETWEEN 20 AND 250
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'RESP'
         AND v.`numeric_value` BETWEEN 4 AND 80
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'O2SAT'
         AND v.`numeric_value` BETWEEN 40 AND 100
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'TEMP'
         AND v.`numeric_value` BETWEEN 30 AND 43
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'FIO2'
         AND v.`numeric_value` BETWEEN 0.21 AND 1
        THEN v.`numeric_value` * 100

        WHEN v.`variable_std` = 'FIO2'
         AND v.`numeric_value` BETWEEN 21 AND 100
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'DIURESIS'
         AND v.`numeric_value` BETWEEN 0 AND 10000
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'GLASGOW_OCULAR'
         AND v.`numeric_value` BETWEEN 1 AND 4
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'GLASGOW_VERBAL'
         AND v.`numeric_value` BETWEEN 1 AND 5
        THEN v.`numeric_value`

        WHEN v.`variable_std` = 'GLASGOW_MOTORA'
         AND v.`numeric_value` BETWEEN 1 AND 6
        THEN v.`numeric_value`

        ELSE NULL
    END AS `numeric_value_clean`,

    CASE
        WHEN v.`variable_std` = 'DISPOSITIU_O2'
         AND (
                v.`numeric_value` > 0
             OR (
                    v.`text_value` IS NOT NULL
                AND TRIM(v.`text_value`) <> ''
                AND UPPER(TRIM(v.`text_value`)) NOT IN (
                    'NO', 'N', '0', '0.0', 'FALSE', 'FALS',
                    'NO APLICA', 'N/A', 'NA', 'CAP', 'SENSE'
                )
             )
         )
        THEN 1
        ELSE 0
    END AS `porta_o2_flag`
FROM vitals_raw v;

CREATE INDEX idx_cvse_epi_data_variable
    ON clean_vital_signs_events (`Episodi`, `data_index`, `variable_std`, `event_time`);

CREATE INDEX idx_cvse_nhc_data
    ON clean_vital_signs_events (`Nhc`, `data_index`);


/* ---------------------------------------------------------
   2) DAILY VITAL SIGNS TABLE

   This table pivots the cleaned event-level records into one row per
   episode-patient-day. For each numeric variable, only the last valid daily
   value is kept. Oxygen support is summarized as a daily binary flag.
   --------------------------------------------------------- */

DROP TABLE IF EXISTS daily_vital_signs;

CREATE TABLE daily_vital_signs AS
WITH

/* Last valid numeric value per episode, patient, day, and variable. */
daily_last_numeric AS (
    SELECT
        x.`Episodi`,
        x.`Nhc`,
        x.`data_index`,
        x.`variable_std`,
        x.`numeric_value_clean`
    FROM (
        SELECT
            vc.*,
            ROW_NUMBER() OVER (
                PARTITION BY
                    vc.`Episodi`,
                    vc.`Nhc`,
                    vc.`data_index`,
                    vc.`variable_std`
                ORDER BY vc.`event_time` DESC, vc.`id` DESC
            ) AS rn
        FROM clean_vital_signs_events vc
        WHERE vc.`variable_std` IN (
            'SBP','DBP','TAM','HR','RESP','O2SAT','TEMP','FIO2',
            'DIURESIS','GLASGOW_OCULAR','GLASGOW_VERBAL','GLASGOW_MOTORA'
        )
          AND vc.`numeric_value_clean` IS NOT NULL
    ) x
    WHERE x.rn = 1
),

/* Daily oxygen-device flag. Any positive event makes the whole day positive. */
daily_porta_o2 AS (
    SELECT
        vc.`Episodi`,
        vc.`Nhc`,
        vc.`data_index`,
        MAX(vc.`porta_o2_flag`) AS `porta_o2`
    FROM clean_vital_signs_events vc
    WHERE vc.`variable_std` = 'DISPOSITIU_O2'
    GROUP BY
        vc.`Episodi`,
        vc.`Nhc`,
        vc.`data_index`
),

/* Pivot numeric variables from long event format to daily wide format. */
daily_numeric_pivot AS (
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`,
        MAX(CASE WHEN d.`variable_std` = 'SBP' THEN d.`numeric_value_clean` END) AS `SBP`,
        MAX(CASE WHEN d.`variable_std` = 'DBP' THEN d.`numeric_value_clean` END) AS `DBP`,
        MAX(CASE WHEN d.`variable_std` = 'TAM' THEN d.`numeric_value_clean` END) AS `TAM`,
        MAX(CASE WHEN d.`variable_std` = 'HR' THEN d.`numeric_value_clean` END) AS `HR`,
        MAX(CASE WHEN d.`variable_std` = 'RESP' THEN d.`numeric_value_clean` END) AS `RESP`,
        MAX(CASE WHEN d.`variable_std` = 'O2SAT' THEN d.`numeric_value_clean` END) AS `O2SAT`,
        MAX(CASE WHEN d.`variable_std` = 'TEMP' THEN d.`numeric_value_clean` END) AS `TEMP`,
        MAX(CASE WHEN d.`variable_std` = 'FIO2' THEN d.`numeric_value_clean` END) AS `FIO2`,
        MAX(CASE WHEN d.`variable_std` = 'DIURESIS' THEN d.`numeric_value_clean` END) AS `DIURESIS`,
        MAX(CASE WHEN d.`variable_std` = 'GLASGOW_OCULAR' THEN d.`numeric_value_clean` END) AS `GLASGOW_OCULAR`,
        MAX(CASE WHEN d.`variable_std` = 'GLASGOW_VERBAL' THEN d.`numeric_value_clean` END) AS `GLASGOW_VERBAL`,
        MAX(CASE WHEN d.`variable_std` = 'GLASGOW_MOTORA' THEN d.`numeric_value_clean` END) AS `GLASGOW_MOTORA`
    FROM daily_last_numeric d
    GROUP BY
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`
),

/* Keep days with at least one numeric vital sign or an active oxygen flag. */
all_days AS (
    SELECT `Episodi`, `Nhc`, `data_index`
    FROM daily_numeric_pivot
    UNION
    SELECT `Episodi`, `Nhc`, `data_index`
    FROM daily_porta_o2
    WHERE `porta_o2` = 1
)

SELECT
    ad.`Episodi`,
    ad.`Nhc`,
    ad.`data_index`,

    np.`SBP`,
    np.`DBP`,
    np.`TAM`,
    np.`HR`,
    np.`RESP`,
    np.`O2SAT`,
    np.`TEMP`,
    np.`FIO2`,
    np.`DIURESIS`,

    CASE
        WHEN COALESCE(np.`GLASGOW_OCULAR`, 0)
           + COALESCE(np.`GLASGOW_VERBAL`, 0)
           + COALESCE(np.`GLASGOW_MOTORA`, 0) > 0
        THEN COALESCE(np.`GLASGOW_OCULAR`, 0)
           + COALESCE(np.`GLASGOW_VERBAL`, 0)
           + COALESCE(np.`GLASGOW_MOTORA`, 0)
        ELSE NULL
    END AS `GLASGOW`,

    COALESCE(o2.`porta_o2`, 0) AS `porta_o2`

FROM all_days ad
LEFT JOIN daily_numeric_pivot np
    ON ad.`Episodi` = np.`Episodi`
   AND ad.`Nhc` = np.`Nhc`
   AND ad.`data_index` = np.`data_index`
LEFT JOIN daily_porta_o2 o2
    ON ad.`Episodi` = o2.`Episodi`
   AND ad.`Nhc` = o2.`Nhc`
   AND ad.`data_index` = o2.`data_index`;

CREATE UNIQUE INDEX idx_dvs_episodi_nhc_data_index
    ON daily_vital_signs (`Episodi`, `Nhc`, `data_index`);

CREATE INDEX idx_dvs_nhc_data_index
    ON daily_vital_signs (`Nhc`, `data_index`);

CREATE INDEX idx_dvs_data_index
    ON daily_vital_signs (`data_index`);
