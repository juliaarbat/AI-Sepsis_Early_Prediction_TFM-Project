/* ============================================================
   DAILY LABORATORY RECORDS

   Purpose:
   - Build daily_laboratory_records with one row per Episodi + Nhc + data_index.
   - Add daily laboratory values, microbiology results available by that day,
     and positive culture/resistance history in the previous 90 days.

   Strategy:
   - The process is split into materialized steps instead of one very large CTE,
     which improves runtime and makes intermediate tables auditable.
   - Created tables:
       1) laboratory_observed_days: days observed in daily_vital_signs.
       2) clean_laboratory_events: filtered and normalized laboratory events.
       3) daily_laboratory_aggregates: daily numeric and microbiology summaries.
       4) laboratory_model_days: vital-sign days plus laboratory days.
       5) historical_microbiology_events: patient-level microbiology history.
       6) prior_microbiology_90d: rolling 90-day microbiology history.
       7) daily_laboratory_records: final daily laboratory table.

   Operational rules:
   - Unit of analysis: one row per Episodi + Nhc + data_index.
   - Numeric values are aligned to DataPetició.
   - Microbiology is aligned to the date when the result becomes available.
   - The temporal universe includes days with vital signs and days with
     selected laboratory records, restricted to the hospital admission
     interval.
   - For each numeric variable, the last valid result of the day is kept.
   - Positive microbiology remains visible on later model days once available.
   - Prior microbiology uses events from the same patient in the previous
     90 days, excluding the current day.
   ============================================================ */


/* ------------------------------------------------------------
   0) RECOMMENDED SOURCE INDEXES

   Run only if these indexes do not already exist. They mainly support joins
   by patient and request/validation dates.
   ------------------------------------------------------------*/

CREATE INDEX idx_cohort_nhc_dates
    ON base_hospitalization_cohort (`Nhc`, `DataIniciUrgencies`, `DataIngres`);

CREATE INDEX idx_lab_patient_request
    ON tab_dt_sepsis_laboratori_001_ano (`PacientSAP`(32), `DataPetició`);

CREATE INDEX idx_lab_patient_validation
    ON tab_dt_sepsis_laboratori_001_ano (`PacientSAP`(32), `DataValidacióProva`);




/* ------------------------------------------------------------
   1) OBSERVED MODEL DAYS

   Creates the initial episode-day grid from daily_vital_signs. Laboratory-only
   days are added later, after laboratory events have been cleaned.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS laboratory_observed_days;

CREATE TABLE laboratory_observed_days AS
WITH
cohort_base AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        b.`DataIngres`,
        b.`DataAlta`
    FROM base_hospitalization_cohort b
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND b.`DataIngres` IS NOT NULL
)
SELECT
    b.`Episodi`,
    b.`Nhc`,
    cv.`data_index`,
    DATEDIFF(cv.`data_index`, DATE(b.`DataIngres`)) AS `dia_relatiu`
FROM cohort_base b
INNER JOIN daily_vital_signs cv
    ON b.`Episodi` = cv.`Episodi`
   AND b.`Nhc` = cv.`Nhc`
WHERE cv.`data_index` IS NOT NULL
  AND cv.`data_index` >= DATE(b.`DataIngres`)
  AND (
        b.`DataAlta` IS NULL
     OR cv.`data_index` <= DATE(b.`DataAlta`)
  )
GROUP BY
    b.`Episodi`,
    b.`Nhc`,
    cv.`data_index`,
    DATEDIFF(cv.`data_index`, DATE(b.`DataIngres`));

CREATE INDEX idx_lod_epi_data
    ON laboratory_observed_days (`Episodi`, `data_index`);

CREATE INDEX idx_lod_nhc_data
    ON laboratory_observed_days (`Nhc`, `data_index`);


/* ------------------------------------------------------------
   2) CLEAN LABORATORY EVENTS

   Joins the base cohort with the raw laboratory table, keeps only selected
   numeric tests and microbiology records, standardizes numeric test codes,
   applies basic plausibility ranges, and creates simple microbiology flags.
   Materializing this step reduces the volume of the raw laboratory table.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS clean_laboratory_events;

CREATE TABLE clean_laboratory_events AS
WITH
cohort_base AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        b.`DataIngres`,
        b.`DataIniciUrgencies`,
        b.`DataAlta`
    FROM base_hospitalization_cohort b
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND b.`DataIngres` IS NOT NULL
),

/* Single mapping of numeric test codes to final laboratory variables. */
map_laboratory_numeric AS (
    SELECT 2640 AS `prova_codi`, 'ph_arterial' AS `variable_lab`, CAST(6.5 AS DECIMAL(18,4)) AS `valor_min`, CAST(7.8 AS DECIMAL(18,4)) AS `valor_max`, 0 AS `exclou_zero`
    UNION ALL SELECT 2642, 'pao2_arterial', NULL, NULL, 0
    UNION ALL SELECT 2641, 'paco2_arterial', NULL, NULL, 0
    UNION ALL SELECT 2643, 'bicarbonat_arterial', NULL, NULL, 0
    UNION ALL SELECT 2645, 'exc_base_arterial', NULL, NULL, 0
    UNION ALL SELECT 7336, 'lactat_arterial', NULL, CAST(200 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2653, 'ph_venos', CAST(6.5 AS DECIMAL(18,4)), CAST(7.8 AS DECIMAL(18,4)), 0
    UNION ALL SELECT 2655, 'pao2_venos', NULL, NULL, 0
    UNION ALL SELECT 2654, 'paco2_venos', NULL, NULL, 0
    UNION ALL SELECT 2656, 'bicarbonat_venos', NULL, NULL, 0
    UNION ALL SELECT 2658, 'exc_base_venos', NULL, NULL, 0
    UNION ALL SELECT 7339, 'lactat_venos', NULL, CAST(200 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2418, 'hematocrit', NULL, NULL, 0
    UNION ALL SELECT 2419, 'hemoglobina', NULL, CAST(25 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2428, 'leucocits', NULL, CAST(200 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2429, 'pct_neutrofils', NULL, NULL, 0
    UNION ALL SELECT 2439, 'granulocits_immadurs', NULL, NULL, 0
    UNION ALL SELECT 2424, 'plaquetes', NULL, CAST(2000 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2470, 'fibrinogen', NULL, NULL, 0
    UNION ALL SELECT 2460, 'temps_protrombina_pct', NULL, NULL, 0
    UNION ALL SELECT 2691, 'pcr', NULL, NULL, 0
    UNION ALL SELECT 2693, 'procalcitonina', NULL, NULL, 0
    UNION ALL SELECT 2515, 'glucosa', NULL, NULL, 0
    UNION ALL SELECT 2516, 'urea', NULL, NULL, 0
    UNION ALL SELECT 2517, 'creatinina', NULL, CAST(20 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2526, 'bilirubina_total', NULL, CAST(40 AS DECIMAL(18,4)), 1
    UNION ALL SELECT 2529, 'got_ast', NULL, NULL, 0
    UNION ALL SELECT 9218, 'albumina', NULL, NULL, 0
    UNION ALL SELECT 2519, 'proteines_totals', NULL, NULL, 0
    UNION ALL SELECT 2689, 'troponina', NULL, NULL, 0
),

lab_raw AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        l.`Id`,
        l.`DataPetició`,
        DATE(l.`DataPetició`) AS `data_index`,
        COALESCE(l.`DataResultatProva`, l.`DataValidacióProva`, l.`DataPetició`) AS `ordre_resultat`,
        CAST(l.`ProvaCodi` AS UNSIGNED) AS `ProvaCodi`,
        CAST(l.`ContenidorCodi` AS UNSIGNED) AS `ContenidorCodi`,
        l.`ResultatNumèric`,
        l.`PrimerResultatCodificatDescripció`,
        l.`DataCreacióProva`,
        l.`DataResultatProva`,
        l.`DataValidacióProva`,
        l.`MecanismeResistència1Codi`,
        l.`MecanismeResistència2Codi`,
        l.`MecanismeResistència3Codi`,
        l.`MecanismeResistència4Codi`,
        l.`MecanismeResistència5Codi`
    FROM cohort_base b
    INNER JOIN tab_dt_sepsis_laboratori_001_ano l
        ON b.`Nhc` = l.`PacientSAP`
    WHERE l.`DataPetició` IS NOT NULL
      AND l.`DataPetició` >= COALESCE(b.`DataIniciUrgencies`, b.`DataIngres`)
      AND (
            b.`DataAlta` IS NULL
         OR DATE(l.`DataPetició`) <= DATE(b.`DataAlta`)
      )
      AND (
            CAST(l.`ProvaCodi` AS UNSIGNED) IN (
                2640,2642,2641,2643,2645,7336,
                2653,2655,2654,2656,2658,7339,
                2418,2419,2428,2429,2439,2424,
                2470,2460,2691,2693,2515,2516,2517,
                2526,2529,9218,2519,2689,
                4146,4151,7685,4167
            )
            OR CAST(l.`ContenidorCodi` AS UNSIGNED) IN (
                59,69,653,
                483,484,487,495,496,497,499,500,501,579,602,603,651
            )
          )
)
SELECT
    l.`Episodi`,
    l.`Nhc`,
    l.`Id`,
    l.`DataPetició`,
    l.`data_index`,
    l.`ordre_resultat`,
    l.`ProvaCodi`,
    l.`ContenidorCodi`,
    l.`ResultatNumèric`,
    /* Clean numeric value used by the daily pivot. The original source value
       is kept for traceability, while impossible or clearly wrong values are
       excluded from ResultatNumericClean. */
    CASE
        WHEN m.`variable_lab` IS NULL
          OR l.`ResultatNumèric` IS NULL
        THEN NULL

        WHEN COALESCE(m.`exclou_zero`, 0) = 1
         AND l.`ResultatNumèric` <= 0
        THEN NULL

        WHEN m.`valor_min` IS NOT NULL
         AND l.`ResultatNumèric` < m.`valor_min`
        THEN NULL

        WHEN m.`valor_max` IS NOT NULL
         AND l.`ResultatNumèric` > m.`valor_max`
        THEN NULL

        ELSE l.`ResultatNumèric`
    END AS `ResultatNumericClean`,
    l.`PrimerResultatCodificatDescripció`,
    l.`DataCreacióProva`,
    l.`DataResultatProva`,
    l.`DataValidacióProva`,
    l.`MecanismeResistència1Codi`,
    l.`MecanismeResistència2Codi`,
    l.`MecanismeResistència3Codi`,
    l.`MecanismeResistència4Codi`,
    l.`MecanismeResistència5Codi`,

    /* Final numeric variable name. NULL means the row is microbiology only. */
    m.`variable_lab`,

    CASE
        WHEN l.`ProvaCodi` = 4146
          OR l.`ContenidorCodi` IN (483,484,487,495,496,497,499,500,501,579,602,603,651)
        THEN 1 ELSE 0
    END AS `es_hemocultiu`,

    CASE
        WHEN 'BLEE' IN (
            COALESCE(l.`MecanismeResistència1Codi`, ''),
            COALESCE(l.`MecanismeResistència2Codi`, ''),
            COALESCE(l.`MecanismeResistència3Codi`, ''),
            COALESCE(l.`MecanismeResistència4Codi`, ''),
            COALESCE(l.`MecanismeResistència5Codi`, '')
        )
        THEN 1 ELSE 0
    END AS `blee_flag`,

    CASE
        WHEN 'CARB' IN (
            COALESCE(l.`MecanismeResistència1Codi`, ''),
            COALESCE(l.`MecanismeResistència2Codi`, ''),
            COALESCE(l.`MecanismeResistència3Codi`, ''),
            COALESCE(l.`MecanismeResistència4Codi`, ''),
            COALESCE(l.`MecanismeResistència5Codi`, '')
        )
        THEN 1 ELSE 0
    END AS `cre_flag`,

    CASE
        WHEN 'MRSA' IN (
            COALESCE(l.`MecanismeResistència1Codi`, ''),
            COALESCE(l.`MecanismeResistència2Codi`, ''),
            COALESCE(l.`MecanismeResistència3Codi`, ''),
            COALESCE(l.`MecanismeResistència4Codi`, ''),
            COALESCE(l.`MecanismeResistència5Codi`, '')
        )
        THEN 1 ELSE 0
    END AS `mrsa_flag`,

    CASE
        WHEN 'ERV' IN (
            COALESCE(l.`MecanismeResistència1Codi`, ''),
            COALESCE(l.`MecanismeResistència2Codi`, ''),
            COALESCE(l.`MecanismeResistència3Codi`, ''),
            COALESCE(l.`MecanismeResistència4Codi`, ''),
            COALESCE(l.`MecanismeResistència5Codi`, '')
        )
        THEN 1 ELSE 0
    END AS `vre_flag`
FROM lab_raw l
LEFT JOIN map_laboratory_numeric m
    ON l.`ProvaCodi` = m.`prova_codi`;

CREATE INDEX idx_cle_epi_data_variable
    ON clean_laboratory_events (`Episodi`, `data_index`, `variable_lab`, `ordre_resultat`, `DataPetició`, `Id`);

CREATE INDEX idx_cle_epi_nhc_data
    ON clean_laboratory_events (`Episodi`, `Nhc`, `data_index`);

CREATE INDEX idx_cle_nhc_order
    ON clean_laboratory_events (`Nhc`, `ordre_resultat`);


/* ------------------------------------------------------------
   3) DAILY LABORATORY AGGREGATES

   Selects the last valid numeric result per day and variable, pivots numeric
   tests into final columns, and adds microbiology results only after they are
   operationally available. This prevents temporal leakage from future results.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS daily_laboratory_aggregates;

CREATE TABLE daily_laboratory_aggregates AS
WITH numeric_rank AS (
    SELECT
        l.`Episodi`,
        l.`Nhc`,
        l.`data_index`,
        l.`variable_lab`,
        l.`ResultatNumericClean`,
        ROW_NUMBER() OVER (
            PARTITION BY l.`Episodi`, l.`data_index`, l.`variable_lab`
            ORDER BY l.`ordre_resultat` DESC, l.`DataPetició` DESC, l.`Id` DESC
        ) AS rn
    FROM clean_laboratory_events l
    WHERE l.`variable_lab` IS NOT NULL
      AND l.`ResultatNumericClean` IS NOT NULL
),
numeric_pivot AS (
    SELECT
        n.`Episodi`,
        n.`Nhc`,
        n.`data_index`,
        MAX(CASE WHEN n.`variable_lab` = 'ph_arterial' THEN n.`ResultatNumericClean` END) AS `ph_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'pao2_arterial' THEN n.`ResultatNumericClean` END) AS `pao2_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'paco2_arterial' THEN n.`ResultatNumericClean` END) AS `paco2_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'bicarbonat_arterial' THEN n.`ResultatNumericClean` END) AS `bicarbonat_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'exc_base_arterial' THEN n.`ResultatNumericClean` END) AS `exc_base_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'lactat_arterial' THEN n.`ResultatNumericClean` END) AS `lactat_arterial`,
        MAX(CASE WHEN n.`variable_lab` = 'ph_venos' THEN n.`ResultatNumericClean` END) AS `ph_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'pao2_venos' THEN n.`ResultatNumericClean` END) AS `pao2_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'paco2_venos' THEN n.`ResultatNumericClean` END) AS `paco2_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'bicarbonat_venos' THEN n.`ResultatNumericClean` END) AS `bicarbonat_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'exc_base_venos' THEN n.`ResultatNumericClean` END) AS `exc_base_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'lactat_venos' THEN n.`ResultatNumericClean` END) AS `lactat_venos`,
        MAX(CASE WHEN n.`variable_lab` = 'hematocrit' THEN n.`ResultatNumericClean` END) AS `hematocrit`,
        MAX(CASE WHEN n.`variable_lab` = 'hemoglobina' THEN n.`ResultatNumericClean` END) AS `hemoglobina`,
        MAX(CASE WHEN n.`variable_lab` = 'leucocits' THEN n.`ResultatNumericClean` END) AS `leucocits`,
        MAX(CASE WHEN n.`variable_lab` = 'pct_neutrofils' THEN n.`ResultatNumericClean` END) AS `pct_neutrofils`,
        MAX(CASE WHEN n.`variable_lab` = 'granulocits_immadurs' THEN n.`ResultatNumericClean` END) AS `granulocits_immadurs`,
        MAX(CASE WHEN n.`variable_lab` = 'plaquetes' THEN n.`ResultatNumericClean` END) AS `plaquetes`,
        MAX(CASE WHEN n.`variable_lab` = 'fibrinogen' THEN n.`ResultatNumericClean` END) AS `fibrinogen`,
        MAX(CASE WHEN n.`variable_lab` = 'temps_protrombina_pct' THEN n.`ResultatNumericClean` END) AS `temps_protrombina_pct`,
        MAX(CASE WHEN n.`variable_lab` = 'pcr' THEN n.`ResultatNumericClean` END) AS `pcr`,
        MAX(CASE WHEN n.`variable_lab` = 'procalcitonina' THEN n.`ResultatNumericClean` END) AS `procalcitonina`,
        MAX(CASE WHEN n.`variable_lab` = 'glucosa' THEN n.`ResultatNumericClean` END) AS `glucosa`,
        MAX(CASE WHEN n.`variable_lab` = 'urea' THEN n.`ResultatNumericClean` END) AS `urea`,
        MAX(CASE WHEN n.`variable_lab` = 'creatinina' THEN n.`ResultatNumericClean` END) AS `creatinina`,
        MAX(CASE WHEN n.`variable_lab` = 'bilirubina_total' THEN n.`ResultatNumericClean` END) AS `bilirubina_total`,
        MAX(CASE WHEN n.`variable_lab` = 'got_ast' THEN n.`ResultatNumericClean` END) AS `got_ast`,
        MAX(CASE WHEN n.`variable_lab` = 'albumina' THEN n.`ResultatNumericClean` END) AS `albumina`,
        MAX(CASE WHEN n.`variable_lab` = 'proteines_totals' THEN n.`ResultatNumericClean` END) AS `proteines_totals`,
        MAX(CASE WHEN n.`variable_lab` = 'troponina' THEN n.`ResultatNumericClean` END) AS `troponina`
    FROM numeric_rank n
    WHERE n.rn = 1
    GROUP BY
        n.`Episodi`,
        n.`Nhc`,
        n.`data_index`
),
micro_events AS (
    /* Positive blood culture: available only when positivity is known. */
    SELECT
        l.`Episodi`,
        l.`Nhc`,
        DATE(
            DATE_ADD(
                l.`DataCreacióProva`,
                INTERVAL TIMESTAMPDIFF(HOUR, l.`DataCreacióProva`, l.`DataResultatProva`) HOUR
            )
        ) AS `data_disponibilitat`,
        1 AS `hemocultiu_positiu`,
        l.`DataCreacióProva` AS `hemocultiu_positiu_data_extraccio`,
        NULL AS `hemocultiu_germen`,
        TIMESTAMPDIFF(HOUR, l.`DataCreacióProva`, l.`DataResultatProva`) AS `hemocultiu_temps_positivitat_h`,
        NULL AS `urocultiu_resultat`,
        NULL AS `aspirat_traqueal_germen`,
        NULL AS `broncoaspirat_germen`,
        NULL AS `bal_germen`,
        0 AS `ag_pneumococ`,
        0 AS `ag_legionella`
    FROM clean_laboratory_events l
    WHERE l.`es_hemocultiu` = 1
      AND l.`PrimerResultatCodificatDescripció` = 'P'
      AND l.`DataCreacióProva` IS NOT NULL
      AND l.`DataResultatProva` IS NOT NULL
      AND l.`DataResultatProva` >= l.`DataCreacióProva`

    UNION ALL

    /* Organism identification: available at result/validation time, not extraction time. */
    SELECT
        l.`Episodi`,
        l.`Nhc`,
        DATE(COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`)) AS `data_disponibilitat`,
        0 AS `hemocultiu_positiu`,
        NULL AS `hemocultiu_positiu_data_extraccio`,
        l.`PrimerResultatCodificatDescripció` AS `hemocultiu_germen`,
        NULL AS `hemocultiu_temps_positivitat_h`,
        NULL AS `urocultiu_resultat`,
        NULL AS `aspirat_traqueal_germen`,
        NULL AS `broncoaspirat_germen`,
        NULL AS `bal_germen`,
        0 AS `ag_pneumococ`,
        0 AS `ag_legionella`
    FROM clean_laboratory_events l
    WHERE l.`ContenidorCodi` IN (483,484,487,495,496,497,499,500,501,579,602,603,651)
      AND COALESCE(l.`PrimerResultatCodificatDescripció`, '') NOT IN ('', 'N', 'P', '48N')
      AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`) IS NOT NULL

    UNION ALL

    SELECT
        l.`Episodi`,
        l.`Nhc`,
        DATE(COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`)) AS `data_disponibilitat`,
        0 AS `hemocultiu_positiu`,
        NULL AS `hemocultiu_positiu_data_extraccio`,
        NULL AS `hemocultiu_germen`,
        NULL AS `hemocultiu_temps_positivitat_h`,
        l.`PrimerResultatCodificatDescripció` AS `urocultiu_resultat`,
        NULL AS `aspirat_traqueal_germen`,
        NULL AS `broncoaspirat_germen`,
        NULL AS `bal_germen`,
        0 AS `ag_pneumococ`,
        0 AS `ag_legionella`
    FROM clean_laboratory_events l
    WHERE l.`ProvaCodi` = 4151
      AND COALESCE(l.`PrimerResultatCodificatDescripció`, '') <> ''
      AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`) IS NOT NULL

    UNION ALL

    SELECT
        l.`Episodi`,
        l.`Nhc`,
        DATE(COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`)) AS `data_disponibilitat`,
        0 AS `hemocultiu_positiu`,
        NULL AS `hemocultiu_positiu_data_extraccio`,
        NULL AS `hemocultiu_germen`,
        NULL AS `hemocultiu_temps_positivitat_h`,
        NULL AS `urocultiu_resultat`,
        CASE WHEN l.`ContenidorCodi` = 69 THEN l.`PrimerResultatCodificatDescripció` END AS `aspirat_traqueal_germen`,
        CASE WHEN l.`ContenidorCodi` = 59 THEN l.`PrimerResultatCodificatDescripció` END AS `broncoaspirat_germen`,
        CASE WHEN l.`ContenidorCodi` = 653 THEN l.`PrimerResultatCodificatDescripció` END AS `bal_germen`,
        0 AS `ag_pneumococ`,
        0 AS `ag_legionella`
    FROM clean_laboratory_events l
    WHERE l.`ContenidorCodi` IN (59,69,653)
      AND COALESCE(l.`PrimerResultatCodificatDescripció`, '') NOT IN ('', 'N', 'P')
      AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`) IS NOT NULL

    UNION ALL

    SELECT
        l.`Episodi`,
        l.`Nhc`,
        DATE(COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`)) AS `data_disponibilitat`,
        0 AS `hemocultiu_positiu`,
        NULL AS `hemocultiu_positiu_data_extraccio`,
        NULL AS `hemocultiu_germen`,
        NULL AS `hemocultiu_temps_positivitat_h`,
        NULL AS `urocultiu_resultat`,
        NULL AS `aspirat_traqueal_germen`,
        NULL AS `broncoaspirat_germen`,
        NULL AS `bal_germen`,
        CASE WHEN l.`ProvaCodi` = 7685 THEN 1 ELSE 0 END AS `ag_pneumococ`,
        CASE WHEN l.`ProvaCodi` = 4167 THEN 1 ELSE 0 END AS `ag_legionella`
    FROM clean_laboratory_events l
    WHERE l.`ProvaCodi` IN (7685,4167)
      AND l.`PrimerResultatCodificatDescripció` = 'P'
      AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`) IS NOT NULL
),
aggregate_days AS (
    SELECT `Episodi`, `Nhc`, `data_index`
    FROM laboratory_observed_days
    UNION
    SELECT `Episodi`, `Nhc`, `data_index`
    FROM numeric_pivot
    UNION
    SELECT `Episodi`, `Nhc`, `data_disponibilitat` AS `data_index`
    FROM micro_events
    WHERE `data_disponibilitat` IS NOT NULL
),
micro_daily AS (
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`,
        MAX(CASE WHEN e.`hemocultiu_positiu` = 1 THEN 1 ELSE 0 END) AS `hemocultiu_positiu`,
        MIN(e.`hemocultiu_positiu_data_extraccio`) AS `hemocultiu_positiu_data_extraccio`,
        GROUP_CONCAT(DISTINCT e.`hemocultiu_germen` SEPARATOR ' | ') AS `hemocultiu_germen`,
        MIN(e.`hemocultiu_temps_positivitat_h`) AS `hemocultiu_temps_positivitat_h`,
        GROUP_CONCAT(DISTINCT e.`urocultiu_resultat` SEPARATOR ' | ') AS `urocultiu_resultat`,
        GROUP_CONCAT(DISTINCT e.`aspirat_traqueal_germen` SEPARATOR ' | ') AS `aspirat_traqueal_germen`,
        GROUP_CONCAT(DISTINCT e.`broncoaspirat_germen` SEPARATOR ' | ') AS `broncoaspirat_germen`,
        GROUP_CONCAT(DISTINCT e.`bal_germen` SEPARATOR ' | ') AS `bal_germen`,
        MAX(CASE WHEN e.`ag_pneumococ` = 1 THEN 1 ELSE 0 END) AS `ag_pneumococ`,
        MAX(CASE WHEN e.`ag_legionella` = 1 THEN 1 ELSE 0 END) AS `ag_legionella`
    FROM aggregate_days d
    LEFT JOIN micro_events e
        ON d.`Episodi` = e.`Episodi`
       AND d.`Nhc` = e.`Nhc`
       AND e.`data_disponibilitat` <= d.`data_index`
    GROUP BY
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`
    HAVING `hemocultiu_positiu` = 1
        OR `hemocultiu_germen` IS NOT NULL
        OR `urocultiu_resultat` IS NOT NULL
        OR `aspirat_traqueal_germen` IS NOT NULL
        OR `broncoaspirat_germen` IS NOT NULL
        OR `bal_germen` IS NOT NULL
        OR `ag_pneumococ` = 1
        OR `ag_legionella` = 1
),
days_with_laboratory_data AS (
    SELECT `Episodi`, `Nhc`, `data_index` FROM numeric_pivot
    UNION
    SELECT `Episodi`, `Nhc`, `data_index` FROM micro_daily
)
SELECT
    k.`Episodi`,
    k.`Nhc`,
    k.`data_index`,
    n.`ph_arterial`,
    n.`pao2_arterial`,
    n.`paco2_arterial`,
    n.`bicarbonat_arterial`,
    n.`exc_base_arterial`,
    n.`lactat_arterial`,
    n.`ph_venos`,
    n.`pao2_venos`,
    n.`paco2_venos`,
    n.`bicarbonat_venos`,
    n.`exc_base_venos`,
    n.`lactat_venos`,
    n.`hematocrit`,
    n.`hemoglobina`,
    n.`leucocits`,
    n.`pct_neutrofils`,
    n.`granulocits_immadurs`,
    n.`plaquetes`,
    n.`fibrinogen`,
    n.`temps_protrombina_pct`,
    n.`pcr`,
    n.`procalcitonina`,
    n.`glucosa`,
    n.`urea`,
    n.`creatinina`,
    n.`bilirubina_total`,
    n.`got_ast`,
    n.`albumina`,
    n.`proteines_totals`,
    n.`troponina`,
    m.`hemocultiu_positiu`,
    m.`hemocultiu_positiu_data_extraccio`,
    m.`hemocultiu_germen`,
    m.`hemocultiu_temps_positivitat_h`,
    m.`urocultiu_resultat`,
    m.`aspirat_traqueal_germen`,
    m.`broncoaspirat_germen`,
    m.`bal_germen`,
    m.`ag_pneumococ`,
    m.`ag_legionella`
FROM days_with_laboratory_data k
LEFT JOIN numeric_pivot n
    ON k.`Episodi` = n.`Episodi`
   AND k.`Nhc` = n.`Nhc`
   AND k.`data_index` = n.`data_index`
LEFT JOIN micro_daily m
    ON k.`Episodi` = m.`Episodi`
   AND k.`Nhc` = m.`Nhc`
   AND k.`data_index` = m.`data_index`;

CREATE INDEX idx_dla_epi_data
    ON daily_laboratory_aggregates (`Episodi`, `data_index`);


/* ------------------------------------------------------------
   4) LABORATORY MODEL DAYS

   Expands the daily calendar to include both vital-sign days and days with
   selected laboratory or microbiology information within the hospital
   admission interval. Downstream Python preprocessing applies additional
   temporal quality controls and carry-forward rules.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS laboratory_model_days;

CREATE TABLE laboratory_model_days AS
SELECT
    d.`Episodi`,
    d.`Nhc`,
    d.`data_index`,
    d.`dia_relatiu`
FROM laboratory_observed_days d

UNION

SELECT
    a.`Episodi`,
    a.`Nhc`,
    a.`data_index`,
    DATEDIFF(a.`data_index`, DATE(b.`DataIngres`)) AS `dia_relatiu`
FROM daily_laboratory_aggregates a
INNER JOIN base_hospitalization_cohort b
    ON a.`Episodi` = b.`Episodi`
   AND a.`Nhc` = b.`Nhc`
WHERE a.`data_index` IS NOT NULL
  AND a.`data_index` >= DATE(b.`DataIngres`)
  AND (
        b.`DataAlta` IS NULL
     OR a.`data_index` <= DATE(b.`DataAlta`)
  );

CREATE UNIQUE INDEX idx_lmd_epi_nhc_data
    ON laboratory_model_days (`Episodi`, `Nhc`, `data_index`);

CREATE INDEX idx_lmd_nhc_data
    ON laboratory_model_days (`Nhc`, `data_index`);


/* ------------------------------------------------------------
   5) PRIOR MICROBIOLOGY HISTORY

   Builds a patient-level microbiology event table and then computes whether
   the same patient had positive cultures or resistance markers in the previous
   90 days. This is materialized because rolling date-window joins are costly.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS historical_microbiology_events;

CREATE TABLE historical_microbiology_events AS
SELECT
    d.`Nhc`,
    COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`, l.`DataPetició`) AS `data_event_micro`,
    CASE
        WHEN (CAST(l.`ProvaCodi` AS UNSIGNED) = 4146 OR CAST(l.`ContenidorCodi` AS UNSIGNED) IN (483,484,487,495,496,497,499,500,501,579,602,603,651))
         AND l.`PrimerResultatCodificatDescripció` = 'P'
        THEN 1
        WHEN CAST(l.`ProvaCodi` AS UNSIGNED) = 4151
         AND COALESCE(l.`PrimerResultatCodificatDescripció`, '') NOT IN ('', 'N', 'P')
        THEN 1
        WHEN CAST(l.`ContenidorCodi` AS UNSIGNED) IN (59,69,653)
         AND COALESCE(l.`PrimerResultatCodificatDescripció`, '') NOT IN ('', 'N', 'P')
        THEN 1
        ELSE 0
    END AS `cultiu_positiu_flag`,
    CASE WHEN 'BLEE' IN (COALESCE(l.`MecanismeResistència1Codi`, ''), COALESCE(l.`MecanismeResistència2Codi`, ''), COALESCE(l.`MecanismeResistència3Codi`, ''), COALESCE(l.`MecanismeResistència4Codi`, ''), COALESCE(l.`MecanismeResistència5Codi`, '')) THEN 1 ELSE 0 END AS `blee_flag`,
    CASE WHEN 'CARB' IN (COALESCE(l.`MecanismeResistència1Codi`, ''), COALESCE(l.`MecanismeResistència2Codi`, ''), COALESCE(l.`MecanismeResistència3Codi`, ''), COALESCE(l.`MecanismeResistència4Codi`, ''), COALESCE(l.`MecanismeResistència5Codi`, '')) THEN 1 ELSE 0 END AS `cre_flag`,
    CASE WHEN 'MRSA' IN (COALESCE(l.`MecanismeResistència1Codi`, ''), COALESCE(l.`MecanismeResistència2Codi`, ''), COALESCE(l.`MecanismeResistència3Codi`, ''), COALESCE(l.`MecanismeResistència4Codi`, ''), COALESCE(l.`MecanismeResistència5Codi`, '')) THEN 1 ELSE 0 END AS `mrsa_flag`,
    CASE WHEN 'ERV' IN (COALESCE(l.`MecanismeResistència1Codi`, ''), COALESCE(l.`MecanismeResistència2Codi`, ''), COALESCE(l.`MecanismeResistència3Codi`, ''), COALESCE(l.`MecanismeResistència4Codi`, ''), COALESCE(l.`MecanismeResistència5Codi`, '')) THEN 1 ELSE 0 END AS `vre_flag`
FROM (
    SELECT
        `Nhc`,
        MIN(`data_index`) AS `min_data_index`,
        MAX(`data_index`) AS `max_data_index`
    FROM laboratory_model_days
    GROUP BY `Nhc`
) d
INNER JOIN tab_dt_sepsis_laboratori_001_ano l
    ON d.`Nhc` = l.`PacientSAP`
WHERE COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`, l.`DataPetició`) IS NOT NULL
  AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`, l.`DataPetició`) >= DATE_SUB(d.`min_data_index`, INTERVAL 90 DAY)
  AND COALESCE(l.`DataValidacióProva`, l.`DataResultatProva`, l.`DataPetició`) <= d.`max_data_index`
  AND (
        CAST(l.`ProvaCodi` AS UNSIGNED) IN (4146,4151)
        OR CAST(l.`ContenidorCodi` AS UNSIGNED) IN (
            59,69,653,
            483,484,487,495,496,497,499,500,501,579,602,603,651
        )
        OR COALESCE(l.`MecanismeResistència1Codi`, '') IN ('BLEE','CARB','MRSA','ERV')
        OR COALESCE(l.`MecanismeResistència2Codi`, '') IN ('BLEE','CARB','MRSA','ERV')
        OR COALESCE(l.`MecanismeResistència3Codi`, '') IN ('BLEE','CARB','MRSA','ERV')
        OR COALESCE(l.`MecanismeResistència4Codi`, '') IN ('BLEE','CARB','MRSA','ERV')
        OR COALESCE(l.`MecanismeResistència5Codi`, '') IN ('BLEE','CARB','MRSA','ERV')
      );

CREATE INDEX idx_hme_nhc_event
    ON historical_microbiology_events (`Nhc`, `data_event_micro`);


DROP TABLE IF EXISTS prior_microbiology_90d;

CREATE TABLE prior_microbiology_90d AS
SELECT
    d.`Episodi`,
    d.`data_index`,
    MAX(CASE WHEN h.`cultiu_positiu_flag` = 1 THEN 1 ELSE 0 END) AS `cultiu_positiu_previ_90d`,
    MAX(CASE WHEN h.`blee_flag` = 1 THEN 1 ELSE 0 END) AS `colonitzacio_previa_blee`,
    MAX(CASE WHEN h.`cre_flag` = 1 THEN 1 ELSE 0 END) AS `colonitzacio_previa_cre`,
    MAX(CASE WHEN h.`mrsa_flag` = 1 THEN 1 ELSE 0 END) AS `colonitzacio_previa_mrsa`,
    MAX(CASE WHEN h.`vre_flag` = 1 THEN 1 ELSE 0 END) AS `colonitzacio_previa_vre`
FROM laboratory_model_days d
LEFT JOIN historical_microbiology_events h
    ON h.`Nhc` = d.`Nhc`
   AND h.`data_event_micro` < d.`data_index`
   AND h.`data_event_micro` >= DATE_SUB(d.`data_index`, INTERVAL 90 DAY)
GROUP BY
    d.`Episodi`,
    d.`data_index`;

CREATE INDEX idx_pm90_epi_data
    ON prior_microbiology_90d (`Episodi`, `data_index`);


/* ------------------------------------------------------------
   6) FINAL DAILY LABORATORY TABLE

   Starts from all vital-sign or laboratory model days, adds same-day
   laboratory aggregates, adds prior 90-day microbiology history, and sets
   missing binary flags to 0.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS daily_laboratory_records;

CREATE TABLE daily_laboratory_records AS
SELECT
    d.`Episodi`,
    d.`Nhc`,
    d.`data_index`,

    a.`ph_arterial`,
    a.`pao2_arterial`,
    a.`paco2_arterial`,
    a.`bicarbonat_arterial`,
    a.`exc_base_arterial`,
    a.`lactat_arterial`,
    a.`ph_venos`,
    a.`pao2_venos`,
    a.`paco2_venos`,
    a.`bicarbonat_venos`,
    a.`exc_base_venos`,
    a.`lactat_venos`,
    a.`hematocrit`,
    a.`hemoglobina`,
    a.`leucocits`,
    a.`pct_neutrofils`,
    a.`granulocits_immadurs`,
    a.`plaquetes`,
    a.`fibrinogen`,
    a.`temps_protrombina_pct`,
    a.`pcr`,
    a.`procalcitonina`,
    a.`glucosa`,
    a.`urea`,
    a.`creatinina`,
    a.`bilirubina_total`,
    a.`got_ast`,
    a.`albumina`,
    a.`proteines_totals`,
    a.`troponina`,

    COALESCE(a.`hemocultiu_positiu`, 0) AS `hemocultiu_positiu`,
    a.`hemocultiu_positiu_data_extraccio`,
    a.`hemocultiu_germen`,
    a.`hemocultiu_temps_positivitat_h`,
    a.`urocultiu_resultat`,
    a.`aspirat_traqueal_germen`,
    a.`broncoaspirat_germen`,
    a.`bal_germen`,
    COALESCE(a.`ag_pneumococ`, 0) AS `ag_pneumococ`,
    COALESCE(a.`ag_legionella`, 0) AS `ag_legionella`,
    COALESCE(p.`colonitzacio_previa_blee`, 0) AS `colonitzacio_previa_blee`,
    COALESCE(p.`colonitzacio_previa_cre`, 0) AS `colonitzacio_previa_cre`,
    COALESCE(p.`colonitzacio_previa_mrsa`, 0) AS `colonitzacio_previa_mrsa`,
    COALESCE(p.`colonitzacio_previa_vre`, 0) AS `colonitzacio_previa_vre`,
    COALESCE(p.`cultiu_positiu_previ_90d`, 0) AS `cultiu_positiu_previ_90d`
FROM laboratory_model_days d
LEFT JOIN daily_laboratory_aggregates a
    ON d.`Episodi` = a.`Episodi`
   AND d.`data_index` = a.`data_index`
LEFT JOIN prior_microbiology_90d p
    ON d.`Episodi` = p.`Episodi`
   AND d.`data_index` = p.`data_index`;

CREATE INDEX idx_dlr_epi_data
    ON daily_laboratory_records (`Episodi`, `data_index`);

CREATE UNIQUE INDEX idx_dlr_epi_nhc_data
    ON daily_laboratory_records (`Episodi`, `Nhc`, `data_index`);

CREATE INDEX idx_dlr_nhc_data
    ON daily_laboratory_records (`Nhc`, `data_index`);
