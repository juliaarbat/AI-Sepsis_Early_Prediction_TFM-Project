/* =========================================================
   HOSPITALIZATION COHORT AND COMORBIDITIES

  Operational summary
   - Unit of analysis: one adult hospital admission episode.
   - DataIngres: the date on which the hospital admission was administratively started.
   - DataIniciUrgencies: start date of the linked emergency episode, when
     available and occurring on or before DataIngres; otherwise, DataIngres
     is retained.
   - DataAlta: last valid discharge date recorded in the episode movements,
     when available.
   - Initial critical care admission: episodes whose first hospital movement
     is to a critical-care unit are excluded from the base cohort.
   - Initial sepsis: episodes with an emergency or admission diagnosis
     compatible with sepsis are excluded, since the cohort is designed to
     study in-hospital sepsis risk.
   - Comorbidities: patient-level flags derived from available diagnosis and
     procedure codes. Procedures that define a condition, such as splenectomy,
     are treated directly as comorbidity evidence.
   ========================================================= */

/* ---------------------------------------------------------
   RECOMMENDED INDEXES ON SOURCE TABLES
   Run these statements only once, if the indexes do not already exist.
   Recreating indexes on every execution can add several minutes to the
   total runtime.
   --------------------------------------------------------- */

CREATE INDEX idx_hosp_episodi_dataingres
    ON tab_dt_sepsis_hosp_001_ano (`Episodi`, `DataIngres`);

CREATE INDEX idx_passosserveis_episodi_datainici_dataalta
    ON tab_dt_hosp_passos_serveis_001_ano (`Episodi`, `DataIniciMoviment`, `DataAlta`);

CREATE INDEX idx_activitatquirurgica_episodi_dataiq
    ON tab_dt_activitat_quirurgica_001_ano (`Episodi`, `DataHoraIQ`);

CREATE INDEX idx_urg_episodi_post_data
    ON tab_dt_sepsis_urgencies_001_ano (`EpisodiHospitalitzacióPosterior`, `DataEpisodiHospitalitzacióPosterior`);

CREATE INDEX idx_diag_historia
    ON tab_dt_sepsis_diagnostics_001_ano (`HISTORIA`);

/* ---------------------------------------------------------
   FINAL TABLE
   --------------------------------------------------------- */

DROP TABLE IF EXISTS base_hospitalization_cohort;

CREATE TABLE base_hospitalization_cohort AS
WITH

/* 1) Select one valid adult hospital episode candidate.
   The source table can contain repeated rows per episode, so we keep the
   first hospitalization row ordered by DataIngres. */
ingres_base AS (
    SELECT
        h.`Episodi`,
        h.`TipusEpisodi`,
        h.`Nhc`,
        h.`DataIngres`,
        h.`Data_Naixament`,
        h.`DataHospitalitzacioAnterior`,
        h.`Edat`,
        h.`Sexe`,
        h.`ClaseAdmisio`,
        h.`Origenadmissiócodi`,
        h.`ServeiIngrés`,
        h.`DiagnosticPcodi`,
        h.`PassaperUCI`,
        h.`PassaperUCICCA`,
        h.`PassaperUnitatCoronaria`,
        h.`PassaperReanimacioPQ`,
        h.`PassaperSemicrítics`,
        h.`PassaperUnitatdeCremats`,
        h.`EstadaenUCI`,
        h.`EstadaenUCICCA`,
        h.`EstadaenUnitatCoronaria`,
        h.`EstadaenREAPostQuirurgica`,
        h.`EstadaenUnitatCremats`,
        h.`EsCirurgiaCardiaca`,
        h.`EsAltraCirurgiaCardiaca`,
        h.`EsCirurgiaValvular`,
        h.`EsCirurgiaCoronaria`,
        ROW_NUMBER() OVER (
            PARTITION BY h.`Episodi`
            ORDER BY h.`DataIngres`
        ) AS rn
    FROM tab_dt_sepsis_hosp_001_ano h
    WHERE h.`Episodi` IS NOT NULL
      AND h.`DataIngres` IS NOT NULL
      AND h.`TipusEpisodi` = 'H'
),

/* 2) Keep the selected episode row and validate age.
   If the recorded age is missing or implausible, age is recalculated from
   birth date and admission date when possible. */
ingres_episodi AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        b.`DataIngres`,
        b.`Data_Naixament`,
        b.`DataHospitalitzacioAnterior`,
        b.`Sexe`,
        b.`ClaseAdmisio`,
        b.`Origenadmissiócodi`,
        b.`ServeiIngrés`,
        b.`DiagnosticPcodi`,
        b.`PassaperUCI`,
        b.`PassaperUCICCA`,
        b.`PassaperUnitatCoronaria`,
        b.`PassaperReanimacioPQ`,
        b.`PassaperSemicrítics`,
        b.`PassaperUnitatdeCremats`,
        b.`EstadaenUCI`,
        b.`EstadaenUCICCA`,
        b.`EstadaenUnitatCoronaria`,
        b.`EstadaenREAPostQuirurgica`,
        b.`EstadaenUnitatCremats`,
        b.`EsCirurgiaCardiaca`,
        b.`EsAltraCirurgiaCardiaca`,
        b.`EsCirurgiaValvular`,
        b.`EsCirurgiaCoronaria`,
        CASE
            WHEN b.`Edat` IS NULL OR b.`Edat` < 0 OR b.`Edat` > 120 THEN
                CASE
                    WHEN b.`Data_Naixament` IS NOT NULL
                     AND b.`DataIngres` IS NOT NULL
                     AND (
                            TIMESTAMPDIFF(YEAR, b.`Data_Naixament`, b.`DataIngres`)
                            - CASE
                                WHEN DATE_FORMAT(b.`DataIngres`, '%m%d') < DATE_FORMAT(b.`Data_Naixament`, '%m%d')
                                THEN 1 ELSE 0
                              END
                         ) BETWEEN 0 AND 120
                    THEN
                        TIMESTAMPDIFF(YEAR, b.`Data_Naixament`, b.`DataIngres`)
                        - CASE
                            WHEN DATE_FORMAT(b.`DataIngres`, '%m%d') < DATE_FORMAT(b.`Data_Naixament`, '%m%d')
                            THEN 1 ELSE 0
                          END
                    ELSE NULL
                END
            ELSE b.`Edat`
        END AS edat_anys
    FROM ingres_base b
    WHERE b.rn = 1
),

/* 3) Normalize hospital ward movements.
   Dates are converted to DATETIME and each movement is flagged as critical
   care when any intensive or semi-critical bed indicator is active. */
passos_serveis_norm AS (
    SELECT
        ps.`Episodi`,
        ps.`NHC` AS `Nhc`,
        ps.`ServeiHospitalitzacio`,
        ps.`ServeiHospitalitzacioDescripció`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(ps.`DataIniciMoviment` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataIniciMoviment`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(ps.`DataAlta` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataAlta`,
        CASE
            WHEN COALESCE(ps.`ÉsLlitintensiuUCIqualsevolUCI`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCIGeneral`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCICardíaca`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCICoronària`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCIPediàtrica`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCINeonats`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuSemicríticsRespiratori`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuIctus`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuUCICremats`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuREAPostQuirúgica`, 0) = 1
              OR COALESCE(ps.`ÉsLlitintensiuSemicrítics`, 0) = 1
            THEN 1 ELSE 0
        END AS `Es_critic`
    FROM tab_dt_hosp_passos_serveis_001_ano ps
    WHERE ps.`Episodi` IS NOT NULL
),

/* 4) Normalize surgical activity.
   Surgery start and end timestamps are converted to DATETIME so duration and
   urgent surgery indicators can be calculated later. */
activitat_quirurgica_norm AS (
    SELECT
        aq.`Episodi`,
        aq.`IndicadorUrgentProgramat`,
        aq.`Tipuscirurgia`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(aq.`DataHoraIQ` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataHoraIQ`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(aq.`DataHorafimovimentIQ` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataHorafimovimentIQ`
    FROM tab_dt_activitat_quirurgica_001_ano aq
    WHERE aq.`Episodi` IS NOT NULL
),

/* 5) Summarize ward movements at episode level.
   This step derives the final discharge date, whether the first movement was
   critical care, whether the episode ever entered critical care, the last
   critical care discharge time, and total hours spent in critical care beds. */
passos_serveis_resum AS (
    SELECT
        ps.`Episodi`,
        MAX(
            CASE
                WHEN ps.`DataAlta` >= '9999-01-01' THEN NULL
                ELSE ps.`DataAlta`
            END
        ) AS `DataAlta`,
        CAST(
            SUBSTRING_INDEX(
                GROUP_CONCAT(ps.`Es_critic` ORDER BY ps.`DataIniciMoviment`, ps.`Es_critic` DESC SEPARATOR ','),
                ',',
                1
            ) AS UNSIGNED
        ) AS `Primer_moviment_es_critic`,
        MAX(CASE WHEN ps.`Es_critic` = 1 THEN 1 ELSE 0 END) AS `Passa_per_critics_moviments`,
        MAX(
            CASE
                WHEN ps.`Es_critic` = 1
                 AND ps.`DataAlta` IS NOT NULL
                 AND ps.`DataAlta` < '9999-01-01'
                THEN ps.`DataAlta`
                ELSE NULL
            END
        ) AS `Data_hora_alta_critics`,
        SUM(
            CASE
                WHEN ps.`Es_critic` = 1
                 AND ps.`DataIniciMoviment` IS NOT NULL
                 AND ps.`DataAlta` IS NOT NULL
                 AND ps.`DataAlta` < '9999-01-01'
                 AND ps.`DataAlta` >= ps.`DataIniciMoviment`
                THEN TIMESTAMPDIFF(
                    MINUTE,
                    ps.`DataIniciMoviment`,
                    ps.`DataAlta`
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_critics_serveis`
    FROM passos_serveis_norm ps
    WHERE ps.`Episodi` IS NOT NULL
      AND ps.`DataIniciMoviment` IS NOT NULL
    GROUP BY ps.`Episodi`
),

/* 6) Normalize emergency diagnosis codes.
   Codes are uppercased and stripped of dots/spaces so ICD patterns can be
   matched consistently. */
urgencies_diag_norm AS (
    SELECT
        u.`EpisodiHospitalitzacióPosterior` AS `Episodi`,
        UPPER(REPLACE(REPLACE(TRIM(u.`CodiDiagnòstic`), '.', ''), ' ', '')) AS `diag_code`
    FROM tab_dt_sepsis_urgencies_001_ano u
    WHERE u.`EpisodiHospitalitzacióPosterior` IS NOT NULL
      AND u.`CodiDiagnòstic` IS NOT NULL
      AND TRIM(u.`CodiDiagnòstic`) <> ''
),

/* 7) Identify episodes with sepsis already present at emergency admission.
   These episodes will be excluded because the cohort focuses on new
   in-hospital sepsis risk. */
urgencies_sepsia_ingres AS (
    SELECT DISTINCT
        udn.`Episodi`
    FROM urgencies_diag_norm udn
    WHERE udn.`diag_code` LIKE 'A40%'
       OR udn.`diag_code` LIKE 'A41%'
       OR udn.`diag_code` LIKE 'P36%'
       OR udn.`diag_code` IN ('R6520', 'R6521', 'A021', 'B377', 'O85', 'T814')
),

/* 8) Recover the linked emergency episode start date.
   When several emergency records point to the same hospitalization, the
   earliest linked date is used. */
urgencies_inici AS (
    SELECT
        u.`EpisodiHospitalitzacióPosterior` AS `Episodi`,
        MIN(u.`DataEpisodiHospitalitzacióPosterior`) AS `DataIniciUrgencies`
    FROM tab_dt_sepsis_urgencies_001_ano u
    WHERE u.`EpisodiHospitalitzacióPosterior` IS NOT NULL
      AND u.`DataEpisodiHospitalitzacióPosterior` IS NOT NULL
    GROUP BY u.`EpisodiHospitalitzacióPosterior`
),

/* 9) Summarize surgical activity at episode level.
   DataHoraIQ marks whether surgery occurred. Surgical duration is only kept
   when the interval is positive and plausible (1 minute to 24 hours), because
   many records have identical start and end timestamps. */
cirurgia_resum AS (
    SELECT
        aq.`Episodi`,
        1 AS `Cirurgia_activitat`,
        MAX(
            CASE
                WHEN aq.`IndicadorUrgentProgramat` = 'U'
                  OR UPPER(aq.`Tipuscirurgia`) LIKE '%URGENT%'
                THEN 1 ELSE 0
            END
        ) AS `Urgencia_cirurgia`,
        MAX(
            CASE
                WHEN aq.`DataHoraIQ` IS NOT NULL
                 AND aq.`DataHorafimovimentIQ` IS NOT NULL
                 AND TIMESTAMPDIFF(MINUTE, aq.`DataHoraIQ`, aq.`DataHorafimovimentIQ`) BETWEEN 1 AND 1440
                THEN 1 ELSE 0
            END
        ) AS `Temps_cirurgia_disponible`,
        SUM(
            CASE
                WHEN aq.`DataHoraIQ` IS NOT NULL
                 AND aq.`DataHorafimovimentIQ` IS NOT NULL
                 AND TIMESTAMPDIFF(MINUTE, aq.`DataHoraIQ`, aq.`DataHorafimovimentIQ`) BETWEEN 1 AND 1440
                THEN TIMESTAMPDIFF(
                    MINUTE,
                    aq.`DataHoraIQ`,
                    aq.`DataHorafimovimentIQ`
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_cirurgia`
    FROM activitat_quirurgica_norm aq
    WHERE aq.`Episodi` IS NOT NULL
    GROUP BY aq.`Episodi`
),

/* 10) Build the derived base cohort fields.
   This step combines hospitalization, emergency, ward movement, and surgery
   summaries into one row per episode with admission source, dates, prior
   hospitalization flags, critical care exposure, and surgery indicators. */
base_derivada AS (
    SELECT
        ie.`Episodi`,
        ie.`Nhc`,
        ie.`DataIngres`,
        CASE
            WHEN ui.`DataIniciUrgencies` IS NOT NULL
             AND ui.`DataIniciUrgencies` <= ie.`DataIngres`
            THEN ui.`DataIniciUrgencies`
            ELSE ie.`DataIngres`
        END AS `DataIniciUrgencies`,
        psr.`DataAlta`,

        CASE
            WHEN ie.`ClaseAdmisio` = 1 THEN 'Urgències'
            WHEN ie.`ClaseAdmisio` = 2 THEN 'Programat'
            ELSE NULL
        END AS `Font_admissio`,

        ie.`Origenadmissiócodi` AS `Centre_origen`,
        ie.`ServeiIngrés` AS `Codi_servei_admissor`,
        ie.`DiagnosticPcodi` AS `Diagnostic_ingres`,

        ie.edat_anys AS `Edat`,

        ie.`Sexe` AS `Sexe`,

        CASE
            WHEN ie.`DataHospitalitzacioAnterior` IS NOT NULL
             AND TIMESTAMPDIFF(DAY, ie.`DataHospitalitzacioAnterior`, ie.`DataIngres`) BETWEEN 0 AND 90
            THEN 1 ELSE 0
        END AS `Hospitalitzacio_recent_90d`,

        CASE
            WHEN ie.`DataHospitalitzacioAnterior` IS NOT NULL
             AND TIMESTAMPDIFF(DAY, ie.`DataHospitalitzacioAnterior`, ie.`DataIngres`) BETWEEN 0 AND 30
            THEN 1 ELSE 0
        END AS `Reingres_30d`,

        CASE
            WHEN COALESCE(psr.`Passa_per_critics_moviments`, 0) = 1
              OR COALESCE(ie.`PassaperUCI`, 0) = 1
              OR COALESCE(ie.`PassaperUCICCA`, 0) = 1
              OR COALESCE(ie.`PassaperUnitatCoronaria`, 0) = 1
              OR COALESCE(ie.`PassaperReanimacioPQ`, 0) = 1
              OR COALESCE(ie.`PassaperSemicrítics`, 0) = 1
              OR COALESCE(ie.`PassaperUnitatdeCremats`, 0) = 1
            THEN 1 ELSE 0
        END AS `Passa_per_critics`,

        COALESCE(psr.`Temps_critics_serveis`, 0)
          + COALESCE(cr.`Temps_cirurgia`, 0) AS `Temps_critics`,

        psr.`Data_hora_alta_critics`,

        /* Surgery is mainly identified from surgical activity records.
           Post-surgical recovery stay is kept as a fallback to avoid losing
           older episodes that were already marked in the hospital table. */
        CASE
            WHEN COALESCE(cr.`Cirurgia_activitat`, 0) = 1
              OR COALESCE(ie.`EstadaenREAPostQuirurgica`, 0) > 0
            THEN 1 ELSE 0
        END AS `Cirurgia`,

        COALESCE(cr.`Urgencia_cirurgia`, 0) AS `Urgencia_cirurgia`,
        COALESCE(cr.`Temps_cirurgia_disponible`, 0) AS `Temps_cirurgia_disponible`,
        CASE
            WHEN (
                    COALESCE(cr.`Cirurgia_activitat`, 0) = 1
                 OR COALESCE(ie.`EstadaenREAPostQuirurgica`, 0) > 0
                 )
             AND COALESCE(cr.`Temps_cirurgia_disponible`, 0) = 0
            THEN NULL
            ELSE COALESCE(cr.`Temps_cirurgia`, 0)
        END AS `Temps_cirurgia`
    FROM ingres_episodi ie
    LEFT JOIN passos_serveis_resum psr
        ON ie.`Episodi` = psr.`Episodi`
    LEFT JOIN cirurgia_resum cr
        ON ie.`Episodi` = cr.`Episodi`
    LEFT JOIN urgencies_inici ui
        ON ie.`Episodi` = ui.`Episodi`
),

/* 11) Normalize diagnosis codes for comorbidity detection.
   This step stacks secondary diagnoses from the diagnosis table and admission
   diagnoses from the hospitalization table. Procedure fields are handled in
   the next block because their meaning is procedural. */
diagnostics_norm AS (
    SELECT
        x.`Nhc`,
        UPPER(REPLACE(REPLACE(TRIM(x.`diag_code_raw`), '.', ''), ' ', '')) AS `diag_code`,
        UPPER(TRIM(x.`diag_desc_raw`)) AS `diag_desc`
    FROM (
        SELECT d.`HISTORIA` AS `Nhc`, d.`DS1` AS `diag_code_raw`, CAST(NULL AS CHAR(255)) AS `diag_desc_raw`
        FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS2`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS3`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS4`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS5`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS6`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS7`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS8`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS9`,  CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS10`, CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS11`, CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS12`, CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS13`, CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`DS14`, CAST(NULL AS CHAR(255)) FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT h.`Nhc`, h.`DiagnosticPcodi`, h.`DiagnosticPdesc`
        FROM tab_dt_sepsis_hosp_001_ano h
        UNION ALL
        SELECT h.`Nhc`, h.`DiagnòsticCodi`, h.`Diagnòstic`
        FROM tab_dt_sepsis_hosp_001_ano h
    ) x
    WHERE x.`Nhc` IS NOT NULL
      AND x.`diag_code_raw` IS NOT NULL
      AND TRIM(x.`diag_code_raw`) <> ''
),

/* 12) Normalize procedure codes that can define comorbidities.
   PP and PS1..PS10 are used when a procedure directly implies a condition,
   for example splenectomy with ICD-10-PCS 07TP* codes. */
procediments_norm AS (
    SELECT
        x.`Nhc`,
        UPPER(REPLACE(REPLACE(TRIM(x.`proc_code_raw`), '.', ''), ' ', '')) AS `proc_code`
    FROM (
        SELECT d.`HISTORIA` AS `Nhc`, d.`PP` AS `proc_code_raw`
        FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS1`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS2`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS3`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS4`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS5`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS6`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS7`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS8`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS9`  FROM tab_dt_sepsis_diagnostics_001_ano d
        UNION ALL
        SELECT d.`HISTORIA`, d.`PS10` FROM tab_dt_sepsis_diagnostics_001_ano d
    ) x
    WHERE x.`Nhc` IS NOT NULL
      AND x.`proc_code_raw` IS NOT NULL
      AND TRIM(x.`proc_code_raw`) <> ''
),

/* 13) Create the patient universe for comorbidity aggregation.
   A patient is included if they have at least one diagnosis or procedure code. */
pacients_codis AS (
    SELECT `Nhc` FROM diagnostics_norm
    UNION
    SELECT `Nhc` FROM procediments_norm
),

/* 14) Aggregate procedure-defined flags at patient level. */
procediments_pacient AS (
    SELECT
        pn.`Nhc`,
        MAX(CASE
            WHEN pn.`proc_code` LIKE '07TP%'
            THEN 1 ELSE 0
        END) AS `CODI_ESPLENECTOMIA`
    FROM procediments_norm pn
    GROUP BY pn.`Nhc`
),

/* 15) Aggregate diagnosis- and procedure-based comorbidities at patient level.
   Each flag is set to 1 if any available code or text description matches the
   operational definition for that comorbidity. */
comorbiditats_pacient AS (
    SELECT
        pc.`Nhc`,

        /* DIABETES MELLITUS */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'E10%'
              OR dn.`diag_code` LIKE 'E11%'
              OR dn.`diag_code` LIKE 'E12%'
              OR dn.`diag_code` LIKE 'E13%'
              OR dn.`diag_code` LIKE 'E14%'
            THEN 1 ELSE 0
        END) AS `COMORB_DIABETES_MELLITUS`,

        /* SOLID TUMOR */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'C%'
             AND dn.`diag_code` NOT LIKE 'C81%'
             AND dn.`diag_code` NOT LIKE 'C82%'
             AND dn.`diag_code` NOT LIKE 'C83%'
             AND dn.`diag_code` NOT LIKE 'C84%'
             AND dn.`diag_code` NOT LIKE 'C85%'
             AND dn.`diag_code` NOT LIKE 'C86%'
             AND dn.`diag_code` NOT LIKE 'C88%'
             AND dn.`diag_code` NOT LIKE 'C90%'
             AND dn.`diag_code` NOT LIKE 'C91%'
             AND dn.`diag_code` NOT LIKE 'C92%'
             AND dn.`diag_code` NOT LIKE 'C93%'
             AND dn.`diag_code` NOT LIKE 'C94%'
             AND dn.`diag_code` NOT LIKE 'C95%'
             AND dn.`diag_code` NOT LIKE 'D46%'
             AND dn.`diag_code` NOT LIKE 'D47%'
            THEN 1 ELSE 0
        END) AS `COMORB_NEOPLASIA_SOLIDA`,

        /* HEMATOLOGICAL MALIGNANCY */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'C81%'
              OR dn.`diag_code` LIKE 'C82%'
              OR dn.`diag_code` LIKE 'C83%'
              OR dn.`diag_code` LIKE 'C84%'
              OR dn.`diag_code` LIKE 'C85%'
              OR dn.`diag_code` LIKE 'C86%'
              OR dn.`diag_code` LIKE 'C88%'
              OR dn.`diag_code` LIKE 'C90%'
              OR dn.`diag_code` LIKE 'C91%'
              OR dn.`diag_code` LIKE 'C92%'
              OR dn.`diag_code` LIKE 'C93%'
              OR dn.`diag_code` LIKE 'C94%'
              OR dn.`diag_code` LIKE 'C95%'
              OR dn.`diag_code` LIKE 'D46%'
              OR dn.`diag_code` LIKE 'D47%'
            THEN 1 ELSE 0
        END) AS `COMORB_NEOPLASIA_HEMATOLOGICA`,

        /* SEVERE ALCOHOL USE DISORDER */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'F102%'
              OR dn.`diag_code` LIKE 'K70%'
              OR dn.`diag_desc` LIKE '%ALCOHOL%'
              OR dn.`diag_desc` LIKE '%ENOLIS%'
            THEN 1 ELSE 0
        END) AS `COMORB_ENOLISME_SEVER`,

        /* LIVER CIRRHOSIS */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'K74%'
              OR dn.`diag_desc` LIKE '%CIRROSI%'
            THEN 1 ELSE 0
        END) AS `COMORB_CIRROSI_HEPATICA`,

        /* HIV / AIDS */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'B20%'
              OR dn.`diag_code` LIKE 'B21%'
              OR dn.`diag_code` LIKE 'B22%'
              OR dn.`diag_code` LIKE 'B24%'
            THEN 1 ELSE 0
        END) AS `COMORB_VIH_SIDA`,

        /* SOLID ORGAN TRANSPLANT */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'Z94%'
             AND dn.`diag_code` NOT LIKE 'Z9481%'
             AND dn.`diag_code` NOT LIKE 'Z9484%'
            THEN 1 ELSE 0
        END) AS `COMORB_TRASPLANT_ORGAN_SOLID`,

        /* BONE MARROW TRANSPLANT */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'Z9481%'
              OR dn.`diag_code` LIKE 'Z9484%'
              OR dn.`diag_desc` LIKE '%MOLL D''OS%'
              OR dn.`diag_desc` LIKE '%MEDUL%'
              OR dn.`diag_desc` LIKE '%TRASPLANT%HEMATOPOI%'
            THEN 1 ELSE 0
        END) AS `COMORB_TRASPLANT_MOLL_OS`,

        /* AGAMMAGLOBULINEMIA */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'D800%'
              OR dn.`diag_desc` LIKE '%AGAMMAGLOBULIN%'
            THEN 1 ELSE 0
        END) AS `COMORB_AGAMMAGLOBULINEMIA`,

        /* HYPOGAMMAGLOBULINEMIA */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'D801%'
              OR dn.`diag_code` LIKE 'D807%'
              OR dn.`diag_desc` LIKE '%HIPOGAMMAGLOBULIN%'
            THEN 1 ELSE 0
        END) AS `COMORB_HIPOGAMMAGLOBULINEMIA`,

        /* MALABSORPTION SYNDROMES */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'K90%'
              OR dn.`diag_desc` LIKE '%MALABSORT%'
              OR dn.`diag_desc` LIKE '%CROHN%'
              OR dn.`diag_desc` LIKE '%BUDELL CURT%'
              OR dn.`diag_desc` LIKE '%INTESTI CURT%'
            THEN 1 ELSE 0
        END) AS `COMORB_MALABSORTIVES`,

        /* SEVERE MALNUTRITION */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'E43%'
              OR dn.`diag_code` LIKE 'E44%'
              OR dn.`diag_desc` LIKE '%MALNUTRIC%'
              OR dn.`diag_desc` LIKE '%DESNUTRIC%'
            THEN 1 ELSE 0
        END) AS `COMORB_MALNUTRICIO_SEVERA`,

        /* ASPLENIA */
        MAX(CASE
            WHEN dn.`diag_desc` LIKE '%ASPLEN%'
              OR dn.`diag_code` LIKE 'Q8901%'
              OR dn.`diag_code` LIKE 'D73%'
              OR COALESCE(pp.`CODI_ESPLENECTOMIA`, 0) = 1
            THEN 1 ELSE 0
        END) AS `COMORB_ASPLENIA`,

        /* SPLENECTOMY */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'Z9041%'
              OR dn.`diag_desc` LIKE '%ESPLENECTOM%'
              OR COALESCE(pp.`CODI_ESPLENECTOMIA`, 0) = 1
            THEN 1 ELSE 0
        END) AS `COMORB_ESPLENECTOMIA`,

        /* CHRONIC KIDNEY DISEASE ON DIALYSIS */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'N186%'
              OR dn.`diag_code` LIKE 'Z992%'
              OR dn.`diag_desc` LIKE '%DIALI%'
            THEN 1 ELSE 0
        END) AS `COMORB_IRC_DIALISI`,

        /* SEVERE NEUTROPENIA */
        MAX(CASE
            WHEN dn.`diag_code` LIKE 'D70%'
              OR dn.`diag_desc` LIKE '%NEUTROPEN%'
            THEN 1 ELSE 0
        END) AS `COMORB_NEUTROPENIA_GREU`

    FROM pacients_codis pc
    LEFT JOIN diagnostics_norm dn
        ON pc.`Nhc` = dn.`Nhc`
    LEFT JOIN procediments_pacient pp
        ON pc.`Nhc` = pp.`Nhc`
    GROUP BY pc.`Nhc`
)

/* 16) Create the final base cohort.
   Comorbidity flags are attached to each episode, missing flags are set to 0,
   and exclusion criteria are applied: study period, adult age, sepsis already
   present at admission, and initial admission to critical care. */
SELECT
    bd.*,

    COALESCE(c.`COMORB_DIABETES_MELLITUS`, 0)      AS `COMORB_DIABETES_MELLITUS`,
    COALESCE(c.`COMORB_NEOPLASIA_SOLIDA`, 0)       AS `COMORB_NEOPLASIA_SOLIDA`,
    COALESCE(c.`COMORB_NEOPLASIA_HEMATOLOGICA`, 0) AS `COMORB_NEOPLASIA_HEMATOLOGICA`,
    COALESCE(c.`COMORB_ENOLISME_SEVER`, 0)         AS `COMORB_ENOLISME_SEVER`,
    COALESCE(c.`COMORB_CIRROSI_HEPATICA`, 0)       AS `COMORB_CIRROSI_HEPATICA`,
    COALESCE(c.`COMORB_VIH_SIDA`, 0)               AS `COMORB_VIH_SIDA`,
    COALESCE(c.`COMORB_TRASPLANT_ORGAN_SOLID`, 0)  AS `COMORB_TRASPLANT_ORGAN_SOLID`,
    COALESCE(c.`COMORB_TRASPLANT_MOLL_OS`, 0)      AS `COMORB_TRASPLANT_MOLL_OS`,
    COALESCE(c.`COMORB_AGAMMAGLOBULINEMIA`, 0)     AS `COMORB_AGAMMAGLOBULINEMIA`,
    COALESCE(c.`COMORB_HIPOGAMMAGLOBULINEMIA`, 0)  AS `COMORB_HIPOGAMMAGLOBULINEMIA`,
    COALESCE(c.`COMORB_MALABSORTIVES`, 0)          AS `COMORB_MALABSORTIVES`,
    COALESCE(c.`COMORB_MALNUTRICIO_SEVERA`, 0)     AS `COMORB_MALNUTRICIO_SEVERA`,
    COALESCE(c.`COMORB_ASPLENIA`, 0)               AS `COMORB_ASPLENIA`,
    COALESCE(c.`COMORB_ESPLENECTOMIA`, 0)          AS `COMORB_ESPLENECTOMIA`,
    COALESCE(c.`COMORB_IRC_DIALISI`, 0)            AS `COMORB_IRC_DIALISI`,
    COALESCE(c.`COMORB_NEUTROPENIA_GREU`, 0)       AS `COMORB_NEUTROPENIA_GREU`

FROM base_derivada bd
LEFT JOIN urgencies_sepsia_ingres usi
    ON bd.`Episodi` = usi.`Episodi`
LEFT JOIN passos_serveis_resum psr_final
    ON bd.`Episodi` = psr_final.`Episodi`
LEFT JOIN comorbiditats_pacient c
    ON bd.`Nhc` = c.`Nhc`
WHERE bd.`DataIngres` >= '2018-01-01'
  AND bd.`DataIngres` < '2027-01-01'
  AND bd.`Edat` >= 18
  AND usi.`Episodi` IS NULL
  AND COALESCE(psr_final.`Primer_moviment_es_critic`, 0) = 0
  AND NOT (
        bd.`Diagnostic_ingres` IS NOT NULL
    AND (
           UPPER(REPLACE(REPLACE(TRIM(bd.`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A40%'
        OR UPPER(REPLACE(REPLACE(TRIM(bd.`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'A41%'
        OR UPPER(REPLACE(REPLACE(TRIM(bd.`Diagnostic_ingres`), '.', ''), ' ', '')) LIKE 'P36%'
        OR UPPER(REPLACE(REPLACE(TRIM(bd.`Diagnostic_ingres`), '.', ''), ' ', '')) IN ('R6520', 'R6521', 'A021', 'B377', 'O85', 'T814')
    )
  );


/* ---------------------------------------------------------
   FINAL INDEXES ON THE RESULTING TABLE
   These indexes support downstream joins and date filtering in the later SQL
   scripts that build daily features and the final modeling table.
   --------------------------------------------------------- */

CREATE INDEX idx_cohort_final_episodi
    ON base_hospitalization_cohort (`Episodi`);

CREATE INDEX idx_cohort_final_nhc
    ON base_hospitalization_cohort(`Nhc`);

CREATE INDEX idx_cohort_final_dataingres
    ON base_hospitalization_cohort (`DataIngres`);

CREATE INDEX idx_cohort_final_datainiciurg
    ON base_hospitalization_cohort (`DataIniciUrgencies`);

CREATE INDEX idx_cohort_final_dataalta
    ON base_hospitalization_cohort (`DataAlta`);

