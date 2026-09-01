/* ============================================================
   BUILD DAILY PHARMACY FEATURES

   Operational notes:
   - The pharmacy source contains dispensing/consumption records, not infusion
     rates or real administered doses in mcg/kg/min.
   - Vasopressors are therefore represented as daily exposure flags only.
   - Antimicrobial treatment is identified with therapeutic groups
     J01/J02/H02/L01, according to the project definition.
   - Intravenous route cannot be confirmed because the source does not provide
     a reliable route field.

   Outputs:
   - clean_pharmacy_events: pharmacy records restricted to the base cohort.
   - daily_pharmacy_features: one row per Episodi + Nhc + Data_dia with
     vasopressor and antimicrobial features.
   ============================================================ */


/* ------------------------------------------------------------
   Recommended source indexes
   Run only once if they do not already exist.
   ------------------------------------------------------------ */

CREATE INDEX idx_pharmacy_episode_date
    ON tab_dt_sepsis_farmacia_001_ano (`EpisodiID`, `DataConsum`);

CREATE INDEX idx_pharmacy_episode_date_group
    ON tab_dt_sepsis_farmacia_001_ano (`EpisodiID`, `DataConsum`, `GrupTerapèuticCodi`);

CREATE INDEX idx_pharmacy_drug_code
    ON tab_dt_sepsis_farmacia_001_ano (`MedicamentCodi`);


/* ------------------------------------------------------------
   0) Observed hospital days
   Uses the expanded daily grid from daily_laboratory_records so pharmacy
   features align with model days based on vital signs or laboratory records.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS pharmacy_observed_days;

CREATE TABLE pharmacy_observed_days AS
WITH base_cohort AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        b.`DataIngres`
    FROM base_hospitalization_cohort b
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND b.`DataIngres` IS NOT NULL
)
SELECT
    b.`Episodi`,
    b.`Nhc`,
    d.`data_index` AS `Data_dia`,
    DATEDIFF(d.`data_index`, DATE(b.`DataIngres`)) AS `dia_relatiu`
FROM base_cohort b
INNER JOIN daily_laboratory_records d
    ON b.`Episodi` = d.`Episodi`
   AND b.`Nhc` = d.`Nhc`
WHERE d.`data_index` IS NOT NULL
GROUP BY
    b.`Episodi`,
    b.`Nhc`,
    d.`data_index`,
    DATEDIFF(d.`data_index`, DATE(b.`DataIngres`));

CREATE INDEX idx_pharmacy_observed_days_episode_day
    ON pharmacy_observed_days (`Episodi`, `Data_dia`);

CREATE INDEX idx_pharmacy_observed_days_nhc_day
    ON pharmacy_observed_days (`Nhc`, `Data_dia`);

/* ------------------------------------------------------------
   1) Clean pharmacy events
   Keeps only cohort episodes and records needed for vasopressor or
   antimicrobial features.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS clean_pharmacy_events;

CREATE TABLE clean_pharmacy_events AS
SELECT
    b.`Episodi`,
    b.`Nhc`,
    f.`CentreCodi`,
    f.`EpisodiID`,
    f.`Episodi` AS `Episodi_farmacia`,
    CAST(f.`MedicamentCodi` AS UNSIGNED) AS `Codi_medicament`,
    f.`MedicamentDescripció` AS `Descripcio_medicament`,
    f.`MedicamentActiu` AS `Principi_actiu`,
    f.`GrupTerapèuticCodi` AS `Codi_grup_terapeutic`,
    f.`GrupTerapèuticDescripció` AS `Descripcio_grup_terapeutic`,
    CAST(f.`MedicamentDosi` AS DECIMAL(14,4)) AS `Dosi_mg_presentacio`,
    f.`DataConsum` AS `Data_hora_consum`,
    DATE(f.`DataConsum`) AS `Data_dia`,
    f.`data_carrega`
FROM base_hospitalization_cohort b
INNER JOIN tab_dt_sepsis_farmacia_001_ano f
    ON b.`Episodi` = f.`EpisodiID`
   AND f.`DataConsum` >= b.`DataIngres`
WHERE f.`EpisodiID` IS NOT NULL
  AND f.`DataConsum` IS NOT NULL
  AND (
        CAST(f.`MedicamentCodi` AS UNSIGNED) IN (
            1968, 25143,   /* Dobutamine */
            149,           /* Dopamine */
            28823, 655     /* Noradrenaline */
        )
        OR UPPER(COALESCE(f.`MedicamentDescripció`, '')) LIKE '%NORADRENALINA%'
        OR UPPER(COALESCE(f.`MedicamentActiu`, '')) LIKE '%NORADRENALINA%'
        OR UPPER(COALESCE(f.`MedicamentDescripció`, '')) LIKE '%NOREPINEFRINA%'
        OR UPPER(COALESCE(f.`MedicamentActiu`, '')) LIKE '%NOREPINEFRINA%'
        OR UPPER(COALESCE(f.`MedicamentDescripció`, '')) LIKE '%ADRENALINA%'
        OR UPPER(COALESCE(f.`MedicamentActiu`, '')) LIKE '%ADRENALINA%'
        OR UPPER(COALESCE(f.`MedicamentDescripció`, '')) LIKE '%EPINEFRINA%'
        OR UPPER(COALESCE(f.`MedicamentActiu`, '')) LIKE '%EPINEFRINA%'
        OR f.`GrupTerapèuticCodi` LIKE 'J01%'
        OR f.`GrupTerapèuticCodi` LIKE 'J02%'
        OR f.`GrupTerapèuticCodi` LIKE 'H02%'
        OR f.`GrupTerapèuticCodi` LIKE 'L01%'
  );

CREATE INDEX idx_clean_pharmacy_events_episode_day_drug
    ON clean_pharmacy_events (`Episodi`, `Data_dia`, `Nhc`, `Codi_medicament`);


/* ------------------------------------------------------------
   2) Prior antimicrobial exposure
   Flags whether the patient had J01/J02/H02/L01 pharmacy exposure during the
   90 days before each observed hospital day.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS prior_antimicrobial_exposure_90d;

CREATE TABLE prior_antimicrobial_exposure_90d AS
SELECT
    d.`Episodi`,
    d.`Data_dia`,
    CASE
        WHEN COUNT(p.`EpisodiID`) > 0 THEN 1
        ELSE 0
    END AS `antibiotics_previs_90d`
FROM pharmacy_observed_days d
LEFT JOIN base_hospitalization_cohort bp
    ON bp.`Nhc` = d.`Nhc`
LEFT JOIN tab_dt_sepsis_farmacia_001_ano p
    ON p.`EpisodiID` = bp.`Episodi`
   AND p.`DataConsum` >= DATE_SUB(d.`Data_dia`, INTERVAL 90 DAY)
   AND p.`DataConsum` < d.`Data_dia`
   AND (
        p.`GrupTerapèuticCodi` LIKE 'J01%'
        OR p.`GrupTerapèuticCodi` LIKE 'J02%'
        OR p.`GrupTerapèuticCodi` LIKE 'H02%'
        OR p.`GrupTerapèuticCodi` LIKE 'L01%'
   )
GROUP BY
    d.`Episodi`,
    d.`Data_dia`;

CREATE INDEX idx_prior_antimicrobial_90d_episode_day
    ON prior_antimicrobial_exposure_90d (`Episodi`, `Data_dia`);


/* ------------------------------------------------------------
   3) Active antimicrobial duration
   Builds the daily antimicrobial flag and estimates duration as consecutive
   active days multiplied by 24 hours.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS daily_antimicrobial_duration;

CREATE TABLE daily_antimicrobial_duration AS
WITH antimicrobial_days AS (
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`Data_dia`,
        CASE
            WHEN COUNT(f.`Episodi`) > 0 THEN 1
            ELSE 0
        END AS `antibiotic`
    FROM pharmacy_observed_days d
    LEFT JOIN clean_pharmacy_events f
        ON d.`Episodi` = f.`Episodi`
       AND d.`Data_dia` = f.`Data_dia`
       AND (
            f.`Codi_grup_terapeutic` LIKE 'J01%'
            OR f.`Codi_grup_terapeutic` LIKE 'J02%'
            OR f.`Codi_grup_terapeutic` LIKE 'H02%'
            OR f.`Codi_grup_terapeutic` LIKE 'L01%'
       )
    GROUP BY
        d.`Episodi`,
        d.`Nhc`,
        d.`Data_dia`
),
antimicrobial_streaks AS (
    SELECT
        a.*,
        SUM(CASE WHEN a.`antibiotic` = 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY a.`Episodi`
                ORDER BY a.`Data_dia`
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS `active_streak_group`
    FROM antimicrobial_days a
)
SELECT
    s.`Episodi`,
    s.`Nhc`,
    s.`Data_dia`,
    s.`antibiotic`,
    CASE
        WHEN s.`antibiotic` = 1
        THEN
            SUM(s.`antibiotic`) OVER (
                PARTITION BY s.`Episodi`, s.`active_streak_group`
                ORDER BY s.`Data_dia`
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) * 24
        ELSE 0
    END AS `atb_duracio`
FROM antimicrobial_streaks s;

CREATE INDEX idx_daily_antimicrobial_duration_episode_day
    ON daily_antimicrobial_duration (`Episodi`, `Data_dia`);


/* ------------------------------------------------------------
   4) Daily pharmacy features
   Produces one row per observed episode-day with vasopressor exposure,
   antimicrobial treatment, cumulative antimicrobial duration, and prior
   antimicrobial exposure.
   ------------------------------------------------------------ */

DROP TABLE IF EXISTS daily_pharmacy_features;

CREATE TABLE daily_pharmacy_features AS
WITH classified_pharmacy_events AS (
    SELECT
        f.*,
        CASE
            WHEN f.`Codi_medicament` IN (1968, 25143)
            THEN 1 ELSE 0
        END AS `es_dobutamina`,

        CASE
            WHEN f.`Codi_medicament` = 149
            THEN 1 ELSE 0
        END AS `es_dopamina`,

        CASE
            WHEN f.`Codi_medicament` IN (28823, 655)
              OR UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%NORADRENALINA%'
              OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%NORADRENALINA%'
              OR UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%NOREPINEFRINA%'
              OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%NOREPINEFRINA%'
            THEN 1 ELSE 0
        END AS `es_noradrenalina`,

        CASE
            WHEN (
                    UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%ADRENALINA%'
                 OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%ADRENALINA%'
                 OR UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%EPINEFRINA%'
                 OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%EPINEFRINA%'
                 )
             AND NOT (
                    UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%NORADRENALINA%'
                 OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%NORADRENALINA%'
                 OR UPPER(COALESCE(f.`Descripcio_medicament`, '')) LIKE '%NOREPINEFRINA%'
                 OR UPPER(COALESCE(f.`Principi_actiu`, '')) LIKE '%NOREPINEFRINA%'
                 )
            THEN 1 ELSE 0
        END AS `es_adrenalina`
    FROM clean_pharmacy_events f
),
aggregated_features AS (
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`Data_dia`,

        /* Daily vasopressor exposure flags. */
        COALESCE(MAX(f.`es_dobutamina`), 0) AS `vasopressor_dobutamina`,
        COALESCE(MAX(f.`es_dopamina`), 0) AS `vasopressor_dopamina`,
        COALESCE(MAX(f.`es_noradrenalina`), 0) AS `vasopressor_noradrenalina`,
        COALESCE(MAX(f.`es_adrenalina`), 0) AS `vasopressor_adrenalina`,

        /* Daily antimicrobial features available in the pharmacy source. */
        COALESCE(MAX(ad.`antibiotic`), 0) AS `antibiotic`,
        COALESCE(MAX(ad.`atb_duracio`), 0) AS `atb_duracio`,
        COALESCE(MAX(a.`antibiotics_previs_90d`), 0) AS `antibiotics_previs_90d`

    FROM pharmacy_observed_days d
    LEFT JOIN classified_pharmacy_events f
        ON d.`Episodi` = f.`Episodi`
       AND d.`Data_dia` = f.`Data_dia`
    LEFT JOIN daily_antimicrobial_duration ad
        ON d.`Episodi` = ad.`Episodi`
       AND d.`Data_dia` = ad.`Data_dia`
    LEFT JOIN prior_antimicrobial_exposure_90d a
        ON d.`Episodi` = a.`Episodi`
       AND d.`Data_dia` = a.`Data_dia`
    GROUP BY
        d.`Episodi`,
        d.`Nhc`,
        d.`Data_dia`
)
SELECT
    `Episodi`,
    `Nhc`,
    `Data_dia`,
    `vasopressor_dobutamina`,
    `vasopressor_dopamina`,
    `vasopressor_noradrenalina`,
    `vasopressor_adrenalina`,
    CASE
        WHEN (
            `vasopressor_dobutamina` +
            `vasopressor_dopamina` +
            `vasopressor_noradrenalina` +
            `vasopressor_adrenalina`
        ) >= 1
        THEN 1 ELSE 0
    END AS `vasopressor_qualsevol`,
    CASE
        WHEN (
            `vasopressor_dobutamina` +
            `vasopressor_dopamina` +
            `vasopressor_noradrenalina` +
            `vasopressor_adrenalina`
        ) >= 2
        THEN 1 ELSE 0
    END AS `vasopressor_multiple`,
    `antibiotic`,
    `atb_duracio`,
    `antibiotics_previs_90d`
FROM aggregated_features;

CREATE INDEX idx_daily_pharmacy_features_episode_day
    ON daily_pharmacy_features (`Episodi`, `Data_dia`);

CREATE UNIQUE INDEX idx_daily_pharmacy_features_episode_nhc_day
    ON daily_pharmacy_features (`Episodi`, `Nhc`, `Data_dia`);
