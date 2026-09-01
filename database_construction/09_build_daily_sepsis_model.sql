/* ============================================================
   BUILD DAILY SEPSIS MODEL

   Purpose:
   - Create daily_sepsis_model with one row per Episodi + Nhc + data_index.
   - Join the base cohort with daily vital signs, laboratory records, pharmacy
     features, critical-care exposure, surgery, and invasive-device history.
   - Use an expanded daily calendar based on vital signs and selected
     laboratory records.

   Key rules:
   - Keep episodes with at least one vital-sign event 24 hours after DataIngres.
   - Keep only episodes with at least one creatinine or platelet value on a
     model day, to support renal/coagulation SOFA components.
   - Keep only model days within the hospital admission interval.
   - Fixed cohort variables are repeated on every episode-day.
   - Time-dependent critical-care and surgery variables are recalculated up to
     the end of each data_index.
   - Missing daily flags are encoded as 0.
   ============================================================ */

DROP TABLE IF EXISTS daily_sepsis_model;

CREATE TABLE daily_sepsis_model AS
WITH model_days AS (
    SELECT
        b.`Episodi`,
        b.`Nhc`,
        cv.`data_index` AS `data_index`,
        DATEDIFF(cv.`data_index`, DATE(b.`DataIngres`)) AS `dia_relatiu`
    FROM base_hospitalization_cohort b
    INNER JOIN daily_vital_signs cv
        ON b.`Episodi` = cv.`Episodi`
       AND b.`Nhc` = cv.`Nhc`
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND cv.`data_index` IS NOT NULL
      AND cv.`data_index` >= DATE(b.`DataIngres`)
      AND (
            b.`DataAlta` IS NULL
         OR cv.`data_index` <= DATE(b.`DataAlta`)
      )

    UNION

    SELECT
        b.`Episodi`,
        b.`Nhc`,
        l.`data_index` AS `data_index`,
        DATEDIFF(l.`data_index`, DATE(b.`DataIngres`)) AS `dia_relatiu`
    FROM base_hospitalization_cohort b
    INNER JOIN daily_laboratory_records l
        ON b.`Episodi` = l.`Episodi`
       AND b.`Nhc` = l.`Nhc`
    WHERE b.`Episodi` IS NOT NULL
      AND b.`Nhc` IS NOT NULL
      AND l.`data_index` IS NOT NULL
      AND l.`data_index` >= DATE(b.`DataIngres`)
      AND (
            b.`DataAlta` IS NULL
         OR l.`data_index` <= DATE(b.`DataAlta`)
      )
),

episodes_with_24h_vital_data AS (
    /* Keep episodes with an observed vital-sign event at least 24 hours after
       admission. DataAlta is not used as the stay-length criterion. */
    SELECT
        b.`Episodi`,
        b.`Nhc`
    FROM base_hospitalization_cohort b
    INNER JOIN clean_vital_signs_events cv
        ON b.`Episodi` = cv.`Episodi`
       AND b.`Nhc` = cv.`Nhc`
    WHERE b.`DataIngres` IS NOT NULL
      AND cv.`event_time` IS NOT NULL
      AND TIMESTAMPDIFF(HOUR, b.`DataIngres`, cv.`event_time`) >= 24
    GROUP BY b.`Episodi`, b.`Nhc`
),

episodes_with_creatinine_or_platelets AS (
    /* Keep all model days for episodes with at least one creatinine or
       platelet value on any model day. */
    SELECT
        d.`Episodi`,
        d.`Nhc`
    FROM model_days d
    INNER JOIN daily_laboratory_records l
        ON d.`Episodi` = l.`Episodi`
       AND d.`Nhc` = l.`Nhc`
       AND d.`data_index` = l.`data_index`
    WHERE (
            l.`creatinina` IS NOT NULL
         OR l.`plaquetes` IS NOT NULL
      )
    GROUP BY d.`Episodi`, d.`Nhc`
),

normalized_service_movements AS (
    SELECT
        ps.`Episodi`,
        ps.`NHC` AS `Nhc`,
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

normalized_surgery_activity AS (
    SELECT
        aq.`Episodi`,
        aq.`IndicadorUrgentProgramat`,
        aq.`Tipuscirurgia`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(aq.`DataHoraIQ` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataHoraIQ`,
        CAST(NULLIF(REPLACE(REPLACE(CAST(aq.`DataHorafimovimentIQ` AS CHAR), 'T', ' '), 'Z', ''), '') AS DATETIME) AS `DataHorafimovimentIQ`
    FROM tab_dt_activitat_quirurgica_001_ano aq
    WHERE aq.`Episodi` IS NOT NULL
),

critical_care_movements AS (
    SELECT DISTINCT
        ps.`Episodi`,
        ps.`Nhc`,
        ps.`DataIniciMoviment`,
        ps.`DataAlta`
    FROM normalized_service_movements ps
    WHERE ps.`Episodi` IS NOT NULL
      AND ps.`DataIniciMoviment` IS NOT NULL
      AND ps.`Es_critic` = 1
),

daily_critical_care AS (
    /* Daily critical-care status and cumulative hours up to each data_index. */
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`,
        MAX(
            CASE
                WHEN mc.`DataIniciMoviment` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND (
                        mc.`DataAlta` IS NULL
                     OR mc.`DataAlta` >= '9999-01-01'
                     OR mc.`DataAlta` > d.`data_index`
                 )
                THEN 1 ELSE 0
            END
        ) AS `En_critics_dia`,
        SUM(
            CASE
                WHEN mc.`DataIniciMoviment` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND (
                        mc.`DataAlta` IS NULL
                     OR mc.`DataAlta` >= '9999-01-01'
                     OR mc.`DataAlta` > d.`data_index`
                 )
                THEN GREATEST(
                    TIMESTAMPDIFF(
                        MINUTE,
                        GREATEST(mc.`DataIniciMoviment`, d.`data_index`),
                        LEAST(
                            CASE
                                WHEN mc.`DataAlta` IS NULL
                                  OR mc.`DataAlta` >= '9999-01-01'
                                THEN DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                                ELSE mc.`DataAlta`
                            END,
                            DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                        )
                    ),
                    0
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_critics_dia`,
        MAX(
            CASE
                WHEN mc.`DataIniciMoviment` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                THEN 1 ELSE 0
            END
        ) AS `Passa_per_critics`,
        SUM(
            CASE
                WHEN mc.`DataIniciMoviment` IS NOT NULL
                 AND mc.`DataIniciMoviment` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                THEN GREATEST(
                    TIMESTAMPDIFF(
                        MINUTE,
                        mc.`DataIniciMoviment`,
                        CASE
                            WHEN mc.`DataAlta` IS NULL
                              OR mc.`DataAlta` >= '9999-01-01'
                              OR mc.`DataAlta` > DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                            THEN DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                            ELSE mc.`DataAlta`
                        END
                    ),
                    0
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_critics`,
        MAX(
            CASE
                WHEN mc.`DataAlta` IS NOT NULL
                 AND mc.`DataAlta` < '9999-01-01'
                 AND mc.`DataAlta` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                THEN mc.`DataAlta`
                ELSE NULL
            END
        ) AS `Data_hora_alta_critics`
    FROM model_days d
    LEFT JOIN critical_care_movements mc
        ON d.`Episodi` = mc.`Episodi`
       AND mc.`DataIniciMoviment` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
    GROUP BY d.`Episodi`, d.`Nhc`, d.`data_index`
),

pre_critical_return_labs_ranked AS (
    /* Last clean creatinine, platelet, and bilirubin values within the 72 hours
       before critical-care discharge. Uses clean events so tests from days
       without vital signs are still available. */
    SELECT
        cd.`Episodi`,
        cd.`Nhc`,
        cd.`data_index`,
        l.`variable_lab`,
        l.`ResultatNumericClean`,
        l.`DataPetició`,
        ROW_NUMBER() OVER (
            PARTITION BY cd.`Episodi`, cd.`Nhc`, cd.`data_index`, l.`variable_lab`
            ORDER BY l.`DataPetició` DESC, l.`ordre_resultat` DESC, l.`Id` DESC
        ) AS rn
    FROM daily_critical_care cd
    INNER JOIN clean_laboratory_events l
        ON cd.`Episodi` = l.`Episodi`
       AND cd.`Nhc` = l.`Nhc`
    WHERE cd.`Data_hora_alta_critics` IS NOT NULL
      AND cd.`Data_hora_alta_critics` >= cd.`data_index`
      AND cd.`Data_hora_alta_critics` < DATE_ADD(cd.`data_index`, INTERVAL 1 DAY)
      AND cd.`Temps_critics` > 24
      AND l.`variable_lab` IN ('creatinina', 'plaquetes', 'bilirubina_total')
      AND l.`ResultatNumericClean` IS NOT NULL
      AND l.`DataPetició` >= DATE_SUB(cd.`Data_hora_alta_critics`, INTERVAL 3 DAY)
      AND l.`DataPetició` < cd.`Data_hora_alta_critics`
),

pre_critical_return_labs AS (
    SELECT
        `Episodi`,
        `Nhc`,
        `data_index`,
        MAX(CASE WHEN `variable_lab` = 'creatinina' THEN `ResultatNumericClean` END) AS `creatinina_pre_retorn_critics_3d`,
        MAX(CASE WHEN `variable_lab` = 'creatinina' THEN `DataPetició` END) AS `data_creatinina_pre_retorn_critics_3d`,
        MAX(CASE WHEN `variable_lab` = 'plaquetes' THEN `ResultatNumericClean` END) AS `plaquetes_pre_retorn_critics_3d`,
        MAX(CASE WHEN `variable_lab` = 'plaquetes' THEN `DataPetició` END) AS `data_plaquetes_pre_retorn_critics_3d`,
        MAX(CASE WHEN `variable_lab` = 'bilirubina_total' THEN `ResultatNumericClean` END) AS `bilirubina_total_pre_retorn_critics_3d`,
        MAX(CASE WHEN `variable_lab` = 'bilirubina_total' THEN `DataPetició` END) AS `data_bilirubina_total_pre_retorn_critics_3d`
    FROM pre_critical_return_labs_ranked
    WHERE rn = 1
    GROUP BY `Episodi`, `Nhc`, `data_index`
),

daily_surgery AS (
    /* Daily and cumulative surgery exposure up to each data_index. Duration
       only uses plausible positive intervals from 1 minute to 24 hours. */
    SELECT
        d.`Episodi`,
        d.`Nhc`,
        d.`data_index`,
        MAX(
            CASE
                WHEN aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                THEN 1 ELSE 0
            END
        ) AS `Cirurgia`,
        MAX(
            CASE
                WHEN aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND (
                        (
                            aq.`DataHorafimovimentIQ` IS NOT NULL
                        AND aq.`DataHorafimovimentIQ` > d.`data_index`
                        )
                     OR (
                            aq.`DataHorafimovimentIQ` IS NULL
                        AND aq.`DataHoraIQ` >= d.`data_index`
                        )
                 )
                THEN 1 ELSE 0
            END
        ) AS `Cirurgia_dia`,
        MAX(
            CASE
                WHEN aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND (
                        aq.`IndicadorUrgentProgramat` = 'U'
                     OR UPPER(aq.`Tipuscirurgia`) LIKE '%URGENT%'
                 )
                THEN 1 ELSE 0
            END
        ) AS `Urgencia_cirurgia`,
        MAX(
            CASE
                WHEN aq.`DataHoraIQ` IS NOT NULL
                 AND aq.`DataHorafimovimentIQ` IS NOT NULL
                 AND aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND TIMESTAMPDIFF(MINUTE, aq.`DataHoraIQ`, aq.`DataHorafimovimentIQ`) BETWEEN 1 AND 1440
                THEN 1 ELSE 0
            END
        ) AS `Temps_cirurgia_disponible`,
        SUM(
            CASE
                WHEN aq.`DataHoraIQ` IS NOT NULL
                 AND aq.`DataHorafimovimentIQ` IS NOT NULL
                 AND aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND TIMESTAMPDIFF(MINUTE, aq.`DataHoraIQ`, aq.`DataHorafimovimentIQ`) BETWEEN 1 AND 1440
                 AND aq.`DataHorafimovimentIQ` > d.`data_index`
                THEN GREATEST(
                    TIMESTAMPDIFF(
                        MINUTE,
                        GREATEST(aq.`DataHoraIQ`, d.`data_index`),
                        LEAST(aq.`DataHorafimovimentIQ`, DATE_ADD(d.`data_index`, INTERVAL 1 DAY))
                    ),
                    0
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_cirurgia_dia`,
        SUM(
            CASE
                WHEN aq.`DataHoraIQ` IS NOT NULL
                 AND aq.`DataHorafimovimentIQ` IS NOT NULL
                 AND aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                 AND TIMESTAMPDIFF(MINUTE, aq.`DataHoraIQ`, aq.`DataHorafimovimentIQ`) BETWEEN 1 AND 1440
                THEN GREATEST(
                    TIMESTAMPDIFF(
                        MINUTE,
                        aq.`DataHoraIQ`,
                        CASE
                            WHEN aq.`DataHorafimovimentIQ` > DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                            THEN DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
                            ELSE aq.`DataHorafimovimentIQ`
                        END
                    ),
                    0
                ) / 60.0
                ELSE 0
            END
        ) AS `Temps_cirurgia`
    FROM model_days d
    LEFT JOIN normalized_surgery_activity aq
        ON d.`Episodi` = aq.`Episodi`
       AND aq.`DataHoraIQ` < DATE_ADD(d.`data_index`, INTERVAL 1 DAY)
    GROUP BY d.`Episodi`, d.`Nhc`, d.`data_index`
),

episode_invasive_devices AS (
    /* Episode-level invasive-device history from hospital and critical-care
       sources. Reliable device start/end dates are not available, so the flag
       is repeated across all model days for the episode. */
    SELECT
        x.`Episodi`,
        x.`Nhc`,
        MAX(x.`dispositiu_invasiu`) AS `dispositius_invasius_previs`
    FROM (
        SELECT
            b.`Episodi`,
            b.`Nhc`,
            CASE
                WHEN COALESCE(ct.`IndCatèterVenòsCentral`, 0) = 1
                  OR COALESCE(ct.`IndSondaVesicalTotes`, 0) = 1
                  OR COALESCE(ct.`IndSondaVesical`, 0) = 1
                  OR COALESCE(ct.`IndSondaSuprapúbica`, 0) = 1
                  OR COALESCE(ct.`IndSondaVesical3Llums`, 0) = 1
                THEN 1 ELSE 0
            END AS `dispositiu_invasiu`
        FROM base_hospitalization_cohort b
        INNER JOIN tab_dt_sepsis_cateters_001_ano ct
            ON b.`Episodi` = ct.`EpisodiSAP`

        UNION ALL

        SELECT
            b.`Episodi`,
            b.`Nhc`,
            CASE
                WHEN UPPER(cc.`Tipusintervenció`) LIKE '%CATÈTER VENÓS CENTRAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CATETER VENOS CENTRAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CATÈTER  PICCO%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CATETER  PICCO%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CAT. VESICAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CAT.VESICAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%SONDA VESICAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%SONDA SUPRAPÚBICA%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%SONDA SUPRAPUBICA%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CATÈTER URETRAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%CATETER URETRAL%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%TRAQUEOSTOMIA%'
                  OR UPPER(cc.`Tipusintervenció`) LIKE '%TUB ENDOTRAQUEAL%'
                THEN 1 ELSE 0
            END AS `dispositiu_invasiu`
        FROM base_hospitalization_cohort b
        INNER JOIN tab_dt_sepsis_cateters_critics_001_ano cc
            ON b.`Episodi` = cc.`NúmcasSAP`
    ) x
    GROUP BY x.`Episodi`, x.`Nhc`
)

SELECT
    /* ========================================================
       IDENTIFIERS AND TIME
    ======================================================== */
    d.`Episodi`,
    d.`Nhc`,
    d.`data_index`,
    d.`dia_relatiu`,

    /* ========================================================
       FIXED COHORT VARIABLES
    ======================================================== */
    b.`DataIngres`,
    b.`DataIniciUrgencies`,
    b.`DataAlta`,
    b.`Font_admissio` AS `font_admissio`,
    b.`Centre_origen` AS `centre_origen`,
    b.`Codi_servei_admissor` AS `codi_servei_admissor`,
    b.`Diagnostic_ingres` AS `diagnostic_ingres`,
    b.`Edat` AS `edat`,
    b.`Sexe` AS `sexe`,
    b.`Hospitalitzacio_recent_90d` AS `hospitalitzacio_recent_90d`,
    b.`Reingres_30d` AS `reingres_30d`,
    COALESCE(cd.`Passa_per_critics`, 0) AS `passa_per_critics`,
    COALESCE(cd.`En_critics_dia`, 0) AS `en_critics_dia`,
    COALESCE(cd.`Temps_critics_dia`, 0)
      + COALESCE(cqd.`Temps_cirurgia_dia`, 0) AS `temps_critics_dia`,
    COALESCE(cd.`Temps_critics`, 0)
      + COALESCE(cqd.`Temps_cirurgia`, 0) AS `temps_critics`,
    cd.`Data_hora_alta_critics` AS `data_hora_alta_critics`,
    COALESCE(cqd.`Cirurgia`, 0) AS `cirurgia`,
    COALESCE(cqd.`Urgencia_cirurgia`, 0) AS `urgencia_cirurgia`,
    COALESCE(cqd.`Temps_cirurgia_disponible`, 0) AS `temps_cirurgia_disponible`,
    CASE
        WHEN COALESCE(cqd.`Cirurgia`, 0) = 1
         AND COALESCE(cqd.`Temps_cirurgia_disponible`, 0) = 0
        THEN NULL
        ELSE COALESCE(cqd.`Temps_cirurgia`, 0)
    END AS `temps_cirurgia`,

    /* ========================================================
       COMORBIDITIES AND ADDITIONAL BASE VARIABLES
    ======================================================== */
    b.`COMORB_DIABETES_MELLITUS`,
    b.`COMORB_NEOPLASIA_SOLIDA`,
    b.`COMORB_NEOPLASIA_HEMATOLOGICA`,
    b.`COMORB_ENOLISME_SEVER`,
    b.`COMORB_CIRROSI_HEPATICA`,
    b.`COMORB_VIH_SIDA`,
    b.`COMORB_TRASPLANT_ORGAN_SOLID`,
    b.`COMORB_TRASPLANT_MOLL_OS`,
    b.`COMORB_AGAMMAGLOBULINEMIA`,
    b.`COMORB_HIPOGAMMAGLOBULINEMIA`,
    b.`COMORB_MALABSORTIVES`,
    b.`COMORB_MALNUTRICIO_SEVERA`,
    b.`COMORB_ASPLENIA`,
    b.`COMORB_ESPLENECTOMIA`,
    b.`COMORB_IRC_DIALISI`,
    b.`COMORB_NEUTROPENIA_GREU`,

    /* ========================================================
       VITAL SIGNS
    ======================================================== */
    cv.`SBP`,
    cv.`DBP`,
    cv.`TAM`,
    cv.`HR`,
    cv.`RESP`,
    cv.`O2SAT`,
    cv.`TEMP`,
    cv.`FIO2`,
    cv.`DIURESIS`,
    cv.`GLASGOW`,
    cv.`porta_o2`,
    COALESCE(cat.`dispositius_invasius_previs`, 0) AS `dispositius_invasius_previs`,

    /* ========================================================
       LABORATORY RECORDS
    ======================================================== */
    l.`ph_arterial`,
    l.`pao2_arterial`,
    l.`paco2_arterial`,
    l.`bicarbonat_arterial`,
    l.`exc_base_arterial`,
    l.`lactat_arterial`,
    l.`ph_venos`,
    l.`pao2_venos`,
    l.`paco2_venos`,
    l.`bicarbonat_venos`,
    l.`exc_base_venos`,
    l.`lactat_venos`,
    l.`hematocrit`,
    l.`hemoglobina`,
    l.`leucocits`,
    l.`pct_neutrofils`,
    l.`granulocits_immadurs`,
    l.`plaquetes`,
    lprc.`plaquetes_pre_retorn_critics_3d`,
    lprc.`data_plaquetes_pre_retorn_critics_3d`,
    l.`fibrinogen`,
    l.`temps_protrombina_pct`,
    l.`pcr`,
    l.`procalcitonina`,
    l.`glucosa`,
    l.`urea`,
    l.`creatinina`,
    lprc.`creatinina_pre_retorn_critics_3d`,
    lprc.`data_creatinina_pre_retorn_critics_3d`,
    l.`bilirubina_total`,
    lprc.`bilirubina_total_pre_retorn_critics_3d`,
    lprc.`data_bilirubina_total_pre_retorn_critics_3d`,
    l.`got_ast`,
    l.`albumina`,
    l.`proteines_totals`,
    l.`troponina`,
    COALESCE(l.`hemocultiu_positiu`, 0) AS `hemocultiu_positiu`,
    l.`hemocultiu_positiu_data_extraccio`,
    l.`hemocultiu_germen`,
    l.`hemocultiu_temps_positivitat_h`,
    l.`urocultiu_resultat`,
    l.`aspirat_traqueal_germen`,
    l.`broncoaspirat_germen`,
    l.`bal_germen`,
    COALESCE(l.`ag_pneumococ`, 0) AS `ag_pneumococ`,
    COALESCE(l.`ag_legionella`, 0) AS `ag_legionella`,
    COALESCE(l.`colonitzacio_previa_blee`, 0) AS `colonitzacio_previa_blee`,
    COALESCE(l.`colonitzacio_previa_cre`, 0) AS `colonitzacio_previa_cre`,
    COALESCE(l.`colonitzacio_previa_mrsa`, 0) AS `colonitzacio_previa_mrsa`,
    COALESCE(l.`colonitzacio_previa_vre`, 0) AS `colonitzacio_previa_vre`,
    COALESCE(l.`cultiu_positiu_previ_90d`, 0) AS `cultiu_positiu_previ_90d`,

    /* ========================================================
       PHARMACY FEATURES
    ======================================================== */
    COALESCE(f.`vasopressor_dobutamina`, 0) AS `vasopressor_dobutamina`,
    COALESCE(f.`vasopressor_dopamina`, 0) AS `vasopressor_dopamina`,
    COALESCE(f.`vasopressor_noradrenalina`, 0) AS `vasopressor_noradrenalina`,
    COALESCE(f.`vasopressor_adrenalina`, 0) AS `vasopressor_adrenalina`,
    COALESCE(f.`antibiotic`, 0) AS `antibiotic`,
    COALESCE(f.`atb_duracio`, 0) AS `atb_duracio`,
    COALESCE(f.`antibiotics_previs_90d`, 0) AS `antibiotics_previs_90d`,
    COALESCE(f.`vasopressor_qualsevol`, 0) AS `vasopressor_qualsevol`,
    COALESCE(f.`vasopressor_multiple`, 0) AS `vasopressor_multiple`

FROM model_days d
INNER JOIN episodes_with_24h_vital_data e24
    ON d.`Episodi` = e24.`Episodi`
   AND d.`Nhc` = e24.`Nhc`
INNER JOIN episodes_with_creatinine_or_platelets els
    ON d.`Episodi` = els.`Episodi`
   AND d.`Nhc` = els.`Nhc`
INNER JOIN base_hospitalization_cohort b
    ON d.`Episodi` = b.`Episodi`
   AND d.`Nhc` = b.`Nhc`
LEFT JOIN daily_critical_care cd
    ON d.`Episodi` = cd.`Episodi`
   AND d.`Nhc` = cd.`Nhc`
   AND d.`data_index` = cd.`data_index`
LEFT JOIN pre_critical_return_labs lprc
    ON d.`Episodi` = lprc.`Episodi`
   AND d.`Nhc` = lprc.`Nhc`
   AND d.`data_index` = lprc.`data_index`
LEFT JOIN daily_surgery cqd
    ON d.`Episodi` = cqd.`Episodi`
   AND d.`Nhc` = cqd.`Nhc`
   AND d.`data_index` = cqd.`data_index`
LEFT JOIN daily_vital_signs cv
    ON d.`Episodi` = cv.`Episodi`
   AND d.`Nhc` = cv.`Nhc`
   AND d.`data_index` = cv.`data_index`
LEFT JOIN episode_invasive_devices cat
    ON d.`Episodi` = cat.`Episodi`
   AND d.`Nhc` = cat.`Nhc`
LEFT JOIN daily_laboratory_records l
    ON d.`Episodi` = l.`Episodi`
   AND d.`Nhc` = l.`Nhc`
   AND d.`data_index` = l.`data_index`
LEFT JOIN daily_pharmacy_features f
    ON d.`Episodi` = f.`Episodi`
   AND d.`Nhc` = f.`Nhc`
   AND d.`data_index` = f.`Data_dia`;

CREATE UNIQUE INDEX idx_dsm_episode_patient_day
    ON daily_sepsis_model (`Episodi`, `Nhc`, `data_index`);

CREATE INDEX idx_dsm_episode_day
    ON daily_sepsis_model (`Episodi`, `data_index`);

CREATE INDEX idx_dsm_patient_day
    ON daily_sepsis_model (`Nhc`, `data_index`);
