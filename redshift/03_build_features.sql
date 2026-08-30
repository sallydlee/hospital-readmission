-- !! Keep this file's logic in sync with src/preprocessing.py
-- and api/feature_engineering.py. There is currently no
-- automated check to see if the SQL and Python versions agree !!

CREATE OR REPLACE FUNCTION bucket_diagnosis(code VARCHAR(16))
RETURNS VARCHAR(16)
STABLE
AS $$
    SELECT CASE
        WHEN code IS NULL OR code = '' THEN 'missing'
        WHEN code LIKE '250%' THEN 'diabetes'
        WHEN code LIKE 'V%' OR code LIKE 'E%' THEN 'other'
        WHEN TRY_CAST(code AS FLOAT8) IS NULL THEN 'other'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 390 AND 459 OR TRY_CAST(code AS FLOAT8) = 785 THEN 'circulatory'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 460 AND 519 OR TRY_CAST(code AS FLOAT8) = 786 THEN 'respiratory'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 520 AND 579 OR TRY_CAST(code AS FLOAT8) = 787 THEN 'digestive'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 800 AND 999 THEN 'injury'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 710 AND 739 THEN 'musculoskeletal'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 580 AND 629 OR TRY_CAST(code AS FLOAT8) = 788 THEN 'genitourinary'
        WHEN TRY_CAST(code AS FLOAT8) BETWEEN 140 AND 239 THEN 'neoplasm'
        ELSE 'other'
    END
$$ LANGUAGE sql;

DROP TABLE IF EXISTS features;

CREATE TABLE features AS

WITH first_encounter AS (
    -- keep_first_encounter_per_patient(): lowest encounter_id per
    -- patient_nbr is the standard proxy for "first" chronologically,
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id ASC) AS rn
    FROM raw_diabetic_data
),

clean_rows AS (
    -- exclude_terminal_and_bad_quality_rows(): drop deceased/hospice
    -- discharges and known data-quality codes (18=NULL, 25=Not Mapped,
    -- 26=Unknown/Invalid)
    SELECT *
    FROM first_encounter
    WHERE rn = 1
      AND discharge_disposition_id NOT IN (11, 13, 14, 18, 19, 20, 21, 25, 26)
)

SELECT
    -- build_target()
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS readmitted_30d,

    -- handle_remaining_missingness()
    COALESCE(race, 'unknown') AS race,
    gender,

    -- handle_lab_result_missingness()
    COALESCE(max_glu_serum, 'not_tested') AS max_glu_serum,
    COALESCE(a1cresult, 'not_tested') AS "A1Cresult",

    -- group_diagnosis_codes()
    bucket_diagnosis(diag_1) AS diag_1_group,
    bucket_diagnosis(diag_2) AS diag_2_group,
    bucket_diagnosis(diag_3) AS diag_3_group,

    -- bucket_age(): AGE_BUCKETS collapses the 10 UCI age bins into
    -- young / adult / mid_senior
    CASE
        WHEN age IN ('[0-10)', '[10-20)') THEN 'young'
        WHEN age = '[20-30)' THEN 'adult'
        WHEN age IN ('[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)') THEN 'mid_senior'
        ELSE NULL  -- should never happen for valid UCI data; surfaces as a NULL age_group if it does
    END AS age_group,

    -- bucket_payer_code(): PAYER_CODE_BUCKETS, defaulting missing codes to "UN" -> "other"
    CASE COALESCE(payer_code, 'UN')
        WHEN 'MC' THEN 'medicare' WHEN 'MP' THEN 'medicare'
        WHEN 'MD' THEN 'medicaid' WHEN 'DM' THEN 'medicaid'
        WHEN 'SP' THEN 'self_pay'
        WHEN 'HM' THEN 'private' WHEN 'BC' THEN 'private' WHEN 'CP' THEN 'private'
        WHEN 'PO' THEN 'private' WHEN 'CM' THEN 'private' WHEN 'SI' THEN 'private'
        WHEN 'UN' THEN 'other' WHEN 'OG' THEN 'other' WHEN 'OT' THEN 'other'
        WHEN 'CH' THEN 'other' WHEN 'WC' THEN 'other' WHEN 'FR' THEN 'other'
        ELSE 'other'
    END AS payer_group,

    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,

    -- add_medication_summary(): count of all 23 original medication
    -- columns (both the near-constant ones dropped below, and the
    -- remaining 8) that are anything other than "No"
    (CASE WHEN metformin <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN repaglinide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN nateglinide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN chlorpropamide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN glimepiride <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN acetohexamide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN glipizide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN glyburide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN tolbutamide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN pioglitazone <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN rosiglitazone <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN acarbose <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN miglitol <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN troglitazone <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN tolazamide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN examide <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN citoglipton <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN insulin <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN "glyburide-metformin" <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN "glipizide-metformin" <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN "glimepiride-pioglitazone" <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN "metformin-rosiglitazone" <> 'No' THEN 1 ELSE 0 END) +
    (CASE WHEN "metformin-pioglitazone" <> 'No' THEN 1 ELSE 0 END)
        AS num_meds_administered,

    -- binarize_remaining_medications(): only the 8 REMAINING_MED_COLS
    -- (the other 15 are dropped below as near-constant)
    CASE WHEN metformin IN ('Up', 'Down') THEN 1 ELSE 0 END AS metformin,
    CASE WHEN repaglinide IN ('Up', 'Down') THEN 1 ELSE 0 END AS repaglinide,
    CASE WHEN glimepiride IN ('Up', 'Down') THEN 1 ELSE 0 END AS glimepiride,
    CASE WHEN glipizide IN ('Up', 'Down') THEN 1 ELSE 0 END AS glipizide,
    CASE WHEN glyburide IN ('Up', 'Down') THEN 1 ELSE 0 END AS glyburide,
    CASE WHEN pioglitazone IN ('Up', 'Down') THEN 1 ELSE 0 END AS pioglitazone,
    CASE WHEN rosiglitazone IN ('Up', 'Down') THEN 1 ELSE 0 END AS rosiglitazone,
    CASE WHEN insulin IN ('Up', 'Down') THEN 1 ELSE 0 END AS insulin,

    -- binarize_change() / binarize_diabetes_med()
    CASE WHEN change = 'Ch' THEN 1 ELSE 0 END AS change,
    CASE WHEN diabetesmed = 'Yes' THEN 1 ELSE 0 END AS "diabetesMed"

FROM clean_rows;
