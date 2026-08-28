"""
feature_engineering.py — the record-level subset of preprocessing.py's logic
"""

EXCLUDE_DISCHARGE_CODES = [11, 13, 14, 18, 19, 20, 21, 25, 26]

NEAR_CONSTANT_MED_COLS = [
    "acetohexamide", "tolbutamide", "troglitazone", "examide", "citoglipton",
    "tolazamide", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "acarbose",
    "chlorpropamide", "miglitol", "nateglinide", "glyburide-metformin",
]

REMAINING_MED_COLS = [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
]

ALL_MED_COLS = NEAR_CONSTANT_MED_COLS + REMAINING_MED_COLS

PAYER_CODE_BUCKETS = {
    "MC": "medicare", "MP": "medicare",
    "MD": "medicaid", "DM": "medicaid",
    "SP": "self_pay",
    "HM": "private", "BC": "private", "CP": "private",
    "PO": "private", "CM": "private", "SI": "private",
    "UN": "other", "OG": "other", "OT": "other",
    "CH": "other", "WC": "other", "FR": "other",
}

AGE_BUCKETS = {
    "[0-10)": "young", "[10-20)": "young",
    "[20-30)": "adult",
    "[30-40)": "mid_senior", "[40-50)": "mid_senior", "[50-60)": "mid_senior",
    "[60-70)": "mid_senior", "[70-80)": "mid_senior", "[80-90)": "mid_senior",
    "[90-100)": "mid_senior",
}


REQUIRED_FIELDS = [
    "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
] + ALL_MED_COLS


class InvalidRequestError(ValueError):
    """Raised for a well-formed JSON body that fails domain validation
    (missing fields, un-predictable discharge code, etc). Caught in
    application.py and turned into a 400, distinct from unexpected 500s."""


def _bucket_diagnosis(code) -> str:
    if code is None or code == "":
        return "missing"
    code = str(code)
    if code.startswith("250"):
        return "diabetes"
    if code.startswith(("V", "E")):
        return "other"
    try:
        n = float(code)
    except ValueError:
        return "other"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 800 <= n <= 999:
        return "injury"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasm"
    return "other"


def validate_request(payload: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise InvalidRequestError(f"Missing required fields: {missing}")

    try:
        discharge_id = int(payload["discharge_disposition_id"])
    except (TypeError, ValueError):
        raise InvalidRequestError("discharge_disposition_id must be an integer")

    if discharge_id in EXCLUDE_DISCHARGE_CODES:
        raise InvalidRequestError(
            f"discharge_disposition_id={discharge_id} indicates the patient "
            "died, entered hospice, or has an unusable discharge code — "
            "the model was not trained on these cases and cannot score them."
        )


def transform_record(payload: dict) -> dict:
    """
    Turn a raw, validated request payload into the exact feature set
    train.py's pipeline expects (CATEGORICAL_COLS + NUMERIC_COLS).
    Call validate_request(payload) first.
    """
    race = payload.get("race") or "unknown"
    if race in (None, "", "?"):
        race = "unknown"

    max_glu_serum = payload.get("max_glu_serum") or "not_tested"
    if max_glu_serum in (None, "", "None"):
        max_glu_serum = "not_tested"

    a1c_result = payload.get("A1Cresult") or "not_tested"
    if a1c_result in (None, "", "None"):
        a1c_result = "not_tested"

    payer_code = payload.get("payer_code") or "UN"
    payer_group = PAYER_CODE_BUCKETS.get(payer_code, "other")

    age_group = AGE_BUCKETS.get(payload["age"])
    if age_group is None:
        raise InvalidRequestError(
            f"age={payload['age']!r} is not one of the expected UCI age bins, "
            f"e.g. '[70-80)'. Expected one of: {sorted(AGE_BUCKETS)}"
        )


    num_meds_administered = sum(
        1 for col in ALL_MED_COLS if payload.get(col, "No") != "No"
    )

    features = {
        "race": race,
        "gender": payload["gender"],
        "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c_result,
        "diag_1_group": _bucket_diagnosis(payload.get("diag_1")),
        "diag_2_group": _bucket_diagnosis(payload.get("diag_2")),
        "diag_3_group": _bucket_diagnosis(payload.get("diag_3")),
        "age_group": age_group,
        "payer_group": payer_group,
        "admission_type_id": int(payload["admission_type_id"]),
        "discharge_disposition_id": int(payload["discharge_disposition_id"]),
        "admission_source_id": int(payload["admission_source_id"]),
        "time_in_hospital": int(payload["time_in_hospital"]),
        "num_lab_procedures": int(payload["num_lab_procedures"]),
        "num_procedures": int(payload["num_procedures"]),
        "num_medications": int(payload["num_medications"]),
        "number_outpatient": int(payload["number_outpatient"]),
        "number_emergency": int(payload["number_emergency"]),
        "number_inpatient": int(payload["number_inpatient"]),
        "number_diagnoses": int(payload["number_diagnoses"]),
        "num_meds_administered": num_meds_administered,
        "change": 1 if payload["change"] == "Ch" else 0,
        "diabetesMed": 1 if payload["diabetesMed"] == "Yes" else 0,
    }

    for col in REMAINING_MED_COLS:
        features[col] = 1 if payload.get(col) in ("Up", "Down") else 0

    return features
