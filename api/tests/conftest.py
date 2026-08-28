import os
import sys

import pytest

API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Let `import app` / `import feature_engineering` resolve regardless of
# where pytest is invoked from.
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

# app.py loads the model from a path relative to the current working
# directory (same assumption the Dockerfile's WORKDIR /app makes), so
# tests need to run with api/ as cwd.
os.chdir(API_DIR)


@pytest.fixture
def valid_payload():
    """Factory fixture returning a minimal, valid /predict payload.

    Usage: valid_payload() or valid_payload(discharge_disposition_id=11)
    """
    from feature_engineering import REQUIRED_FIELDS

    def _make(**overrides):
        payload = {field: "No" for field in REQUIRED_FIELDS}
        payload.update(
            {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[60-70)",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 1,
                "time_in_hospital": 3,
                "payer_code": "MC",
                "num_lab_procedures": 41,
                "num_procedures": 1,
                "num_medications": 15,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 1,
                "diag_1": "250",
                "diag_2": "401",
                "diag_3": "428",
                "number_diagnoses": 7,
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "change": "No",
                "diabetesMed": "Yes",
                "insulin": "Up",
            }
        )
        payload.update(overrides)
        return payload

    return _make
