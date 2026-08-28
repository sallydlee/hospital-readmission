import pytest

from feature_engineering import InvalidRequestError, transform_record, validate_request


class TestValidateRequest:
    def test_rejects_missing_fields(self):
        with pytest.raises(InvalidRequestError):
            validate_request({})

    def test_rejects_non_integer_discharge_id(self, valid_payload):
        with pytest.raises(InvalidRequestError):
            validate_request(valid_payload(discharge_disposition_id="not-a-number"))

    @pytest.mark.parametrize("code", [11, 13, 19, 20])
    def test_rejects_excluded_discharge_codes(self, valid_payload, code):
        with pytest.raises(InvalidRequestError):
            validate_request(valid_payload(discharge_disposition_id=code))

    def test_accepts_well_formed_payload(self, valid_payload):
        # Should not raise.
        validate_request(valid_payload())


class TestTransformRecord:
    def test_maps_age_to_bucket(self, valid_payload):
        features = transform_record(valid_payload(age="[60-70)"))
        assert features["age_group"] == "mid_senior"

    def test_raises_on_unknown_age_bucket(self, valid_payload):
        with pytest.raises(InvalidRequestError):
            transform_record(valid_payload(age="[100-110)"))

    def test_unknown_payer_code_falls_back_to_other(self, valid_payload):
        features = transform_record(valid_payload(payer_code="ZZ"))
        assert features["payer_group"] == "other"

    def test_missing_race_defaults_to_unknown(self, valid_payload):
        features = transform_record(valid_payload(race="?"))
        assert features["race"] == "unknown"

    @pytest.mark.parametrize(
        "code,expected_group",
        [
            ("250.02", "diabetes"),
            ("410", "circulatory"),
            ("786", "respiratory"),
            ("V27", "other"),
            ("", "missing"),
        ],
    )
    def test_diagnosis_bucketing(self, valid_payload, code, expected_group):
        features = transform_record(valid_payload(diag_1=code))
        assert features["diag_1_group"] == expected_group

    def test_change_and_diabetes_med_are_binary_encoded(self, valid_payload):
        features = transform_record(valid_payload(change="Ch", diabetesMed="Yes"))
        assert features["change"] == 1
        assert features["diabetesMed"] == 1

    def test_medication_dose_change_counts_as_administered(self, valid_payload):
        features = transform_record(valid_payload(insulin="Up", metformin="Down"))
        assert features["insulin"] == 1
        assert features["metformin"] == 1
