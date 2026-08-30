"""
Tests for the predictors/ package: the factory's backend selection,
LocalPredictor against the real bundled model.joblib, and
SageMakerPredictor against a mocked boto3 client (no real AWS calls,
no AWS credentials required to run this suite).
"""
import json
import os

import pytest

from predictors import get_predictor
from predictors.local import LocalPredictor


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib")


class TestLocalPredictor:
    def test_loads_real_bundled_model_and_scores(self, valid_payload):
        from feature_engineering import transform_record

        predictor = LocalPredictor(MODEL_PATH)
        assert predictor.is_healthy()

        features = transform_record(valid_payload())
        probability = predictor.predict_proba(features)
        assert 0.0 <= probability <= 1.0

    def test_unhealthy_when_model_path_is_bad(self):
        predictor = LocalPredictor("does/not/exist.joblib")
        assert not predictor.is_healthy()
        with pytest.raises(RuntimeError):
            predictor.predict_proba({})


class TestSageMakerPredictor:
    def test_predict_proba_calls_invoke_endpoint(self, monkeypatch):
        from predictors.sagemaker import SageMakerPredictor

        class FakeBody:
            def read(self):
                return json.dumps({"probability": 0.42}).encode()

        class FakeRuntimeClient:
            def invoke_endpoint(self, EndpointName, ContentType, Body):
                assert EndpointName == "readmission-risk-endpoint"
                assert ContentType == "application/json"
                assert json.loads(Body) == {"foo": 1}
                return {"Body": FakeBody()}

        class FakeSageMakerClient:
            def describe_endpoint(self, EndpointName):
                return {"EndpointStatus": "InService"}

        def fake_boto3_client(service_name):
            return FakeRuntimeClient() if service_name == "sagemaker-runtime" else FakeSageMakerClient()

        monkeypatch.setattr("boto3.client", fake_boto3_client)

        predictor = SageMakerPredictor("readmission-risk-endpoint")
        assert predictor.predict_proba({"foo": 1}) == pytest.approx(0.42)
        assert predictor.is_healthy() is True

    def test_is_healthy_false_when_endpoint_not_in_service(self, monkeypatch):
        from predictors.sagemaker import SageMakerPredictor

        class FakeSageMakerClient:
            def describe_endpoint(self, EndpointName):
                return {"EndpointStatus": "Creating"}

        monkeypatch.setattr(
            "boto3.client",
            lambda service_name: FakeSageMakerClient(),
        )

        predictor = SageMakerPredictor("readmission-risk-endpoint")
        assert predictor.is_healthy() is False


class TestGetPredictorFactory:
    def test_defaults_to_local(self, monkeypatch):
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        predictor = get_predictor()
        assert isinstance(predictor, LocalPredictor)

    def test_explicit_local(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "local")
        predictor = get_predictor()
        assert isinstance(predictor, LocalPredictor)

    def test_sagemaker_requires_endpoint_name(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "sagemaker")
        monkeypatch.delenv("SAGEMAKER_ENDPOINT_NAME", raising=False)
        with pytest.raises(RuntimeError, match="SAGEMAKER_ENDPOINT_NAME"):
            get_predictor()

    def test_sagemaker_backend_selected(self, monkeypatch):
        from predictors.sagemaker import SageMakerPredictor

        monkeypatch.setenv("MODEL_BACKEND", "sagemaker")
        monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "readmission-risk-endpoint")
        monkeypatch.setattr("boto3.client", lambda service_name: object())

        predictor = get_predictor()
        assert isinstance(predictor, SageMakerPredictor)

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "carrier_pigeon")
        with pytest.raises(RuntimeError, match="Unknown MODEL_BACKEND"):
            get_predictor()
