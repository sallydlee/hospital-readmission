import pytest

from app import application


@pytest.fixture
def client():
    application.config["TESTING"] = True
    with application.test_client() as test_client:
        yield test_client


class TestHealth:
    def test_health_returns_200_when_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"


class TestPredict:
    def test_rejects_non_json_body(self, client):
        resp = client.post("/predict", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_rejects_missing_fields(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_rejects_excluded_discharge_code(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload(discharge_disposition_id=11))
        assert resp.status_code == 400

    def test_valid_payload_returns_prediction(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload())
        assert resp.status_code == 200
        body = resp.get_json()
        assert "readmission_risk_probability" in body
        assert 0.0 <= body["readmission_risk_probability"] <= 1.0
        assert isinstance(body["high_risk"], bool)
        assert body["threshold_used"] == pytest.approx(0.3)


class TestMiscEndpoints:
    def test_unknown_route_returns_404_with_available_endpoints(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        body = resp.get_json()
        assert "/health" in body["available_endpoints"]
        assert "/predict" in body["available_endpoints"]

    def test_metrics_endpoint_exposes_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/plain")
