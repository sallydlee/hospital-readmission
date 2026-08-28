import logging
import os
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from feature_engineering import InvalidRequestError, transform_record, validate_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("model", "model.joblib"))

# Imbalanced target (~9% positive in training). 0.3 threshold
# catches most true readmissions at the cost of more false positives
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.3"))

application = Flask(__name__)

# multiprocess.MultiProcessCollector (used in /metrics below) requires this
# directory to exist, whether running under gunicorn with multiple workers
# or via `python app.py` locally with just one.
PROMETHEUS_MULTIPROC_DIR = os.environ.setdefault(
    "PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc"
)
os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "readmission_api_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "readmission_api_request_latency_seconds",
    "Time spent handling a request",
    ["method", "endpoint"],
)

# Business metric: how predictions are distributed between high/low risk
PREDICTION_OUTCOME = Counter(
    "readmission_api_prediction_outcome_total",
    "Count of predictions by risk category",
    ["risk_level"],
)

# Full distribution of the raw predicted probability
PREDICTION_PROBABILITY = Histogram(
    "readmission_api_prediction_probability",
    "Distribution of predicted readmission risk probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# 1 if the model loaded successfully at startup, 0 if it didn't\
MODEL_LOADED = Gauge(
    "readmission_api_model_loaded",
    "1 if the ML model loaded successfully at startup, 0 otherwise",
    multiprocess_mode="min",
)


try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    MODEL_LOADED.set(1)
except Exception:
    logger.exception(f"Failed to load model from {MODEL_PATH}")
    model = None
    MODEL_LOADED.set(0)


@application.before_request
def _start_timer():
    request._start_time = time.time()


@application.after_request
def _record_request_metrics(response):
    endpoint = request.url_rule.rule if request.url_rule else "unmatched"
    latency = time.time() - getattr(request, "_start_time", time.time())

    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(latency)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, http_status=response.status_code
    ).inc()
    return response


@application.get("/health")
def health():
    if model is None:
        return jsonify(status="unhealthy", reason="model not loaded"), 503
    return jsonify(status="healthy"), 200


@application.post("/predict")
def predict():
    if model is None:
        return jsonify(error="model not loaded"), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="request body must be valid JSON"), 400

    try:
        validate_request(payload)
        features = transform_record(payload)
    except InvalidRequestError as e:
        return jsonify(error=str(e)), 400
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"invalid input: {e}"), 400

    try:
        X = pd.DataFrame([features])
        probability = float(model.predict_proba(X)[0, 1])
    except Exception:
        logger.exception("Prediction failed")
        return jsonify(error="internal error while scoring the request"), 500

    high_risk = probability >= PREDICTION_THRESHOLD

    PREDICTION_PROBABILITY.observe(probability)
    PREDICTION_OUTCOME.labels(risk_level="high" if high_risk else "low").inc()

    return jsonify(
        readmission_risk_probability=round(probability, 4),
        high_risk=high_risk,
        threshold_used=PREDICTION_THRESHOLD,
    ), 200


@application.errorhandler(404)
def not_found(e):
    return jsonify(error="not found", available_endpoints=["/health", "/predict"]), 404


@application.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
