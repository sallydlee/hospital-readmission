"""
LocalPredictor —
loads model.joblib from disk and scores requests in-process
"""
import logging

import joblib
import pandas as pd

from .base import Predictor

logger = logging.getLogger(__name__)


class LocalPredictor(Predictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
            logger.info(f"LocalPredictor: loaded model from {model_path}")
        except Exception:
            logger.exception(f"LocalPredictor: failed to load model from {model_path}")
            self.model = None

    def predict_proba(self, features: dict) -> float:
        if self.model is None:
            raise RuntimeError(f"model not loaded from {self.model_path}")
        X = pd.DataFrame([features])
        return float(self.model.predict_proba(X)[0, 1])

    def is_healthy(self) -> bool:
        return self.model is not None
