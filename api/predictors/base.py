from abc import ABC, abstractmethod


class Predictor(ABC):
    @abstractmethod
    def predict_proba(self, features: dict) -> float:
        """Return the model's predicted probability of readmission
        within 30 days, given the output of
        feature_engineering.transform_record)."""
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if this backend is ready to serve predictions.
        Used by /health and the readmission_api_model_loaded gauge."""
        raise NotImplementedError
