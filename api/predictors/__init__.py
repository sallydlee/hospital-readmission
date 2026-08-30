"""
get_predictor() is the single place app.py asks for a scoring backend.
Which backend it gets back is controlled entirely by the MODEL_BACKEND
environment variable — nothing else in the app needs to change to
switch between them.

    MODEL_BACKEND=local (default)
        -> LocalPredictor, loads MODEL_PATH (default model/model.joblib)
           with joblib and scores in-process. No AWS needed.

    MODEL_BACKEND=sagemaker
        -> SageMakerPredictor, calls the live endpoint named by
           SAGEMAKER_ENDPOINT_NAME (required in this mode). The
           endpoint must already be deployed — see sagemaker/README.md.

See the top-level README for how to set these per environment
(local shell, Docker, Elastic Beanstalk).
"""
import os

from .base import Predictor
from .local import LocalPredictor


def get_predictor() -> Predictor:
    backend = os.environ.get("MODEL_BACKEND", "local").lower()

    if backend == "sagemaker":
        from .sagemaker import SageMakerPredictor  # lazy import: boto3 not required otherwise

        endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
        if not endpoint_name:
            raise RuntimeError(
                "MODEL_BACKEND=sagemaker requires SAGEMAKER_ENDPOINT_NAME to be set."
            )
        return SageMakerPredictor(endpoint_name)

    if backend != "local":
        raise RuntimeError(
            f"Unknown MODEL_BACKEND={backend!r}. Expected 'local' or 'sagemaker'."
        )

    model_path = os.environ.get("MODEL_PATH", os.path.join("model", "model.joblib"))
    return LocalPredictor(model_path)
