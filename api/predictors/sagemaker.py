"""
SageMakerPredictor — scores requests by invoking a live SageMaker
real-time endpoint over the network, instead of loading a model
in-process.

This does NOT create or manage the endpoint. The endpoint must
already exist. If SAGEMAKER_ENDPOINT_NAME points at an endpoint
that doesn't exist or isn't InService, every call here will raise,
and is_healthy() will return False.

Requires `boto3` and AWS credentials with sagemaker:InvokeEndpoint
(and sagemaker:DescribeEndpoint for health checks) on that endpoint's
ARN.
"""
import json
import logging

from .base import Predictor

logger = logging.getLogger(__name__)


class SageMakerPredictor(Predictor):
    def __init__(self, endpoint_name: str):
        import boto3  

        self.endpoint_name = endpoint_name
        self.runtime = boto3.client("sagemaker-runtime")
        self.sagemaker = boto3.client("sagemaker")
        logger.info(f"SageMakerPredictor: configured for endpoint '{endpoint_name}'")

    def predict_proba(self, features: dict) -> float:
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(features),
        )
        body = json.loads(response["Body"].read())
        return float(body["probability"])

    def is_healthy(self) -> bool:
        try:
            status = self.sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            return status["EndpointStatus"] == "InService"
        except Exception:
            logger.exception(f"SageMakerPredictor: health check failed for '{self.endpoint_name}'")
            return False
