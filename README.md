# Hospital Readmission Risk Prediction API

A REST API that predicts whether a diabetic patient is at risk of
hospital readmission within 30 days, trained on the UCI Diabetes
130-US Hospitals dataset. The API supports two interchangeable
scoring backends, chosen with a single environment variable.

## Running the Local Backend

```bash
cd api
pip install -r requirements-dev.txt
python app.py
# or: gunicorn --bind 0.0.0.0:8000 -c gunicorn.conf.py app:application
```

```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
     -d @../demo/payload.json
```

Run the tests:
```bash
cd api
pytest tests/ -v
```

Run via Docker instead:
```bash
cd api
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
```

### Retraining the local model

```bash
cd src
python preprocessing.py  
python train.py            
```
Copy the resulting `model/model.joblib` into `api/model/model.joblib`
to have the running app pick up the new version (the app reads
`MODEL_PATH`, default `api/model/model.joblib`).

## Running the AWS Backend (Redshift + SageMaker)

This backend needs AWS resources provisioned before the app can
use them.

**First time setup:**

1. **Load and transform data in Redshift.**
   ```bash
   aws s3 cp data/diabetic_data.csv s3://<your-bucket>/raw/diabetic_data.csv
   ```
   Then run, in the following order:
   `redshift/01_create_raw_table.sql` → `02_load_from_s3.sql` →
   `03_build_features.sql` → `04_unload_to_s3.sql`.
   (Fill in `<your-bucket>` and `<your-redshift-iam-role-arn>` in the
   `.sql` files first)

2. **Train on SageMaker.** From a notebook, script, or CI job with the
   SageMaker SDK installed (`pip install -r sagemaker/requirements.txt`):
   ```python
   from sagemaker.sklearn.estimator import SKLearn

   estimator = SKLearn(
       entry_point="train_entry.py",
       source_dir="sagemaker",
       dependencies=["src"],
       role="<sagemaker-execution-role-arn>",
       instance_type="ml.m5.large",
       framework_version="1.2-1",
       py_version="py3",
   )
   estimator.fit({"train": "s3://<your-bucket>/redshift-unload/features/"})
   ```

3. **Deploy the trained model to a real-time endpoint:**
   ```python
   from sagemaker.sklearn.model import SKLearnModel

   model = SKLearnModel(
       model_data=estimator.model_data,
       role="<sagemaker-execution-role-arn>",
       entry_point="inference.py",
       source_dir="sagemaker",
       framework_version="1.2-1",
   )
   predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large")
   print(predictor.endpoint_name) 
   ```

4. **Point the Flask app at the endpoint:**
   ```bash
   export MODEL_BACKEND=sagemaker
   export SAGEMAKER_ENDPOINT_NAME=<endpoint name from step 3>
   cd api && python app.py
   ```

**On Elastic Beanstalk**, set the same two environment variables on
the environment's configuration (console → Configuration → Software,
or `eb setenv MODEL_BACKEND=sagemaker SAGEMAKER_ENDPOINT_NAME=...`)

### Verifying the two backends agree

Send the same payload to both and confirm they return the same probability:
```bash
MODEL_BACKEND=local python app.py &        # terminal 1, port 5000
MODEL_BACKEND=sagemaker SAGEMAKER_ENDPOINT_NAME=... python app.py &  # terminal 2, different port
curl -s -X POST http://localhost:5000/predict -d @demo/payload.json | jq .readmission_risk_probability
curl -s -X POST http://localhost:5001/predict -d @demo/payload.json | jq .readmission_risk_probability
```

## Monitoring
```bash
cd monitoring
docker compose up --build
```
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (`admin`/`admin`)

The `readmission_api_model_loaded` gauge reflects whichever backend is
active. `0` means either the local model file failed to load, or the
SageMaker endpoint isn't `InService`.

## CI/CD

`.github/workflows/deploy.yml` runs `pytest` (against the local
backend), builds a deployment package, and deploys it to the `dev` or
`prod` Elastic Beanstalk environment depending on branch. It does not
currently run the SageMaker training pipeline.
