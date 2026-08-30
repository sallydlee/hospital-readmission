-- Exports the `features` table (built by 03_build_features.sql) to S3
-- as CSV, so a SageMaker training job can read it via an S3 input
-- channel


UNLOAD ('SELECT * FROM features')
TO 's3://<your-bucket>/redshift-unload/features/features_'
IAM_ROLE '<your-redshift-iam-role-arn>'
CSV
HEADER
PARALLEL OFF;
