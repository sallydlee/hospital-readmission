-- Bulk-loads data/diabetic_data.csv (uploaded to S3 first) into
-- raw_diabetic_data.
--
-- Prerequisites:
--   1. Upload data/diabetic_data.csv to an S3 bucket, e.g.:
--        aws s3 cp data/diabetic_data.csv s3://<your-bucket>/raw/diabetic_data.csv
--   2. Redshift needs an IAM role attached to the cluster/workgroup
--      with s3:GetObject on that bucket 
--   3. The CSV uses "?" for missing values (see src/preprocessing.py's
--      na_values="?") — NULL AS '?' below reproduces that.

COPY raw_diabetic_data
FROM 's3://<your-bucket>/raw/diabetic_data.csv'
IAM_ROLE '<your-redshift-iam-role-arn>'
CSV
IGNOREHEADER 1
NULL AS '?'
REGION 'us-west-2';
