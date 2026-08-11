import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "readmitted_30d"

CATEGORICAL_COLS = [
    "race", "gender", "max_glu_serum", "A1Cresult",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "age_group", "payer_group", "admission_type_id", 
    "discharge_disposition_id", "admission_source_id"
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "num_meds_administered",
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin", "change", "diabetesMed",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/diabetic_data_clean.csv"),
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "model"),
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    if os.path.isdir(data_path):
        csvs = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_path = os.path.join(data_path, csvs[0])
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {data_path}")
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    print("\n=== Evaluation ===")
    print(f"ROC AUC: {auc:.3f}")
    print(f"Average precision (PR AUC): {avg_precision:.3f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["not readmitted <30d", "readmitted <30d"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print("\nPrecision/recall at a few thresholds:")
    for t in [0.2, 0.3, 0.4, 0.5]:
        idx = np.searchsorted(thresholds, t)
        if idx < len(precisions):
            print(f"  threshold={t:.1f}  precision={precisions[idx]:.3f}  recall={recalls[idx]:.3f}")

    return {"roc_auc": auc, "average_precision": avg_precision}


def print_top_coefficients(pipeline: Pipeline, n: int = 15):
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:n]

    print(f"\nTop {n} features by |coefficient| (positive = pushes toward readmission):")
    for i in order:
        print(f"  {feature_names[i]:40s} {coefs[i]:+.3f}")


def run():
    args = parse_args()

    df = load_data(args.data_path)
    missing_cols = set(CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Expected columns missing from input data: {missing_cols}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,  # keep the ~9% positive rate consistent across train/test
        random_state=args.random_state,
    )
    print(f"Train: {len(X_train):,} rows ({y_train.mean()*100:.1f}% positive)")
    print(f"Test:  {len(X_test):,} rows ({y_test.mean()*100:.1f}% positive)")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test)
    print_top_coefficients(pipeline)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained pipeline (preprocessing + model) to {model_path}")


if __name__ == "__main__":
    run()
