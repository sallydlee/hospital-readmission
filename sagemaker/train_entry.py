import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import parse_args, load_data, build_pipeline, evaluate, print_top_coefficients  # noqa: E402
from sklearn.model_selection import train_test_split
from train import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL 
import joblib  


def run():
    args = parse_args()

    df = load_data(args.data_path)
    missing_cols = set(CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Expected columns missing from input data: {missing_cols}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
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
    print(f"\nSaved trained pipeline to {model_path} (SageMaker will upload this to S3)")


if __name__ == "__main__":
    run()
