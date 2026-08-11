import pandas as pd
import numpy as np

RAW_PATH = "data/diabetic_data.csv"
PROCESSED_PATH = "data/processed/diabetic_data_clean.csv"

# Discharge dispositions where the patient cannot be meaningfully "readmitted"
# (died or entered hospice), plus data-quality codes (18=NULL, 25=Not Mapped,
# 26=Unknown/Invalid) — from IDS_mapping.csv
EXCLUDE_DISCHARGE_CODES = [11, 13, 14, 18, 19, 20, 21, 25, 26]

# The 15 medication columns found to have zero or near-zero variance
# AKA almost entirely "No" across all rows
NEAR_CONSTANT_MED_COLS = [
    "acetohexamide", "tolbutamide", "troglitazone", "examide", "citoglipton",
    "tolazamide", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "acarbose",
    "chlorpropamide", "miglitol", "nateglinide", "glyburide-metformin",
]

# All 23 original medication columns to compute num_meds_administered
ALL_MED_COLS = NEAR_CONSTANT_MED_COLS + [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
]

PAYER_CODE_BUCKETS = {
    "MC": "medicare", "MP": "medicare",
    "MD": "medicaid", "DM": "medicaid",
    "SP": "self_pay",
    "HM": "private", "BC": "private", "CP": "private",
    "PO": "private", "CM": "private", "SI": "private",
    "UN": "other", "OG": "other", "OT": "other",
    "CH": "other", "WC": "other", "FR": "other",
}

AGE_BUCKETS = {
    "[0-10)": "young", "[10-20)": "young",
    "[20-30)": "adult",
    "[30-40)": "mid_senior", "[40-50)": "mid_senior", "[50-60)": "mid_senior",
    "[60-70)": "mid_senior", "[70-80)": "mid_senior", "[80-90)": "mid_senior",
    "[90-100)": "mid_senior",
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values="?")
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def keep_first_encounter_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    encounter_id is assigned sequentially as encounters occur, so sorting
    by encounter_id ascending and keeping the first per patient is the
    standard proxy for chronological order
    """
    before = len(df)
    df = (
        df.sort_values("encounter_id", ascending=True)
          .drop_duplicates(subset="patient_nbr", keep="first")
    )
    print(f"Kept first encounter per patient: {before:,} -> {len(df):,} rows")
    return df.drop(columns=["patient_nbr"])


def exclude_terminal_and_bad_quality_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_CODES)]
    print(f"Excluded terminal/bad-quality discharges: {before:,} -> {len(df):,} rows")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    rate = df["readmitted_30d"].mean() * 100
    print(f"Positive class rate (readmitted <30 days): {rate:.1f}%")
    return df


def add_medication_summary(df: pd.DataFrame) -> pd.DataFrame:
    df["num_meds_administered"] = (df[ALL_MED_COLS] != "No").sum(axis=1)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["weight", "medical_specialty", "encounter_id"] + NEAR_CONSTANT_MED_COLS
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def binarize_remaining_medications(df: pd.DataFrame) -> pd.DataFrame:
    remaining_med_cols = [c for c in ALL_MED_COLS if c not in NEAR_CONSTANT_MED_COLS]
    for col in remaining_med_cols:
        df[col] = df[col].isin(["Up", "Down"]).astype(int)
    return df


def group_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    def bucket(code):
        if pd.isna(code):
            return "missing"
        code = str(code)
        if code.startswith("250"):
            return "diabetes"
        if code.startswith(("V", "E")):
            return "other"
        try:
            n = float(code)
        except ValueError:
            return "other"
        if 390 <= n <= 459 or n == 785:
            return "circulatory"
        if 460 <= n <= 519 or n == 786:
            return "respiratory"
        if 520 <= n <= 579 or n == 787:
            return "digestive"
        if 800 <= n <= 999:
            return "injury"
        if 710 <= n <= 739:
            return "musculoskeletal"
        if 580 <= n <= 629 or n == 788:
            return "genitourinary"
        if 140 <= n <= 239:
            return "neoplasm"
        return "other"

    for c in ["diag_1", "diag_2", "diag_3"]:
        df[f"{c}_group"] = df[c].apply(bucket)
    return df.drop(columns=["diag_1", "diag_2", "diag_3"])


def bucket_age(df: pd.DataFrame) -> pd.DataFrame:
    df["age_group"] = df["age"].map(AGE_BUCKETS)
    return df.drop(columns=["age"])


def bucket_payer_code(df: pd.DataFrame) -> pd.DataFrame:
    df["payer_code"] = df["payer_code"].fillna("UN")
    df["payer_group"] = df["payer_code"].map(PAYER_CODE_BUCKETS)
    return df.drop(columns=["payer_code"])


def handle_lab_result_missingness(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["max_glu_serum", "A1Cresult"]:
        df[col] = df[col].fillna("not_tested")
    return df


def handle_remaining_missingness(df: pd.DataFrame) -> pd.DataFrame:
    if "race" in df.columns:
        df["race"] = df["race"].fillna("unknown")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"Dropping columns that became constant after processing: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def binarize_change(df: pd.DataFrame) -> pd.DataFrame:
    df['change'] = df['change'].map({'No': 0, 'Ch': 1})
    return df


def binarize_diabetes_med(df: pd.DataFrame) -> pd.DataFrame:
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1})
    return df
    

def run():
    df = load_raw()
    df = keep_first_encounter_per_patient(df)
    df = exclude_terminal_and_bad_quality_rows(df)
    df = build_target(df)
    df = add_medication_summary(df)
    df = drop_columns(df)
    df = binarize_remaining_medications(df)
    df = binarize_change(df)
    df = binarize_diabetes_med(df)
    df = group_diagnosis_codes(df)
    df = bucket_age(df)
    df = bucket_payer_code(df)
    df = handle_lab_result_missingness(df)
    df = handle_remaining_missingness(df)
    df = drop_constant_columns(df)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"-> {PROCESSED_PATH}")
    return df


if __name__ == "__main__":
    run()
