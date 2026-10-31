"""
=============================================================================
Exam Result Prediction Using Predictive Analytics: Data Cleaning (data_cleaning.py)
Author: Priyansha Pathania
=============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_exam_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_student_data.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


def clean_student_data():
    print("=" * 75)
    print("🎓 PROJECT: Exam Result Prediction Using Predictive Analytics")
    print("STAGE 1: LOADING & INSPECTING DATASET")
    print("=" * 75)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 1. Check pass_status target column
    if "pass_status" not in df.columns:
        raise ValueError("Target column 'pass_status' not found in dataset!")

    print("\nTarget 'pass_status' distribution:")
    print(df["pass_status"].value_counts(dropna=False))

    # 2. Check duplicates
    dups = df.duplicated().sum()
    print(f"\nDuplicate rows found: {dups}")
    if dups > 0:
        df = df.drop_duplicates()
        print(f"Removed {dups} duplicate rows. Remaining: {len(df):,}")

    # 3. Prevent Data Leakage
    # Exclude post-exam outcomes that directly leak the exam result
    leakage_columns = [
        "student_id",           # Non-predictive identifier
        "exam_score",           # Final numeric score (directly decides Pass/Fail >= 50)
        "performance_grade",    # Letter grade (A, B, C, D vs F)
        "performance_level",    # Performance bucket (High, Medium, Low)
        "questions_correct"     # Number of correct questions directly determines score
    ]
    
    cols_to_drop = [c for c in leakage_columns if c in df.columns]
    print(f"\nPreventing Data Leakage: Excluding post-exam outcome columns:")
    for c in cols_to_drop:
        print(f"  - Dropping '{c}'")
    
    df_cleaned = df.drop(columns=cols_to_drop).copy()

    # 4. Check missing values
    missing = df_cleaned.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"\nColumns with missing values ({len(missing_cols)} columns):")
    for col, count in missing_cols.items():
        print(f"  - {col}: {count:,} missing ({count/len(df_cleaned)*100:.2f}%)")

    # 5. Clean text categories: strip extra whitespace
    cat_cols = df_cleaned.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        if c != "pass_status":
            df_cleaned[c] = df_cleaned[c].astype(str).str.strip().replace("nan", np.nan)

    # 6. Save cleaned data
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 70)
    print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
    print(f"Final shape: {df_cleaned.shape[0]:,} rows, {df_cleaned.shape[1]} columns")
    print("=" * 70)
    return df_cleaned


if __name__ == "__main__":
    clean_student_data()
