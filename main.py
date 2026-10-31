"""
=============================================================================
Exam Result Prediction Using Predictive Analytics
Master Execution Script (main.py)
Orchestrates:
  1. Data Cleaning (data_cleaning.py)
  2. Exploratory Data Analysis (eda.py)
  3. Model Training & Benchmark (train_models.py)
=============================================================================
"""

from data_cleaning import clean_student_data
from eda import run_eda
from train_models import train_and_evaluate


def main():
    print("\n" + "=" * 75)
    print("🎓 PROJECT: Exam Result Prediction Using Predictive Analytics")
    print("Predicting Student Pass / Fail Outcomes Using Machine Learning")
    print("=" * 75 + "\n")

    # Step 1: Data Cleaning
    print("[STEP 1/3] Cleaning raw student performance data...")
    cleaned_df = clean_student_data()

    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/3] Generating Exploratory Data Analysis graphs...")
    run_eda(cleaned_df)

    # Step 3: Model Training & Evaluation
    print("\n[STEP 3/3] Training and evaluating exactly five classification models...")
    results_df, best_model = train_and_evaluate()

    print("\n" + "=" * 75)
    print("🎉 PROJECT: Exam Result Prediction Using Predictive Analytics")
    print("Execution completed successfully!")
    print("=" * 75)
    print(f"Top Performing Model : {best_model}")
    print(f"Top Test Accuracy    : {results_df.loc[results_df['Model'] == best_model, 'Accuracy'].values[0]*100:.2f}%")
    print(f"Top Test F1-Score    : {results_df.loc[results_df['Model'] == best_model, 'F1-Score'].values[0]*100:.2f}%")
    print(f"Models Evaluated     : Logistic Regression, Decision Tree, Random Forest, KNN, SVM")
    print(f"Saved Best Model     : models/best_exam_model.joblib")
    print(f"Results CSV File     : outputs/model_comparison.csv")
    print(f"Saved Graphs Folder  : graphs/")
    print("=" * 70)
    print("Launch Dashboard via : python -m streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
