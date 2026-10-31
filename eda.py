"""
=============================================================================
Exam Result Prediction Using Predictive Analytics: Exploratory Data Analysis (eda.py)
Generates key visualisations for student performance:
  1. Target Distribution (Pass vs. Fail)
  2. Study Hours vs. Pass Status
  3. Attendance Percentage vs. Pass Status
  4. Previous Exam Score vs. Pass Status
  5. Correlation Heatmap of Key Academic Features
=============================================================================
"""

import os
import tempfile
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CLEAN_FILE = BASE_DIR / "outputs" / "cleaned_student_data.csv"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Clean, professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def run_eda(df: pd.DataFrame = None):
    print("=" * 75)
    print("🎓 PROJECT: Exam Result Prediction Using Predictive Analytics")
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 75)

    if df is None:
        if not CLEAN_FILE.exists():
            from data_cleaning import clean_student_data
            df = clean_student_data()
        else:
            df = pd.read_csv(CLEAN_FILE)

    # 1. Target Distribution (Pass vs Fail)
    plt.figure(figsize=(7, 5))
    counts = df["pass_status"].value_counts()
    colors = ["#2b5c8f", "#e63946"]
    bars = plt.bar(counts.index, counts.values, color=colors, width=0.45, edgecolor="black")
    plt.title("Distribution of Student Exam Results (Pass vs. Fail)", pad=12)
    plt.xlabel("Exam Result Status")
    plt.ylabel("Number of Students")
    for b in bars:
        h = b.get_height()
        pct = (h / len(df)) * 100
        plt.text(b.get_x() + b.get_width() / 2, h + 1500, f"{h:,}\n({pct:.1f}%)", ha="center", weight="bold")
    plt.ylim(0, max(counts.values) * 1.15)
    g1 = GRAPH_DIR / "01_pass_status_distribution.png"
    plt.savefig(g1)
    plt.close()
    print(f"  [1/5] Generated: {g1.name}")

    # 2. Daily Study Hours vs. Pass Status
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="pass_status", y="study_hours_per_day", palette=["#2a9d8f", "#e76f51"], width=0.4)
    plt.title("Daily Study Hours by Exam Result", pad=12)
    plt.xlabel("Exam Result Status")
    plt.ylabel("Daily Study Hours")
    g2 = GRAPH_DIR / "02_study_hours_vs_result.png"
    plt.savefig(g2)
    plt.close()
    print(f"  [2/5] Generated: {g2.name}")

    # 3. Attendance Percentage vs. Pass Status
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df.dropna(subset=["attendance_percentage"]), x="pass_status", y="attendance_percentage", palette=["#457b9d", "#f4a261"], width=0.4)
    plt.title("Class Attendance Percentage by Exam Result", pad=12)
    plt.xlabel("Exam Result Status")
    plt.ylabel("Attendance Percentage (%)")
    g3 = GRAPH_DIR / "03_attendance_vs_result.png"
    plt.savefig(g3)
    plt.close()
    print(f"  [3/5] Generated: {g3.name}")

    # 4. Previous Exam Score vs. Pass Status
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="pass_status", y="previous_exam_score", palette=["#1d3557", "#e63946"], width=0.4)
    plt.title("Previous Exam Score by Exam Result", pad=12)
    plt.xlabel("Exam Result Status")
    plt.ylabel("Previous Exam Score")
    g4 = GRAPH_DIR / "04_previous_score_vs_result.png"
    plt.savefig(g4)
    plt.close()
    print(f"  [4/5] Generated: {g4.name}")

    # 5. Correlation Heatmap of Key Academic Numerical Features
    plt.figure(figsize=(9, 7))
    key_numeric = [
        "study_hours_per_day",
        "previous_exam_score",
        "attendance_percentage",
        "assignment_completion_rate",
        "practice_tests_completed",
        "sleep_hours",
        "stress_level"
    ]
    corr = df[key_numeric].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"label": "Pearson Correlation"})
    plt.title("Correlation Heatmap of Key Academic & Behavioral Features", pad=12)
    plt.xticks(rotation=30, ha="right")
    g5 = GRAPH_DIR / "05_correlation_heatmap.png"
    plt.savefig(g5)
    plt.close()
    print(f"  [5/5] Generated: {g5.name}")

    print("All 5 EDA graphs successfully generated in 'graphs/' folder.\n")


if __name__ == "__main__":
    run_eda()
