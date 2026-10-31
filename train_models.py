"""
=============================================================================
Exam Result Prediction Using Predictive Analytics: Model Training (train_models.py)
Author: Priyansha Pathania

This script trains and evaluates EXACTLY five classification models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)

Evaluation metrics calculated:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

Generated evaluation graphs (saved in 'graphs/'):
  1. model_accuracy_comparison.png
  2. model_metrics_comparison.png
  3. confusion_matrices.png
  4. roc_curves.png
=============================================================================
"""

import os
import tempfile
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Exactly five classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_exam_performance.csv"
CLEAN_FILE = BASE_DIR / "outputs" / "cleaned_student_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
GRAPH_DIR = BASE_DIR / "graphs"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


def train_and_evaluate(sample_size: int = 20000):
    print("=" * 75)
    print("🎓 PROJECT: Exam Result Prediction Using Predictive Analytics")
    print("STAGE 1: VERIFYING DATASET & TARGET COLUMN")
    print("=" * 75)

    # 1. Verify dataset existence
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {DATA_FILE}")

    # Use cleaned dataset (or generate if missing)
    if not CLEAN_FILE.exists():
        from data_cleaning import clean_student_data
        df = clean_student_data()
    else:
        df = pd.read_csv(CLEAN_FILE)

    target_col = "pass_status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Verify actual class distribution
    class_counts = df[target_col].value_counts()
    print(f"Target column '{target_col}' verified:")
    for label, count in class_counts.items():
        print(f"  - {label}: {count:,} ({count / len(df) * 100:.2f}%)")

    # Map target: Fail -> 0, Pass -> 1
    y_all = df[target_col].map({"Fail": 0, "Pass": 1}).astype(int)
    X_all = df.drop(columns=[target_col])

    # Representative stratified sample for fast and interactive training
    if len(df) > sample_size:
        print(f"\nUsing a representative stratified sample of {sample_size:,} records for fast execution.")
        X_sub, _, y_sub, _ = train_test_split(
            X_all, y_all, train_size=sample_size, stratify=y_all, random_state=42
        )
    else:
        X_sub, y_sub = X_all, y_all

    # Identify numeric and categorical columns
    num_cols = X_sub.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_sub.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"Features: {X_sub.shape[1]} total ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

    # 2. Build Preprocessing Pipeline (strictly fitted on training split to prevent leakage)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # 80% Train, 20% Test Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=42
    )
    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # 3. Define the EXACT Five Classification Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine": LinearSVC(random_state=42)
    }

    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING AND EVALUATING THE FIVE MODELS")
    print("=" * 70)

    results = []
    trained_pipelines = {}
    cm_dict = {}
    roc_dict = {}
    errors = {}

    for name, clf in models.items():
        print(f"\n--> Training {name}...")
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

            # Fit pipeline on training data
            pipe.fit(X_train, y_train)

            # Predict on unseen test data
            y_pred = pipe.predict(X_test)

            # Calculate evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Compute ROC-AUC (using predict_proba or decision_function)
            auc = None
            y_score = None
            if hasattr(pipe, "predict_proba"):
                y_score = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_score)
            elif hasattr(pipe, "decision_function"):
                y_score = pipe.decision_function(X_test)
                auc = roc_auc_score(y_test, y_score)

            cm = confusion_matrix(y_test, y_pred)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4) if auc is not None else np.nan,
                "Status": "Trained Successfully"
            })

            trained_pipelines[name] = pipe
            cm_dict[name] = cm
            if y_score is not None and auc is not None:
                roc_dict[name] = (y_test, y_score, auc)

            # Save individual model pipeline to models/
            model_slug = name.lower().replace(" ", "_").replace("-", "_")
            joblib.dump(pipe, MODEL_DIR / f"{model_slug}.joblib")

            auc_str = f"{auc*100:6.2f}%" if auc is not None else "N/A"
            print(f"    [SUCCESS] Accuracy: {acc*100:6.2f}% | Precision: {prec*100:6.2f}% | Recall: {rec*100:6.2f}% | F1: {f1*100:6.2f}% | ROC-AUC: {auc_str}")

        except Exception as e:
            print(f"    [ERROR] Model {name} failed: {e}")
            errors[name] = str(e)
            results.append({
                "Model": name,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1-Score": np.nan,
                "ROC-AUC": np.nan,
                "Status": f"Failed: {e}"
            })

    # Save results to outputs/model_comparison.csv
    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    comparison_file = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_file, index=False)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON (Ranked by Test Accuracy):")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save the best model
    successful_models = results_df.dropna(subset=["Accuracy"])
    if not successful_models.empty:
        best_name = successful_models.iloc[0]["Model"]
        best_pipe = trained_pipelines[best_name]
        joblib.dump(best_pipe, MODEL_DIR / "best_exam_model.joblib")
        print(f"\n[WINNER] Best Performing Model: {best_name}")
        print(f"Saved best model pipeline to: {MODEL_DIR / 'best_exam_model.joblib'}")
    else:
        best_name = None
        print("\n[WARNING] No models trained successfully.")

    # 4. Generate all required evaluation graphs
    print("\n" + "=" * 70)
    print("STAGE 3: GENERATING EVALUATION GRAPHS")
    print("=" * 70)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # --- GRAPH 1: Model Accuracy Comparison ---
    plt.figure(figsize=(9, 5))
    acc_df = successful_models.sort_values("Accuracy", ascending=False)
    colors = ["#2b5c8f", "#2a9d8f", "#e76f51", "#f4a261", "#457b9d"]
    bars = plt.bar(acc_df["Model"], acc_df["Accuracy"] * 100, color=colors[:len(acc_df)], width=0.5, edgecolor="black")
    plt.title("Exam Result Prediction: Model Accuracy Comparison (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classification Model", fontsize=11, fontweight="bold")
    plt.ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 100)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    g1 = GRAPH_DIR / "model_accuracy_comparison.png"
    plt.savefig(g1, dpi=300)
    plt.close()
    print(f"  [1/4] Saved: {g1.name}")

    # --- GRAPH 2: Model Precision, Recall, and F1-Score Comparison ---
    plt.figure(figsize=(10, 5.5))
    metrics_melt = pd.melt(
        successful_models,
        id_vars=["Model"],
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=metrics_melt, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Exam Result Prediction: Model Performance Comparison (Precision, Recall, F1)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classification Model", fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Evaluation Metric", loc="upper right")
    plt.tight_layout()
    g2 = GRAPH_DIR / "model_metrics_comparison.png"
    plt.savefig(g2, dpi=300)
    # Also save as model_comparison.png for backward compatibility
    plt.savefig(GRAPH_DIR / "model_comparison.png", dpi=300)
    plt.close()
    print(f"  [2/4] Saved: {g2.name}")

    # --- GRAPH 3: Confusion Matrices for All 5 Models ---
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(18, 3.8))
    if len(cm_dict) == 1:
        axes = [axes]
    for i, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=["Fail (0)", "Pass (1)"], yticklabels=["Fail (0)", "Pass (1)"])
        axes[i].set_title(name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontsize=9)
        if i == 0:
            axes[i].set_ylabel("Actual Label", fontsize=9)
        else:
            axes[i].set_ylabel("")
    plt.suptitle("Exam Result Prediction: Confusion Matrices for All Five Classification Models", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    g3 = GRAPH_DIR / "confusion_matrices.png"
    plt.savefig(g3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [3/4] Saved: {g3.name}")

    # --- GRAPH 4: ROC Curves for Models ---
    plt.figure(figsize=(8, 6))
    curve_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, (y_t, y_s, auc_val)) in enumerate(roc_dict.items()):
        fpr, tpr, _ = roc_curve(y_t, y_s)
        color = curve_colors[i % len(curve_colors)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2)

    # Diagonal random chance line
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", alpha=0.7)
    plt.title("Exam Result Prediction: Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    g4 = GRAPH_DIR / "roc_curves.png"
    plt.savefig(g4, dpi=300)
    plt.close()
    print(f"  [4/4] Saved: {g4.name}")

    print("\n" + "=" * 70)
    print("ALL EVALUATION GRAPHS AND MODELS SAVED SUCCESSFULLY!")
    print("=" * 70)

    return results_df, best_name


if __name__ == "__main__":
    train_and_evaluate()
