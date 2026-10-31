# Exam Result Prediction Using Predictive Analytics

**Project Name**: Exam Result Prediction Using Predictive Analytics  
**Course / Domain**: Predictive Analytics Assessment  
**Goal**: Predicting student academic outcome (Pass vs. Fail) using Machine Learning  

A simple, beginner-friendly Predictive Analytics project that predicts whether a student will **Pass** or **Fail** (`pass_status`) based on their study habits, attendance, past academic performance, and preparation metrics.

---

## 1. Project Overview
- **Dataset**: `data/student_exam_performance.csv` (100,000 student records, 44 attributes).
- **Target Variable**: `pass_status`
  - `Pass`: 77,389 students (77.39%)
  - `Fail`: 22,611 students (22.61%)
- **Data Leakage Prevention**: Direct post-exam outcomes (`exam_score`, `performance_grade`, `performance_level`, `questions_correct`) and identifiers (`student_id`) were excluded from the input features.
- **Five Classification Models Evaluated**:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)

---

## 2. Project Structure
```
Exam Result Prediction Using Predictive Analytics/
├── data/
│   └── student_exam_performance.csv       # Local dataset (100,000 rows)
├── graphs/
│   ├── 01_pass_status_distribution.png    # EDA: Target class distribution
│   ├── 02_study_hours_vs_result.png       # EDA: Daily study hours vs result
│   ├── 03_attendance_vs_result.png        # EDA: Attendance percentage vs result
│   ├── 04_previous_score_vs_result.png    # EDA: Previous exam score vs result
│   ├── 05_correlation_heatmap.png         # EDA: Correlation heatmap of features
│   ├── model_accuracy_comparison.png      # Evaluation: Model accuracy comparison
│   ├── model_metrics_comparison.png       # Evaluation: Precision, Recall, F1 comparison
│   ├── confusion_matrices.png             # Evaluation: 5 confusion matrices
│   └── roc_curves.png                     # Evaluation: ROC curves and AUC scores
├── models/
│   ├── best_exam_model.joblib             # Best overall model pipeline (SVM)
│   ├── logistic_regression.joblib         # Saved Logistic Regression pipeline
│   ├── decision_tree.joblib               # Saved Decision Tree pipeline
│   ├── random_forest.joblib               # Saved Random Forest pipeline
│   ├── k_nearest_neighbors.joblib         # Saved KNN pipeline
│   └── support_vector_machine.joblib      # Saved SVM pipeline
├── outputs/
│   ├── cleaned_student_data.csv           # Preprocessed clean dataset
│   └── model_comparison.csv              # Real evaluation metrics table
├── main.py                                # Master execution script
├── data_cleaning.py                       # Data cleaning pipeline
├── eda.py                                 # Exploratory Data Analysis & graphs
├── train_models.py                        # Model training & benchmark
├── dashboard.py                           # Interactive Streamlit dashboard
├── requirements.txt                       # Project dependencies
└── README.md                              # Documentation
```

---

## 3. Real Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** 🏆 | **84.78%** | **87.62%** | **93.54%** | **90.49%** | **88.92%** | Trained Successfully |
| **Logistic Regression** | 84.48% | 87.79% | 92.86% | 90.25% | 88.91% | Trained Successfully |
| **Random Forest** | 82.75% | 83.01% | 97.71% | 89.76% | 87.97% | Trained Successfully |
| **Decision Tree** | 81.80% | 85.41% | 92.25% | 88.70% | 83.55% | Trained Successfully |
| **K-Nearest Neighbors (KNN)** | 80.33% | 82.81% | 94.12% | 88.10% | 77.09% | Trained Successfully |

---

## 4. How to Run

### Run the Model Training & Graph Generation Alone:
```powershell
python train_models.py
```

### Run the Full Pipeline (Cleaning + EDA + Training):
```powershell
python main.py
```

### Launch Interactive Streamlit Dashboard:
```powershell
python -m streamlit run dashboard.py
```
