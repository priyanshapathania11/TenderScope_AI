"""
=============================================================================
Exam Result Prediction Using Predictive Analytics: Streamlit Dashboard (dashboard.py)
Interactive UI for model comparison and real-time student pass/fail prediction.
=============================================================================
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Exam Result Prediction Using Predictive Analytics",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CLEAN_FILE = BASE_DIR / "outputs" / "cleaned_student_data.csv"
MODEL_FILE = BASE_DIR / "models" / "best_exam_model.joblib"
METRICS_FILE = BASE_DIR / "outputs" / "model_comparison.csv"
GRAPH_DIR = BASE_DIR / "graphs"


@st.cache_data
def load_data():
    if not CLEAN_FILE.exists():
        st.error("Cleaned data not found! Please run 'python main.py' first.")
        st.stop()
    return pd.read_csv(CLEAN_FILE)


@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return None


@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None


df = load_data()
metrics_df = load_metrics()
model = load_model()

# Sidebar Branding
with st.sidebar:
    st.title("🎓 Project Navigation")
    st.markdown("**Project Name:**")
    st.info("Exam Result Prediction Using Predictive Analytics")
    st.markdown("---")
    st.write("📊 **Course**: Predictive Analytics Assessment")
    st.write("📁 **Dataset**: `student_exam_performance.csv` (100,000 records)")
    st.write("🎯 **Target**: `pass_status` (Pass / Fail)")
    st.write("🏆 **Best Model**: Support Vector Machine (84.78% Accuracy)")
    st.markdown("---")

# Header
st.title("🎓 Exam Result Prediction Using Predictive Analytics")
st.subheader("Predicting Student Pass / Fail Outcomes Using Machine Learning")
st.caption("College Assessment Project | Predictive Analytics")
st.markdown("---")

# KPI Summary
total_students = len(df)
pass_count = (df["pass_status"] == "Pass").sum()
fail_count = (df["pass_status"] == "Fail").sum()
pass_rate = (pass_count / total_students) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", f"{total_students:,}")
col2.metric("Passed Students", f"{pass_count:,}")
col3.metric("Failed Students", f"{fail_count:,}")
col4.metric("Overall Pass Rate", f"{pass_rate:.1f}%")

st.markdown("---")

# Navigation Tabs
tab_models, tab_predict, tab_charts, tab_data = st.tabs([
    "🤖 5 Classification Models",
    "🔮 Predict Student Pass / Fail",
    "📊 Evaluation Charts",
    "📁 Dataset Explorer"
])

# TAB 1: MODEL COMPARISON
with tab_models:
    st.markdown("### 🤖 Exam Result Prediction Using Predictive Analytics: Model Benchmark")
    st.markdown("""
    All models were trained on student preparation features, past performance, and study habits 
    (with all post-exam outcome columns excluded to prevent data leakage).
    """)

    if metrics_df is not None:
        format_dict = {
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}"
        }
        if "ROC-AUC" in metrics_df.columns:
            format_dict["ROC-AUC"] = "{:.2%}"

        st.dataframe(
            metrics_df.style.format(format_dict).highlight_max(subset=["Accuracy", "F1-Score"], color="#c6f6d5"),
            use_container_width=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_acc = px.bar(
                metrics_df,
                x="Model",
                y="Accuracy",
                color="Model",
                title="Model Accuracy Comparison"
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_b:
            fig_f1 = px.bar(
                metrics_df,
                x="Model",
                y="F1-Score",
                color="Model",
                title="Model F1-Score Comparison"
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        st.info("""
        **Key Takeaways**:
        - **Best Model**: **Support Vector Machine (SVM)** achieved the highest test accuracy (**84.78%**) and F1-score (**90.49%**).
        - **Logistic Regression** also performed strongly (**84.48%** accuracy).
        - All 5 models achieved over 80% accuracy.
        """)

# TAB 2: REAL-TIME PREDICTOR
with tab_predict:
    st.markdown("### 🔮 Real-Time Student Pass / Fail Prediction")
    st.write("Enter student study habits and academic history to predict whether the student is likely to pass the exam:")

    if model is None:
        st.error("Trained model not found. Please run 'python main.py' first.")
    else:
        with st.form("exam_prediction_form"):
            p1, p2, p3 = st.columns(3)

            with p1:
                in_study_hours = st.slider("Daily Study Hours:", 0.5, 12.0, 4.0, step=0.5)
                in_prev_score = st.slider("Previous Exam Score:", 0.0, 100.0, 75.0, step=1.0)
                in_attendance = st.slider("Attendance Percentage (%):", 20.0, 100.0, 85.0, step=1.0)
                in_assign = st.slider("Assignment Completion Rate (%):", 0.0, 100.0, 80.0, step=1.0)

            with p2:
                in_sleep = st.slider("Sleep Hours per Night:", 4.0, 12.0, 7.0, step=0.5)
                in_practice_tests = st.slider("Practice Tests Completed:", 0, 25, 6)
                in_stress = st.slider("Stress Level (1 to 10):", 1, 10, 5)
                in_motivation = st.selectbox("Motivation Level:", ["High", "Medium", "Low"])

            with p3:
                in_education = st.selectbox("Education Level:", ["High School", "Undergraduate", "Postgraduate"])
                in_school = st.selectbox("School Type:", ["Public", "Private"])
                in_study_method = st.selectbox("Study Method:", ["Flashcards", "Summarizing", "Practice Testing", "Self-Explanation"])
                in_exam_prep = st.slider("Exam Preparation Days:", 1, 60, 20)

            btn = st.form_submit_button("🚀 Predict Exam Result")

        if btn:
            # Create a sample row using median defaults for non-entered features
            sample_row = df.drop(columns=["pass_status"]).iloc[0:1].copy()

            # Override with user inputs
            sample_row["study_hours_per_day"] = in_study_hours
            sample_row["previous_exam_score"] = in_prev_score
            sample_row["attendance_percentage"] = in_attendance
            sample_row["assignment_completion_rate"] = in_assign
            sample_row["sleep_hours"] = in_sleep
            sample_row["practice_tests_completed"] = in_practice_tests
            sample_row["stress_level"] = in_stress
            sample_row["motivation_level"] = in_motivation
            sample_row["education_level"] = in_education
            sample_row["school_type"] = in_school
            sample_row["study_method"] = in_study_method
            sample_row["exam_preparation_days"] = in_exam_prep

            # Predict
            pred = model.predict(sample_row)[0]

            if pred == 1:
                st.success("🎉 **Prediction: PASS**")
                st.write("The student shows strong study habits, good attendance, and consistent preparation, indicating a high likelihood of passing the exam.")
            else:
                st.error("⚠️ **Prediction: AT RISK OF FAILING**")
                st.write("The indicators suggest the student may be at academic risk. Recommended interventions: increase daily revision, complete more practice tests, and improve attendance.")

# TAB 3: CHARTS
with tab_charts:
    st.markdown("### 📊 Exam Result Prediction Using Predictive Analytics: Evaluation Graphs")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        g1 = GRAPH_DIR / "model_accuracy_comparison.png"
        if g1.exists():
            st.image(str(g1), caption="Model Accuracy Comparison (Test Set)", use_container_width=True)
    with col_c2:
        g2 = GRAPH_DIR / "model_metrics_comparison.png"
        if g2.exists():
            st.image(str(g2), caption="Model Precision, Recall, and F1-Score Comparison", use_container_width=True)

    g3 = GRAPH_DIR / "confusion_matrices.png"
    if g3.exists():
        st.image(str(g3), caption="Confusion Matrices for All Five Classification Models", use_container_width=True)

    g4 = GRAPH_DIR / "roc_curves.png"
    if g4.exists():
        st.image(str(g4), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

# TAB 4: DATASET EXPLORER
with tab_data:
    st.markdown("### 📁 Dataset Preview (First 50 Rows)")
    st.dataframe(df.head(50), use_container_width=True)
