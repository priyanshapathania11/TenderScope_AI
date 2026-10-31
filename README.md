# Exam Result Prediction Using Predictive Analytics

## Project Overview
This project uses Predictive Analytics to analyze student academic data and predict whether a student is likely to pass or fail. It includes data cleaning, exploratory data analysis, visualization, and classification models.

## Objectives
- Clean and prepare the student dataset.
- Explore student academic performance.
- Train and compare five classification models.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Identify the best-performing model based on evaluation results.

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Project Structure
- `data/` — Dataset
- `graphs/` — Generated evaluation graphs
- `models/` — Saved trained models
- `data_cleaning.py` — Data cleaning
- `eda.py` — Exploratory data analysis
- `train_models.py` — Model training and evaluation
- `dashboard.py` — Interactive dashboard
- `main.py` — Main project script
- `requirements.txt` — Required Python libraries

## How to Run
1. Install the required libraries using `python -m pip install -r requirements.txt`.
2. Run the project scripts as required.
3. Launch the dashboard using `python -m streamlit run dashboard.py`.

## Results
Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices. The results are generated from the dataset when the models are trained.