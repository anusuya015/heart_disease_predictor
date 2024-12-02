import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shap
import streamlit as st

# Set styles for plots
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Load dataset
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("heart.csv")  # Default for local development
    return df

uploaded_file = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])
df = load_data(uploaded_file)

# Display dataset
if st.sidebar.checkbox("Show Data"):
    st.write(df)

# Preprocessing
categorical_val = []
continuous_val = []

for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

categorical_val.remove("target")
dataset = pd.get_dummies(df, columns=categorical_val)
col_to_scale = ["age", "trestbps", "chol", "thalach", "oldpeak"]

s_sc = StandardScaler()
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

# Splitting the data
X = dataset.drop("target", axis=1)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
lr_clf = LogisticRegression(solver="liblinear", class_weight="balanced")
lr_clf.fit(X_train, y_train)

# Model Performance
if st.sidebar.checkbox("Show Model Performance"):
    train_acc = accuracy_score(y_train, lr_clf.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
    st.write(f"Training Accuracy: {train_acc:.2f}%")
    st.write(f"Testing Accuracy: {test_acc:.2f}%")

# SHAP Visualizations
if st.sidebar.checkbox("Show Feature Importance (SHAP)"):
    st.subheader("Feature Importance Visualization")
    explainer = shap.Explainer(lr_clf, X_train)
    shap_values = explainer(X_test)

    # Global Feature Importance
    st.markdown("#### Global Feature Importance")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot()

    # Local Explanation
    st.markdown("#### Local Explanation for a Single Prediction")
    single_instance = X_test.iloc[0]
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0].values,
        base_values=shap_values[0].base_values,
        data=single_instance,
        feature_names=X_test.columns,
    ))
    st.pyplot()
