import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"heart.csv")
    return df

# Preprocess the dataset
@st.cache_data
def preprocess_data(df):
    # One-hot encoding for categorical variables
    categorical_val = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=categorical_val)
    
    # Feature scaling
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[col_to_scale] = scaler.fit_transform(df[col_to_scale])
    
    return df, scaler

# Load and preprocess data
df = load_data()
processed_data, scaler = preprocess_data(df)

# Prepare the model
X = processed_data.drop("target", axis=1)
y = processed_data["target"]
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X, y)

# Streamlit App
st.title("Heart Disease Predictor")
st.write("This application predicts the likelihood of heart disease based on medical inputs.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")
def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type (0: Typical Angina, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dL)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (0: Normal, 1: Abnormal, 2: Hypertrophy)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect)", [0, 1, 2, 3])

    # Combine inputs into a DataFrame
    data = {
        'age': [age], 'sex': [1 if sex == "Male" else 0], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    }
    features = pd.DataFrame(data)
    return features

input_data = user_input_features()

# Preprocess user input to match model features
input_data_encoded = pd.get_dummies(input_data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)
input_data_encoded.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.transform(
    input_data_encoded[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
)

# Prediction
prediction_prob = model.predict_proba(input_data_encoded)[:, 1][0]
threshold = 0.4
prediction = (prediction_prob >= threshold).astype(int)

# Display results
st.subheader("Prediction Results")
if prediction == 1:
    st.write(f"**High Risk**: The model predicts heart disease with a probability of {prediction_prob * 100:.2f}%.")
else:
    st.write(f"**Low Risk**: The model predicts no heart disease with a probability of {(1 - prediction_prob) * 100:.2f}%.")

# Feature importance with SHAP
st.subheader("Feature Importance")

# Use SHAP to explain the model
explainer = shap.LinearExplainer(model, X)  # X is the training dataset used for model fitting
shap_values = explainer.shap_values(input_data_encoded)

# Ensure the SHAP values are correctly processed for a single sample (when predicting for one user input)
# Since SHAP is primarily used for interpreting model behavior on the whole dataset, 
# here we are visualizing for a single user input by using a summary plot for one sample.

# Plot the SHAP summary plot for feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], input_data_encoded, plot_type="bar", show=False)  # shap_values[1] is for the positive class
st.pyplot(plt)
