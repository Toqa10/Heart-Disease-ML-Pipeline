import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------
# Styling (Black background + White text)
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #000000; /* Black background */
        color: white; /* White text */
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader, .stExpander {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Title & Description
# ----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    """
    <div style='color:white'>
    This application helps you estimate the probability of having heart disease based on some medical measurements.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Normal Ranges & Meaning of Features
# ----------------------------
st.markdown(
    """
    <div style='color:white'>
    <h4>Features:</h4>
    <ul>
    <li>Age: 20–79 years.</li>
    <li>Sex: 1 = Male, 0 = Female.</li>
    <li>Chest Pain Type (cp): 0–3.</li>
    <li>Resting Blood Pressure (trestbps): Normal < 120 mm Hg.</li>
    <li>Cholesterol (chol): Desirable < 200 mg/dl.</li>
    <li>Fasting Blood Sugar (fbs): 1 = >120 mg/dl (high), 0 = <120 mg/dl (normal).</li>
    <li>Resting ECG (restecg): 0–2.</li>
    <li>Max Heart Rate Achieved (thalach): >100 bpm.</li>
    <li>Exercise Induced Angina (exang): 1 = Yes, 0 = No.</li>
    <li>Oldpeak: ST depression induced by exercise. Normal < 1.0.</li>
    <li>Slope: 0–2.</li>
    <li>Ca: Number of major vessels (0–3).</li>
    <li>Thal: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load Model
# ----------------------------
model_path = Path("models/final_model.pkl")

if model_path.exists():
    model = joblib.load(model_path)
else:
    model = None
    st.warning("⚠️ Model file not found. Please place it in 'models/final_model.pkl'")

# ----------------------------
# Upload CSV File
# ----------------------------
st.subheader("📂 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df)

    if model is not None:
        predictions = model.predict(df)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[:, 1] * 100
        else:
            probabilities = [50.0] * len(predictions)

        results = pd.DataFrame({
            "Prediction": ["💔 High Risk" if p == 1 else "💚 Low Risk" for p in predictions],
            "Probability (%)": probabilities
        })
        st.write("Results:")
        st.dataframe(results)
    else:
        st.error("⚠️ Model not available. Please upload the model file first.")

st.markdown("---")

# ----------------------------
# Manual Input
# ----------------------------
st.subheader("📝 Or Enter Details Manually")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1=yes,0=no)", [1, 0])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=yes,0=no)", [1, 0])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1=Normal,2=Fixed defect,3=Reversible defect)", [1, 2, 3])

# ----------------------------
# Prediction for Manual Input
# ----------------------------
if st.button("Predict Risk"):
    if model is None:
        st.error("⚠️ Model not available. Please upload the model file first.")
    else:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
        else:
            probability = 50.0  # default if model has no predict_proba
        
        if prediction[0] == 1:
            st.error(f"💔 High Risk of Heart Disease ({probability:.2f}% probability)")
            st.markdown("**Advice:** Please consult a cardiologist for further evaluation.")
        else:
            st.success(f"💚 Low Risk of Heart Disease ({probability:.2f}% probability)")
            st.markdown("**Advice:** Maintain a healthy lifestyle and regular check-ups.")
