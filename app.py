import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease ML Pipeline",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------
# White text style
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
    <h4>Age: 20–79 years.</h4>
    <p>Sex: 1 = Male, 0 = Female.</p>
    <p>Chest Pain Type (cp): 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic.</p>
    <p>Resting Blood Pressure (trestbps): Normal < 120 mm Hg.</p>
    <p>Cholesterol (chol): Desirable < 200 mg/dl.</p>
    <p>Fasting Blood Sugar (fbs): 1 = >120 mg/dl (high), 0 = <120 mg/dl (normal).</p>
    <p>Resting ECG (restecg): 0 = Normal, 1 = ST-T abnormality, 2 = Left Ventricular Hypertrophy.</p>
    <p>Max Heart Rate Achieved (thalach): Normal depends on age; usually >100 bpm.</p>
    <p>Exercise Induced Angina (exang): 1 = Yes, 0 = No.</p>
    <p>Oldpeak: ST depression induced by exercise. Normal < 1.0.</p>
    <p>Slope: 0 = Upsloping, 1 = Flat, 2 = Downsloping.</p>
    <p>Ca: Number of major vessels (0–3) colored by fluoroscopy.</p>
    <p>Thal: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect.</p>
    <h4>Interpretation:</h4>
    <p><b>Low Risk:</b> Maintain healthy lifestyle, regular check-ups.</p>
    <p><b>High Risk:</b> Visit a cardiologist for further evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load Model
# ----------------------------
model_path = Path("models/final_model.pkl")
model = None
if model_path.exists():
    model = joblib.load(model_path)

# ----------------------------
# Input Fields
# ----------------------------
st.subheader("Enter Your Details")

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
# Prediction
# ----------------------------
if st.button("Predict Risk"):
    if model is None:
        st.error("Model not available. Please upload the model file first.")
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
