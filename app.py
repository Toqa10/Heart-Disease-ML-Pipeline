import streamlit as st
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
# White text style on black background
# ----------------------------
st.markdown(
    """
    <style>
    body { background-color: #000000; color: white; }
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
    This application helps you estimate the probability of having heart disease based on some medical measurements.
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Normal Ranges & Meaning of Features
# ----------------------------
st.markdown(
    """
    <div>
    <h4>Age:</h4> 20–79 years.<br>
    <h4>Sex:</h4> 1 = Male, 0 = Female.<br>
    <h4>Chest Pain Type (cp):</h4> 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic.<br>
    <h4>Resting Blood Pressure (trestbps):</h4> Normal < 120 mm Hg.<br>
    <h4>Cholesterol (chol):</h4> Desirable < 200 mg/dl.<br>
    <h4>Fasting Blood Sugar (fbs):</h4> 1 = >120 mg/dl, 0 = <120 mg/dl.<br>
    <h4>Resting ECG (restecg):</h4> 0 = Normal, 1 = ST-T abnormality, 2 = Left Ventricular Hypertrophy.<br>
    <h4>Max Heart Rate Achieved (thalach):</h4> Normal >100 bpm.<br>
    <h4>Exercise Induced Angina (exang):</h4> 1 = Yes, 0 = No.<br>
    <h4>Oldpeak:</h4> ST depression induced by exercise. Normal < 1.0.<br>
    <h4>Slope:</h4> 0 = Upsloping, 1 = Flat, 2 = Downsloping.<br>
    <h4>Ca:</h4> Number of major vessels (0–3) colored by fluoroscopy.<br>
    <h4>Thal:</h4> 1 = Normal, 2 = Fixed defect, 3 = Reversible defect.<br>
    <h4>Interpretation:</h4>
    Low Risk: Maintain healthy lifestyle, regular check-ups.<br>
    High Risk: Visit a cardiologist for further evaluation.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load model
# ----------------------------
model_path = Path("models/final_model.pkl")
if model_path.exists():
    # Load tuple: (scaler, model)
    scaler, model = joblib.load(model_path)
else:
    model = None
    scaler = None
    st.warning("⚠️ Model file not found. Please place it in 'models/final_model.pkl'")

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
    if model is None or scaler is None:
        st.error("Model not available. Please upload the model file first.")
    else:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if prediction[0] == 1:
            st.error(f"💔 High Risk of Heart Disease ({probability:.2f}% probability)")
            st.markdown("**Advice:** Please consult a cardiologist for further evaluation.")
        else:
            st.success(f"💚 Low Risk of Heart Disease ({probability:.2f}% probability)")
            st.markdown("**Advice:** Maintain a healthy lifestyle and regular check-ups.")
