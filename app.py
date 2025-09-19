import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------
# إعدادات الصفحة
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# ---------------------------
# CSS لتغيير الخلفية
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f4f8; 
}
[data-testid="stHeader"] {
    background-color: #d62828;
}
h1, h2, h3, h4 {
    color: #333333;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# صورة قلب حقيقي
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Heart_anatomy_labeled.svg/1200px-Heart_anatomy_labeled.svg.png",
    width=150,
    caption="Heart Disease Prediction"
)

st.title("💓 Heart Disease Prediction App")

# ---------------------------
# تحميل الموديل مع حماية
model_path = "models/final_model.pkl"
try:
    model = joblib.load(model_path)
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please place it in 'models/final_model.pkl'")
    model_loaded = False

# ---------------------------
# معلومات المتغيرات والنورمال رينج
info_df = pd.DataFrame({
    "Feature": [
        "Age", "Sex", "Chest Pain Type (cp)", "Resting Blood Pressure (trestbps)",
        "Serum Cholesterol (chol)", "Fasting Blood Sugar (fbs)",
        "Resting ECG (restecg)", "Max Heart Rate (thalach)",
        "Exercise Induced Angina (exang)", "ST Depression (oldpeak)",
        "Slope", "Number of Major Vessels (ca)", "Thal"
    ],
    "Normal Range": [
        "Adults 20–80 yrs", 
        "Male/Female", 
        "0–3 (typical/atypical angina etc.)", 
        "90–120 mmHg", 
        "< 200 mg/dl", 
        "0 (≤120 mg/dl), 1 (>120 mg/dl)", 
        "0–2 (normal to abnormal)", 
        "140–190 bpm", 
        "0 = No, 1 = Yes", 
        "0–1 low ST depression", 
        "0–2 slope types", 
        "0–3 vessels", 
        "0–3 thalassemia categories"
    ],
    "Meaning": [
        "Age of patient", 
        "Gender of patient", 
        "Chest pain type", 
        "Blood pressure at rest", 
        "Cholesterol level in blood", 
        "Blood sugar level after fasting", 
        "ECG test results", 
        "Max heart rate during exercise", 
        "Angina caused by exercise", 
        "ST depression induced by exercise", 
        "Slope of peak exercise ST segment", 
        "Number of major vessels seen in angiography", 
        "Thalassemia status"
    ]
})

with st.expander("ℹ️ Normal Ranges & Meaning of Features"):
    st.dataframe(info_df)

# ---------------------------
# إدخال البيانات من المستخدم
st.subheader("Enter Patient Data")

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)
chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# ---------------------------
# Prediction Button
if st.button("Predict"):
    if model_loaded:
        # تحويل الإدخالات لـ DataFrame
        input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol,
                                    fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=["age", "sex", "cp", "trestbps", "chol",
                                           "fbs", "restecg", "thalach", "exang",
                                           "oldpeak", "slope", "ca", "thal"])
        # التنبؤ
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ **High Risk**: The model predicts Heart Disease.\n\n"
                     "👉 **Next Step**: This is not a diagnosis. Please consult a cardiologist for further evaluation.")
        else:
            st.success("✅ **Low Risk**: The model predicts No Heart Disease.")
    else:
        st.warning("Model not loaded — cannot predict.")

# ---------------------------
# Visualization Example
st.subheader("Example Visualization")
df = pd.DataFrame({
    "Feature": ["Age", "Cholestoral", "Max HR"],
    "Value": [age, chol, thalach]
})
fig = px.bar(df, x="Feature", y="Value", title="Patient Feature Overview")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
st.caption("Built with Streamlit ❤️ — Predictions are estimates, not medical advice.")
