import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import plotly.express as px

# ===== Page Config =====
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",  # قلب حقيقي
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===== Custom Style =====
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5; /* خلفية هادية */
        color: black; /* كل الكلام بالأسود */
    }
    .stButton>button {
        background-color: #d90429;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Header =====
st.title("❤️ Heart Disease Risk Predictor")
st.write("هذا التطبيق يساعدك على تقييم احتمالية الإصابة بأمراض القلب بناءً على بعض القياسات الطبية.")

# ===== Load Model =====
try:
    model = joblib.load("models/final_model.pkl")
    st.success("✅ Model Loaded Successfully")
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please place it in 'models/final_model.pkl'")
    st.stop()

# ===== Input Form =====
st.subheader("أدخل بياناتك الطبية:")

age = st.number_input("Age (العمر)", min_value=18, max_value=100, value=45)
sex = st.selectbox("Sex (الجنس)", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (نوع ألم الصدر)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (ضغط الدم أثناء الراحة)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (الكوليسترول)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (أقصى معدل ضربات قلب)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if sex == "Female":
    sex = 0
else:
    sex = 1

# ===== Prediction =====
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

if st.button("🔍 Predict Risk"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {prob:.2f}%")
        st.write("💡 **نصيحة**: يجب زيارة طبيب متخصص فورًا، ضبط نمط الحياة، متابعة ضغط الدم والكوليسترول.")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {prob:.2f}%")
        st.write("💡 **نصيحة**: استمر في نمط الحياة الصحي ومتابعة الفحوصات الدورية.")

# ===== Normal Ranges and Notes =====
st.subheader("ℹ️ Normal Ranges & Meaning of Features")
st.markdown("""
- **Age**: العمر المثالي بدون أمراض مزمنة.  
- **Blood Pressure**: طبيعي بين 90/60 إلى 120/80 مم زئبق.  
- **Cholesterol**: أقل من 200 mg/dl يعتبر مثالي.  
- **Max Heart Rate**: يختلف حسب العمر، تقريبًا 220 - العمر.  
- **Fasting Blood Sugar**: أقل من 120 mg/dl طبيعي.  
- **نصيحة عامة**: الأكل الصحي، الرياضة المنتظمة، متابعة ضغط الدم والكوليسترول.
""")

# ===== Optional Plot =====
st.subheader("📊 Example Chart")
df = pd.DataFrame({
    "Feature": ["Age", "Cholesterol", "Blood Pressure"],
    "Value": [age, chol, trestbps]
})
fig = px.bar(df, x="Feature", y="Value", title="User Metrics", color="Feature")
st.plotly_chart(fig, use_container_width=True)
