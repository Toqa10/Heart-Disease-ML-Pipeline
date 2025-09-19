import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# --- بالونات ---
st.balloons()

# --- CSS للحركات والألوان ---
st.markdown("""
<style>
/* خلفية متدرجة متحركة */
.reportview-container {
    background: linear-gradient(45deg, #f3ec78, #af4261, #4B4BFF, #4BFF4B);
    background-size: 800% 800%;
    animation: gradientBG 20s ease infinite;
}

@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* عنوان متغير اللون */
h1, h2, h3 {
  animation: color-change 6s infinite;
}

@keyframes color-change {
  0% {color: #FF4B4B;}
  25% {color: #FFB14B;}
  50% {color: #4BFF4B;}
  75% {color: #4B4BFF;}
  100% {color: #FF4B4B;}
}

/* تنسيق الأزرار */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 12px;
    font-size: 18px;
    height: 3em;
    width: 100%;
    transition: 0.4s;
}

.stButton>button:hover {
    background-color: #4B4BFF;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# --- العنوان ---
st.title("💓 Heart Disease Risk Prediction (Colorful UI)")

# --- تحميل النموذج ---
model = joblib.load("models/final_model.pkl")

# --- إدخال البيانات من المستخدم ---
st.header("Enter Patient Data:")

age = st.number_input("Age", 20, 100, 50)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)

if st.button("Predict Risk"):
    with st.spinner("Analyzing..."):
        X = np.array([[age, cholesterol, max_heart_rate, resting_bp]])
        prediction = model.predict(X)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# --- مثال على رسم تفاعلي بالألوان ---
st.subheader("Sample Heart Disease Trends")
sample_df = pd.DataFrame({
    "Age": np.random.randint(30, 80, 50),
    "Cholesterol": np.random.randint(150, 300, 50),
    "Risk": np.random.choice(["High", "Low"], 50)
})
fig = px.scatter(sample_df, x="Age", y="Cholesterol",
                 color="Risk",
                 size_max=10,
                 title="Age vs Cholesterol with Risk Category",
                 color_discrete_map={"High":"#FF4B4B","Low":"#4BFF4B"})
st.plotly_chart(fig)
