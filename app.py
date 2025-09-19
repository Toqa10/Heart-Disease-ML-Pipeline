# app.py
import os
import io
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Plotly optional (fallback to simple bar if not installed)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Styles (black text, light background) ----------------
st.markdown(
    """
    <style>
    /* Background and text color */
    .stApp {
        background-color: #f7f7f7;
        color: #000000;
    }
    /* Buttons styling */
    .stButton>button {
        background-color: #d90429;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
    }
    /* Headings color */
    h1, h2, h3, h4 {
        color: #000000;
    }
    /* Make dataframe text black */
    .stDataFrame td, .stDataFrame th {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Header ----------------
st.title("تطبيق تقييم خطر أمراض القلب")
st.write("هذا التطبيق يساعد على تقييم احتمالية وجود مرض قلبي بناءً على قياسات طبية شائعة. النتائج مجرد تقدير ولا تُعد تشخيصًا طبيًا.")

# ---------------- Model loading / upload ----------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

model = None
model_loaded = False

# If model exists on disk, try to load it
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True
        st.success("✅ تم تحميل الموديل من المسار: models/final_model.pkl")
    except Exception as e:
        st.warning("⚠️ وُجد ملف موديل لكن حدث خطأ عند تحميله. يمكنك إعادة رفعه من هنا أو استبداله.")
        st.error(str(e))

# Allow user to upload model if not loaded
if not model_loaded:
    st.info("لم يتم العثور على موديل جاهز. يمكنك رفع ملف الموديل هنا (final_model.pkl).")
    uploaded_model = st.file_uploader("رفع ملف الموديل (.pkl)", type=["pkl", "joblib"], accept_multiple_files=False)
    if uploaded_model is not None:
        try:
            # save uploaded model to models/final_model.pkl
            bytes_data = uploaded_model.read()
            with open(MODEL_PATH, "wb") as f:
                f.write(bytes_data)
            model = joblib.load(MODEL_PATH)
            model_loaded = True
            st.success("✅ تم رفع وحفظ الموديل بنجاح في models/final_model.pkl")
        except Exception as e:
            st.error("حدث خطأ عند حفظ أو تحميل الموديل: " + str(e))

# ---------------- Sidebar: quick instructions ----------------
with st.sidebar:
    st.header("تعليمات سريعة")
    st.write("""
    1. تأكدي من وجود موديل مدرّب (pipeline) في `models/final_model.pkl`.  
    2. إن لم يكن، ارفعي ملف الموديل من هنا.  
    3. أدخلي القيم ثم اضغطي Predict.  
    """)
    st.markdown("---")
    st.write("**ملاحظة:** النموذج يفترض أن الـ pipeline يتضمن نفس ترتيب وسمات (features) المستخدمة في التدريب:") 
    st.write("`['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']`")

# ---------------- Image (real heart) ----------------
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Heart_anatomy_labeled.svg/800px-Heart_anatomy_labeled.svg.png",
    width=200,
    caption="صورة توضيحية للقلب"
)

# ---------------- Normal ranges & feature meanings ----------------
st.subheader("ℹ️ Normal Ranges & معنى كل متغير")
info_df = pd.DataFrame({
    "الميزة (Feature)": [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ],
    "النطاق الطبيعي (تقريبًا)": [
        "20–80 سنة", "0=Female, 1=Male", "0–3 (أنواع ألم الصدر)", "90–120 mmHg", "<200 mg/dl",
        "0 (≤120mg/dl) أو 1 (>120mg/dl)", "0–2 (نتيجة ECG)", "140–190 bpm (يعتمد على العمر)", "0 أو 1", "0.0–1.5 تقريبًا",
        "0–2 (نوع الميل في ST)", "0–3 (عدد الأوعية الملونة)", "0–3 (حالة الثال)"
    ],
    "ماذا يعني/ماذا نفهم منه؟": [
        "عمر المريض", "الجنس (ذكر/أنثى)", "نوع ألم الصدر (نشاط/غيره)", "ضغط الدم أثناء الراحة",
        "كوليسترول الدم", "مستوى السكر بعد صيام", "نتائج رسم القلب الراحة", "أقصى معدل ضربات قلب أثناء الإجهاد",
        "هل سبب التمرين ذبحة؟", "انخفاض ST بعد التمرين (قيمة تقييمية)", "ميل مقطع ST", "أوعية القلب الرئيسية التي تظهر بالتصوير",
        "تصنيف ثال/ثلاسيميا في البيانات"
    ]
})
with st.expander("عرض الشرح المفصل للنطاقات والمعاني"):
    st.dataframe(info_df, use_container_width=True)

# ---------------- Input form ----------------
st.subheader("أدخل بيانات المريض")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (العمر)", min_value=18, max_value=120, value=50)
    sex_str = st.selectbox("Sex (الجنس)", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=0)
    trestbps = st.number_input("Resting Blood Pressure (trestbps, mmHg)", min_value=70, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol, mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)

with col2:
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], index=0)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2], index=1)
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("Thal (thalassemia)", [0, 1, 2, 3], index=0)

sex = 1 if sex_str == "Male" else 0

# ---------------- Prediction button ----------------
st.markdown("---")
predict_btn = st.button("🔍 Predict Risk")

if predict_btn:
    if not model_loaded:
        st.error("لا يوجد موديل محمّل. ارفع ملف موديل صالح (final_model.pkl) في الشريط الجانبي أولًا.")
    else:
        # Create dataframe with expected column order
        input_df = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

        # Try prediction and probability gracefully
        try:
            pred = model.predict(input_df)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0][1]  # probability of class 1
            elif hasattr(model, "decision_function"):
                # rough mapping: use logistic on decision_function is not implemented here; fallback to None
                proba = None

            # Show results
            if pred == 1:
                st.error(f"⚠️ التوقع: **خطر مرتفع** لوجود مرض قلبي.")
            else:
                st.success("✅ التوقع: **خطر منخفض** لوجود مرض قلبي.")

            if proba is not None:
                st.write(f"احتمالية وجود المرض حسب النموذج: **{proba * 100:.2f}%**")
            else:
                st.write("النموذج لا يدعم إرجاع احتمال (predict_proba).")

            # Actionable advice
            st.markdown("### ماذا تفعل إذا كانت النتيجة High Risk؟")
            st.markdown("""
            - لا تعتبر هذه النتيجة تشخيصًا نهائيًا، لكنها مؤشر لزيارة طبيب قلب (Cardiologist) لإجراء فحوصات إضافية مثل ECG، Echo، أو اختبارات إجهاد.  
            - تحسينات فورية ممكنة: التوقف عن التدخين، تقليل الأطعمة الغنية بالدهون المشبعة، ممارسة نشاط بدني منتظم، مراقبة وضبط ضغط الدم والكوليسترول.  
            - في الحالات الطارئة (ألم صدر مفاجئ، ضيق تنفُّس شديد، إغماء) اتصلي بالطوارئ فورًا.
            """)

        except Exception as e:
            st.error("حدث خطأ أثناء التنبؤ: " + str(e))

# ---------------- Visualization (simple) ----------------
st.markdown("---")
st.subheader("عرض سريع للقياسات الخاصة بالمستخدم")

display_df = pd.DataFrame({
    "Feature": ["age", "chol", "trestbps", "thalach", "oldpeak"],
    "Value": [age, chol, trestbps, thalach, oldpeak]
})

if PLOTLY_AVAILABLE:
    fig = px.bar(display_df, x="Feature", y="Value", title="Patient Metrics Overview",
                 color="Feature", color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.table(display_df)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("ملاحظة: هذا التطبيق للتجريب والتعليم فقط — النتائج ليست بديلاً عن الاستشارة الطبية.")
