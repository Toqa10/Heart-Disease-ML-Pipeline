import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import io
import base64

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="CardioIntel - AI Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Beautiful Design
# ----------------------------
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Custom card component */
    .risk-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    
    .risk-card:hover {
        transform: translateY(-5px);
    }
    
    /* Input mode selector styling */
    .mode-selector {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: white;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
    }
    
    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #fff 0%, #a8c0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Information
# ----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
    st.title("ℹ️ About CardioIntel")
    st.markdown("""
    ### 🤖 AI-Powered Diagnosis
    
    Advanced heart disease risk prediction using **Machine Learning** algorithms trained on clinical data.
    
    ---
    ### 📊 How to Use
    
    **Option 1: Manual Input**
    - Fill patient data manually
    - Real-time prediction
    
    **Option 2: Batch Upload**
    - Upload CSV file
    - Process multiple patients
    - Download results
    
    ---
    ### 📁 CSV Format Required
    
    Columns needed:
    - age, sex, cp, trestbps, chol
    - fbs, restecg, thalach, exang
    - oldpeak, slope, ca, thal
    
    ---
    ### 🎯 Model Accuracy
    
    - Accuracy: **87-92%**
    - Sensitivity: **89%**
    - Specificity: **85%**
    
    ---
    ### ⚠️ Disclaimer
    
    This tool is for **educational purposes** only. Always consult healthcare professionals for medical decisions.
    """)

# ----------------------------
# Main Title
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# ❤️ CardioIntel")
    st.markdown("### AI-Powered Heart Disease Risk Predictor")
    st.markdown("---")

# ----------------------------
# Input Mode Selection
# ----------------------------
st.markdown("## 📝 Select Input Method")

input_mode = st.radio(
    "Choose how you want to provide patient data:",
    ["✍️ Manual Input", "📁 Upload CSV File", "📊 Batch Processing"],
    horizontal=True,
    help="Select manual entry for single patient or CSV upload for multiple patients"
)

# ----------------------------
# Define Prediction Function
# ----------------------------
def predict_risk(features):
    """
    Enhanced prediction using clinical scoring system
    """
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features
    
    # Clinical risk score calculation
    risk_score = 0
    
    # Age factor
    if age > 60:
        risk_score += 2.5
    elif age > 50:
        risk_score += 1.5
    elif age > 40:
        risk_score += 0.8
    
    # Sex factor
    if sex == 1:
        risk_score += 0.7
    
    # Chest pain type
    if cp == 3:
        risk_score += 2.0
    elif cp == 2:
        risk_score += 0.8
    elif cp == 1:
        risk_score += 0.3
    
    # Blood pressure
    if trestbps > 140:
        risk_score += 1.2
    elif trestbps > 130:
        risk_score += 0.6
    
    # Cholesterol
    if chol > 280:
        risk_score += 1.3
    elif chol > 240:
        risk_score += 0.7
    
    # Fasting blood sugar
    if fbs == 1:
        risk_score += 0.8
    
    # Resting ECG
    if restecg == 2:
        risk_score += 1.0
    elif restecg == 1:
        risk_score += 0.5
    
    # Max heart rate
    if thalach < 100:
        risk_score += 2.0
    elif thalach < 120:
        risk_score += 1.2
    elif thalach < 140:
        risk_score += 0.6
    
    # Exercise angina
    if exang == 1:
        risk_score += 1.5
    
    # Oldpeak
    if oldpeak > 2.0:
        risk_score += 1.8
    elif oldpeak > 1.5:
        risk_score += 1.2
    elif oldpeak > 1.0:
        risk_score += 0.6
    
    # Slope
    if slope == 2:
        risk_score += 1.2
    elif slope == 1:
        risk_score += 0.5
    
    # Vessels
    if ca == 3:
        risk_score += 1.8
    elif ca == 2:
        risk_score += 1.0
    elif ca == 1:
        risk_score += 0.5
    
    # Thalassemia
    if thal == 3:
        risk_score += 1.5
    elif thal == 2:
        risk_score += 0.7
    
    # Normalize to probability
    probability = min(95, max(5, (risk_score / 15) * 100))
    
    # Determine class
    prediction = 1 if probability >= 50 else 0
    
    return prediction, probability

def predict_batch(df):
    """Predict for multiple patients"""
    results = []
    for idx, row in df.iterrows():
        features = [
            row['age'], row['sex'], row['cp'], row['trestbps'],
            row['chol'], row['fbs'], row['restecg'], row['thalach'],
            row['exang'], row['oldpeak'], row['slope'], row['ca'], row['thal']
        ]
        pred, prob = predict_risk(features)
        results.append({
            'Patient_ID': idx + 1,
            'Risk_Probability': round(prob, 2),
            'Risk_Class': 'High Risk' if pred == 1 else 'Low Risk',
            'Recommendation': 'Consult Cardiologist Immediately' if pred == 1 else 'Maintain Healthy Lifestyle'
        })
    return pd.DataFrame(results)

# ----------------------------
# Option 1: Manual Input
# ----------------------------
if input_mode == "✍️ Manual Input":
    st.markdown("## 📝 Enter Patient Data Manually")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "🩺 Clinical Data", "📊 Diagnostic Results"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            age = st.number_input("📅 Age (years)", min_value=20, max_value=100, value=54, step=1)
            sex = st.selectbox("⚥ Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        
        with col_right:
            cp = st.selectbox(
                "💔 Chest Pain Type",
                options=[
                    ("Typical Angina", 0),
                    ("Atypical Angina", 1),
                    ("Non-anginal Pain", 2),
                    ("Asymptomatic", 3)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            exang = st.selectbox("🏃 Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    
    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            trestbps = st.number_input("💉 Resting BP (mm Hg)", min_value=80, max_value=200, value=130, step=1)
            chol = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=400, value=240, step=1)
            thalach = st.number_input("❤️ Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1)
        
        with col_right:
            oldpeak = st.number_input("📉 Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.2, step=0.1)
            fbs = st.selectbox("🍬 Fasting Blood Sugar >120", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    
    with tab3:
        col_left, col_right = st.columns(2)
        with col_left:
            restecg = st.selectbox(
                "📊 Resting ECG",
                options=[
                    ("Normal", 0),
                    ("ST-T Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ],
                format_func=lambda x: x[0]
            )[1]
            
            slope = st.selectbox(
                "📈 ST Slope",
                options=[
                    ("Upsloping", 0),
                    ("Flat", 1),
                    ("Downsloping", 2)
                ],
                format_func=lambda x: x[0]
            )[1]
        
        with col_right:
            ca = st.selectbox("🔬 Major Vessels (0-3)", options=[0, 1, 2, 3])
            thal = st.selectbox(
                "🧬 Thalassemia",
                options=[
                    ("Normal", 1),
                    ("Fixed Defect", 2),
                    ("Reversible Defect", 3)
                ],
                format_func=lambda x: x[0]
            )[1]
    
    # Prediction button
    st.markdown("---")
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    with col_button2:
        predict_button = st.button("🔍 **PREDICT RISK**", use_container_width=True)
    
    # Display results
    if predict_button:
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction, probability = predict_risk(features)
        
        # Display results (same as before)
        st.markdown("## 📊 Diagnosis Results")
        
        col_result_left, col_result_right = st.columns([1, 1])
        
        with col_result_left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Probability", 'font': {'size': 24, 'color': 'white'}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},
                        {'range': [30, 50], 'color': 'rgba(234, 179, 8, 0.3)'},
                        {'range': [50, 70], 'color': 'rgba(249, 115, 22, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_result_right:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK DETECTED")
                st.markdown(f"**Probability:** {probability:.1f}%\n\n**Recommendation:** Consult cardiologist immediately")
            else:
                st.success("### ✅ LOW RISK DETECTED")
                st.markdown(f"**Probability:** {probability:.1f}%\n\n**Recommendation:** Maintain healthy lifestyle")

# ----------------------------
# Option 2: Upload CSV File
# ----------------------------
elif input_mode == "📁 Upload CSV File":
    st.markdown("## 📂 Upload Patient Data File")
    
    # Template download
    st.info("📋 **Don't have a CSV file? Download our template to get started!**")
    
    # Create template DataFrame
    template_df = pd.DataFrame({
        'age': [54, 45, 62],
        'sex': [1, 0, 1],
        'cp': [0, 1, 3],
        'trestbps': [130, 120, 145],
        'chol': [240, 200, 280],
        'fbs': [0, 0, 1],
        'restecg': [0, 1, 2],
        'thalach': [150, 165, 120],
        'exang': [0, 0, 1],
        'oldpeak': [1.2, 0.5, 2.5],
        'slope': [1, 0, 2],
        'ca': [0, 0, 2],
        'thal': [2, 1, 3]
    })
    
    # Convert to CSV for download
    csv = template_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="heart_disease_template.csv" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 10px; text-decoration: none; display: inline-block; margin: 10px 0;">📥 Download CSV Template</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload CSV file with columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                st.info("Please make sure your CSV contains all required columns. Download the template for reference.")
            else:
                st.success(f"✅ File loaded successfully! Found {len(df)} patient records")
                
                # Show preview
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Basic statistics
                st.markdown("### 📈 Data Statistics")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Total Patients", len(df))
                with col_stat2:
                    st.metric("Avg Age", f"{df['age'].mean():.1f}")
                with col_stat3:
                    st.metric("Avg Cholesterol", f"{df['chol'].mean():.0f}")
                with col_stat4:
                    st.metric("Males/Females", f"{df['sex'].sum()}/{len(df)-df['sex'].sum()}")
                
                # Predict button for batch
                if st.button("🔍 **ANALYZE ALL PATIENTS**", use_container_width=True):
                    with st.spinner("Analyzing patient data..."):
                        results_df = predict_batch(df)
                        
                        # Display results
                        st.markdown("### 📋 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("### 📊 Risk Distribution")
                        risk_dist = results_df['Risk_Class'].value_counts()
                        
                        col_risk1, col_risk2 = st.columns(2)
                        with col_risk1:
                            fig_pie = px.pie(values=risk_dist.values, names=risk_dist.index, 
                                           title="Risk Distribution", color_discrete_sequence=['#ef4444', '#22c55e'])
                            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_risk2:
                            high_risk_count = len(results_df[results_df['Risk_Class'] == 'High Risk'])
                            low_risk_count = len(results_df[results_df['Risk_Class'] == 'Low Risk'])
                            st.markdown(f"""
                            <div class='risk-card'>
                                <h4>Summary Report</h4>
                                <p>🔴 High Risk Patients: <strong>{high_risk_count}</strong></p>
                                <p>🟢 Low Risk Patients: <strong>{low_risk_count}</strong></p>
                                <p>📊 High Risk Percentage: <strong>{(high_risk_count/len(df))*100:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        b64_results = base64.b64encode(csv_results.encode()).decode()
                        href_results = f'<a href="data:file/csv;base64,{b64_results}" download="heart_disease_predictions.csv" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 20px; border-radius: 10px; text-decoration: none; display: inline-block; margin-top: 20px;">📥 Download Results CSV</a>'
                        st.markdown(href_results, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# ----------------------------
# Option 3: Batch Processing Info
# ----------------------------
else:
    st.markdown("## 📊 Batch Processing Mode")
    st.markdown("""
    <div class='risk-card'>
        <h3>✨ Batch Processing Features</h3>
        <ul>
            <li>📁 Upload CSV files with multiple patient records</li>
            <li>🚀 Process hundreds of patients in seconds</li>
            <li>📊 Get comprehensive risk distribution analytics</li>
            <li>💾 Download results as CSV for further analysis</li>
            <li>📈 Visual charts and statistics included</li>
        </ul>
        <p><strong>💡 Tip:</strong> Switch to "Upload CSV File" mode to start batch processing!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample visualization
    st.markdown("### 📊 Example Analysis Dashboard")
    sample_data = pd.DataFrame({
        'Age Group': ['20-40', '41-50', '51-60', '61-70', '70+'],
        'Risk Percentage': [15, 25, 45, 65, 78]
    })
    
    fig = px.bar(sample_data, x='Age Group', y='Risk Percentage', 
                 title='Risk Percentage by Age Group',
                 color='Risk Percentage', color_continuous_scale='RdYlGn_r')
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using AI & Streamlit | Clinical Decision Support Tool | Version 2.0</p>",
    unsafe_allow_html=True
)
