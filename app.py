# app.py (Streamlit)
import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())
    model = joblib.load('model.joblib')
    predictions = model.predict(data)
    st.write("Predictions:", predictions)
