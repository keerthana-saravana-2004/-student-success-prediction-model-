# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model_pipeline.pkl')

st.title("🎓 Student Approval Prediction")

# Input form
with st.form("form"):
    department = st.selectbox("Department", ["CSE", "EEE", "ME"])
    gpa_level = st.selectbox("GPA Level", ["Low", "Medium", "High"])
    project_submitted = st.radio("Project Submitted", ["Yes", "No"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        'Department': department,
        'GPALevel': gpa_level,
        'ProjectSubmitted': project_submitted
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Not Approved"
    st.subheader(f"Prediction: {result}")
