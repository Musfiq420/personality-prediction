import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('personality_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🧠 Extrovert vs Introvert Personality Predictor")

# User input form
st.subheader("Enter your behaviors:")

# Input fields
time_spent_alone = st.slider("Hours spent alone daily", 0, 11, 4)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Social event attendance (0–10)", 0, 10, 5)
going_outside = st.slider("How often do you go outside? (0–7)", 0, 7, 3)
drained = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("Number of close friends", 0, 15, 5)
post_frequency = st.slider("Social media post frequency", 0, 10, 5)

# Predict button
if st.button("Predict Personality"):
    # Encode input
    input_data = np.array([
        time_spent_alone,
        1 if stage_fear == "Yes" else 0,
        social_event_attendance,
        going_outside,
        1 if drained == "Yes" else 0,
        friends_circle_size,
        post_frequency
    ]).reshape(1, -1)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Output
    st.subheader("Prediction:")
    st.success("🔵 You are likely an **Introvert**" if prediction == 1 else "🟢 You are likely an **Extrovert**")
