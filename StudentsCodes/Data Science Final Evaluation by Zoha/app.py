import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# Load trained model
# ----------------------
model = joblib.load("stress_model.pkl")  # yeh aapka saved Random Forest model

# ----------------------
# Page layout
# ----------------------
st.set_page_config(page_title="Student Stress Predictor", layout="wide")
st.title("📊 Student Stress Level Prediction Dashboard")

# ----------------------
# Sidebar - Input Features
# ----------------------
st.sidebar.header("Enter Student Details")

study_hours = st.sidebar.slider("Study Hours per Day", 0, 16, 5)
sleep_hours = st.sidebar.slider("Sleep Hours per Day", 0, 12, 7)
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 3.0, 0.1)
total_activity = st.sidebar.slider("Total Activity Hours", 0, 16, 3)
physical_activity = st.sidebar.slider("Physical Activity Hours", 0, 5, 1)
social_hours = st.sidebar.slider("Social Hours per Day", 0, 5, 1)
extracurricular = st.sidebar.slider("Extracurricular Hours", 0, 5, 1)

study_sleep_ratio = study_hours / max(1, sleep_hours)

# ----------------------
# Create input DataFrame
# ----------------------
input_df = pd.DataFrame({
    "Study_Hours_Per_Day": [study_hours],
    "Study_Sleep_Ratio": [study_sleep_ratio],
    "Sleep_Hours_Per_Day": [sleep_hours],
    "GPA": [gpa],
    "Total_Activity_Hours": [total_activity],
    "Physical_Activity_Hours_Per_Day": [physical_activity],
    "Social_Hours_Per_Day": [social_hours],
    "Extracurricular_Hours_Per_Day": [extracurricular]
})

# ----------------------
# Predict Button
# ----------------------
if st.button("Predict Stress Level"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)
    
    stress_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    st.subheader(f"Predicted Stress Level: {stress_mapping[pred]}")
    
    st.write("Prediction Probabilities:")
    proba_df = pd.DataFrame(pred_proba, columns=["Low", "Moderate", "High"])
    st.dataframe(proba_df)

# ----------------------
# Feature Importance
# ----------------------
st.subheader("Feature Importance (Random Forest)")

feature_importances = model.feature_importances_
features = input_df.columns
fi_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
fi_df = fi_df.sort_values(by="Importance", ascending=False)

# Bar plot
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# ----------------------
# Optional: Dataset EDA
# ----------------------
st.subheader("Sample Data Insights")
st.write("You can show dataset distribution, boxplots, or correlations here.")
