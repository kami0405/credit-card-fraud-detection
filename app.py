import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_fraud_model_xgb.pkl")

st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Make predictions
    predictions = model.predict(data)
    data["Fraud Prediction"] = predictions

    st.write("Predictions:", data)

    # Allow download of results
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
