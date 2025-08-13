import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("best_fraud_model_xgb.pkl")

# Load the feature names your model expects
expected_features = model.get_booster().feature_names

# Threshold value (match your notebook's setting)
THRESHOLD = 0.5  

def predict_with_threshold(model, X, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]  # Probability of fraud
    return (proba >= threshold).astype(int)

st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Check if all required features are present
    missing = set(expected_features) - set(data.columns)
    if missing:
        st.error(f"Your file is missing these required columns: {missing}")
    else:
        # Keep only the expected features in the correct order
        X = data[expected_features]

<<<<<<< HEAD
        # Make predictions using threshold tuning
        predictions = predict_with_threshold(model, data, threshold=THRESHOLD)
        data["Fraud Prediction"] = predictions
=======
        # Check if 'Class' column exists for evaluation
        if "Class" in data.columns:
            y_true = data["Class"]
>>>>>>> 47ae566b1d55dc3131d95a400770be3147b575b1

            # Make predictions
            predictions = model.predict(X)
            data["Fraud Prediction"] = predictions

            # Show predictions
            st.write("Predictions:", data)

            # Model performance
            st.subheader("📊 Model Performance")
            report = classification_report(y_true, predictions, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

            # Confusion matrix
            cm = confusion_matrix(y_true, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        else:
            # If no labels provided, just make predictions
            predictions = model.predict(X)
            data["Fraud Prediction"] = predictions
            st.write("Predictions:", data)

        # Allow download of results
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
