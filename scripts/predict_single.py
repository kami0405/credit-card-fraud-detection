def predict_transaction(transaction_df, model, scaler, threshold=0.5):
    scaled = scaler.transform(transaction_df)
    prob = model.predict_proba(scaled)[:, 1]
    return "Fraud" if prob >= threshold else "Not Fraud", prob
