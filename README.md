# 💳 Credit Card Fraud Detection with Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning techniques. Due to the **highly imbalanced dataset**, special attention is given to **data preprocessing, resampling, model selection**, and **threshold tuning** to improve **recall on the minority class (fraudulent transactions)**.

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains **284,807 transactions**, with **492 fraudulent** (≈0.17%).
- Evaluation was done on the **original unbalanced dataset** (not SMOTE-balanced) to reflect real-world fraud rates.
- Additional version of the dataset: [Kaggle - Credit Card Fraud Detection (Yashpal Oswal)](https://www.kaggle.com/datasets/yashpaloswal/fraud-detection-credit-card)

## 🧹 Data Preprocessing
- Performed **standard scaling** (data is already PCA-transformed in original dataset).
- Used **SMOTE** to oversample the minority class (fraud) for model training.
- Ensured no data leakage by splitting before balancing.

## 🧠 ML Models Explored
- SVM
- LR (Logistic Regression)
- Neural Network
- Random Forest
- XGBoost (Final model)
- Custom threshold tuning to improve fraud recall

XGBoost was selected as the final model due to its superior performance on fraud recall after threshold tuning and its robustness with tabular data.

## 🧪 Final Metrics
- **Model:** XGBoost with default threshold (0.5)
- **Recall (Fraud):** ~65%
- **Precision (Fraud):** ~69%
- Significantly improved fraud detection performance compared to baseline RandomForest

## 📈 Visualizations
- Precision-Recall Curve
- Feature Importance Plot

## 🎯 Goal
Maximize **recall** while maintaining reasonable **precision**, making the system more effective in real-world fraud detection where missing a fraud is more costly than a false alert.
