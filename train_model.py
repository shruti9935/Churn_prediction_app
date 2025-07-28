import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process
from sklearn.impute import SimpleImputer

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.markdown("<h1 style='text-align: center; color: teal;'>🔍 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Upload your customer data below to predict churn risk and explore insights:")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

# Utility: Fuzzy match columns
def match_column(df_columns, target_name, threshold=80):
    match, score = process.extractOne(target_name.lower(), [col.lower() for col in df_columns])
    if score >= threshold:
        for col in df_columns:
            if col.lower() == match:
                return col
    return None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(data.head())

        expected_features = {
            "Gender": 0,
            "Age": 35,
            "CustomerId": 0,
            "Tenure": 3,
            "Balance": 50000.0,
            "Satisfaction Score": 3,
            "EstimatedSalary": 100000.0
        }

        X_input = pd.DataFrame()
        missing_columns = []

        for col_name, default in expected_features.items():
            match = match_column(data.columns, col_name)
            if match:
                X_input[col_name] = data[match]
            else:
                missing_columns.append(col_name)
                X_input[col_name] = default

        if X_input['Gender'].dtype == 'object':
            X_input['Gender'] = X_input['Gender'].map({'Female': 0, 'Male': 1})

        # Handle missing values (NaNs)
        imputer = SimpleImputer(strategy='mean')
        X_input_imputed = pd.DataFrame(imputer.fit_transform(X_input), columns=X_input.columns)

        X_scaled = scaler.transform(X_input_imputed)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        data['Churn Prediction'] = preds
        data['Churn Probability'] = (probs * 100).round(2)

        st.success("✅ Churn prediction completed.")
        display_cols = ['Churn Prediction', 'Churn Probability']
        if 'CustomerId' in data.columns:
            display_cols.insert(0, 'CustomerId')
        st.dataframe(data[display_cols])

        if missing_columns:
            st.warning(f"⚠️ Using default values for missing columns: {', '.join(missing_columns)}")

        # Gender plot
        st.markdown("### 👥 Churn Rate by Gender")
        data['Gender Label'] = X_input_imputed['Gender'].map({0: 'Female', 1: 'Male'})
        gender_churn = data.groupby('Gender Label')['Churn Prediction'].mean().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=gender_churn, x='Gender Label', y='Churn Prediction', ax=ax1)
        ax1.set_ylabel("Churn Rate")
        ax1.set_xlabel("Gender")
        st.pyplot(fig1)

        # Balance tier plot
        st.markdown("### 💰 Churn Rate by Balance Tier")
        data['Balance Tier'] = pd.cut(X_input_imputed['Balance'], bins=[0, 50000, 100000, 200000],
                                      labels=['Low', 'Mid', 'High'])
        balance_churn = data.groupby('Balance Tier')['Churn Prediction'].mean().reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(data=balance_churn, x='Balance Tier', y='Churn Prediction', ax=ax2)
        ax2.set_ylabel("Churn Rate")
        ax2.set_xlabel("Balance Tier")
        st.pyplot(fig2)

        # Model comparison
        st.markdown("### 🧐 Model Performance Comparison")
        comparison_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy": [0.73, 0.76, 0.78],
            "Precision": [0.65, 0.70, 0.72],
            "Recall": [0.60, 0.68, 0.69],
            "F1 Score": [0.62, 0.69, 0.71]
        })
        comparison_df = comparison_df.round(3)
        st.dataframe(comparison_df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.warning("⚠️ Please upload a CSV file to continue.")
