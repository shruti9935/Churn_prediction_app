import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.markdown("<h1 style='text-align: center; color: teal;'>\ud83d\udd0d Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Upload your customer data below to predict churn risk and explore insights:")

# File uploader section
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

# Only proceed if a file is uploaded
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("\ud83d\udcc4 Uploaded Data Preview:")
        st.dataframe(data.head())

        # Required columns
        required_cols = ['Gender', 'Age', 'CustomerId', 'Tenure', 'Balance', 'Satisfaction Score', 'EstimatedSalary']
        if not all(col in data.columns for col in required_cols):
            st.error("\u274c Uploaded CSV is missing required columns.")
        else:
            if data['Gender'].dtype == 'object':
                data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
            data = data[data['Gender'].isin([0, 1])]

            X_input = data[required_cols]
            X_scaled = scaler.transform(X_input)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            data['Churn Prediction'] = preds
            data['Churn Probability'] = (probs * 100).round(2)

            st.success("\u2705 Churn prediction completed.")
            st.dataframe(data[['CustomerId', 'Churn Prediction', 'Churn Probability']])

            # Churn Rate by Gender
            st.markdown("### \ud83d\udc65 Churn Rate by Gender")
            data['Gender Label'] = data['Gender'].map({0: 'Female', 1: 'Male'})
            gender_churn = data.groupby('Gender Label')['Churn Prediction'].mean().reset_index()
            fig1, ax1 = plt.subplots()
            sns.barplot(data=gender_churn, x='Gender Label', y='Churn Prediction', ax=ax1)
            ax1.set_ylabel("Churn Rate")
            ax1.set_xlabel("Gender")
            st.pyplot(fig1)

            # Churn Rate by Balance Tier
            st.markdown("### \ud83d\udcb0 Churn Rate by Balance Tier")
            data['Balance Tier'] = pd.cut(data['Balance'], bins=[0, 50000, 100000, 200000],
                                          labels=['Low', 'Mid', 'High'])
            balance_churn = data.groupby('Balance Tier')['Churn Prediction'].mean().reset_index()
            fig2, ax2 = plt.subplots()
            sns.barplot(data=balance_churn, x='Balance Tier', y='Churn Prediction', ax=ax2)
            ax2.set_ylabel("Churn Rate")
            ax2.set_xlabel("Balance Tier")
            st.pyplot(fig2)

            # Model Comparison Table
            st.markdown("### \ud83e\uddd0 Model Performance Comparison")
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
        st.error(f"\u274c Error processing file: {e}")

else:
    st.warning("\u26a0\ufe0f Please upload a CSV file to continue.")

