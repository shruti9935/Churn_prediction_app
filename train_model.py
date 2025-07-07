import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("🔍 Customer Churn Predictor")
st.write("Enter customer details to predict churn risk:")

# Input fields for customer data
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", value=50000.0)
satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
salary = st.number_input("Estimated Salary", value=100000.0)

# Encode gender: Female = 0, Male = 1
gender_encoded = 1 if gender == "Male" else 0

# Dummy values for CustomerId and Surname
input_data = np.array([gender_encoded, age, 0, tenure, 0, balance, satisfaction, salary])

if st.button("Predict Churn"):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown(f"### 🔢 Churn Probability: `{prob:.2%}`")
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

st.markdown("---")
st.subheader("📤 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    import pandas as pd

    try:
        # Read the CSV
        data = pd.read_csv(uploaded_file)

        # Check for required columns
        required_cols = ['Gender', 'Age', 'CustomerId', 'Tenure', 'Balance', 'Satisfaction Score', 'EstimatedSalary']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        else:
            st.write("📄 Uploaded Data Preview:")
            st.dataframe(data.head())

            # Preprocess
            X_input = data[required_cols].copy()
            X_input['Gender'] = X_input['Gender'].map({'Female': 0, 'Male': 1})  # Encode gender

            # Predict
            X_scaled = scaler.transform(X_input)
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            # Add results to original data
            data['Churn Prediction'] = preds
            data['Churn Probability'] = (probs * 100).round(2)

            st.success("✅ Churn prediction completed.")
            st.dataframe(data[['CustomerId', 'Churn Prediction', 'Churn Probability']])

            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", data=csv, file_name="churn_predictions.csv", mime='text/csv')

            # --------------------
            # 📊 KPI Dashboard
            st.markdown("---")
            st.subheader("📊 Churn Insights Dashboard")

            total = len(data)
            churned = data['Churn Prediction'].sum()
            churn_rate = churned / total
            high_risk_count = (data['Churn Probability'] > 70).sum()
            avg_salary_churned = data[data['Churn Prediction'] == 1]['EstimatedSalary'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", total)
            col2.metric("Churn Rate", f"{churn_rate:.2%}")
            col3.metric("High-Risk (>70%)", high_risk_count)

            st.metric("Avg Salary of Churned Customers", f"${avg_salary_churned:,.2f}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("---")
st.subheader("📊 Churn Rate by Segment")

if uploaded_file and 'Churn Prediction' in data.columns:
    try:
        # Group 1: Churn by Gender
        gender_churn = data.groupby('Gender')['Churn Prediction'].mean().reset_index()
        gender_churn['Gender'] = gender_churn['Gender'].map({0: 'Female', 1: 'Male'})

        st.markdown("#### 👥 Churn Rate by Gender")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=gender_churn, x='Gender', y='Churn Prediction', ax=ax1)
        ax1.set_ylabel("Churn Rate")
        st.pyplot(fig1)

        # Group 2: Churn by Balance Tier
        data['Balance Tier'] = pd.cut(data['Balance'], bins=[0, 50000, 100000, 200000],
                                      labels=['Low', 'Mid', 'High'])
        balance_churn = data.groupby('Balance Tier')['Churn Prediction'].mean().reset_index()

        st.markdown("#### 💰 Churn Rate by Balance Tier")
        fig2, ax2 = plt.subplots()
        sns.barplot(data=balance_churn, x='Balance Tier', y='Churn Prediction', ax=ax2)
        ax2.set_ylabel("Churn Rate")
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"Couldn't create segment insights: {e}")
st.markdown("### 🤖 Model Performance Comparison")

# Define the DataFrame manually (copy from your notebook)
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.73, 0.76, 0.78],
    "Precision": [0.65, 0.70, 0.72],
    "Recall": [0.60, 0.68, 0.69],
    "F1 Score": [0.62, 0.69, 0.71]
})

comparison_df = comparison_df.round(3)
st.dataframe(comparison_df)
