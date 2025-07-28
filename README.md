# Customer Churn Prediction App 🧠📉

An interactive machine learning web app to predict customer churn, built using Python, Streamlit, and scikit-learn — and deployed on Streamlit Cloud.

## 🔍 Project Overview
This end-to-end project predicts whether a customer is likely to churn using demographic and behavioral data. It includes:

- 📊 Model comparison (Logistic Regression, Random Forest, XGBoost)
- 📥 CSV upload for batch prediction
- 🔁 Manual prediction via UI controls
- 📈 Churn probability and visual segment insights
- 🧠 SHAP explainability (optional extension)
- ✅ Deployed live using Streamlit Cloud

---

## 🚀 Try the App Live
🔗 [https://churnpredictionapp-jtmmqkbthjfcdcufpkcdt3.streamlit.app/]

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn)
- **scikit-learn** (logistic regression, random forest)
- **XGBoost**
- **SMOTE** (imbalanced-learn)
- **Streamlit** (UI + deployment)
- **SHAP** (optional explainability)
- **Joblib** (model serialization)

---

## 📁 Project Structure
```
churn-prediction-app/
│
├── churn_app.py                # Main Streamlit app file
├── churn_model.pkl            # Trained model file
├── scaler.pkl                 # StandardScaler used during training
├── requirements.txt           # Dependencies for Streamlit Cloud
└── README.md                  # This file
```

---

## 📦 Features

### ✅ Manual Prediction
- Select gender, age, tenure, balance, etc.
- See churn probability and predicted outcome

### ✅ Batch Prediction from CSV
- Upload customer data
- See churn predictions for each customer
- Download prediction results

### ✅ Visual Churn Insights
- Churn rate by gender
- Churn rate by balance tier

### ✅ Model Comparison Table
- Logistic Regression vs Random Forest vs XGBoost
- Accuracy, Precision, Recall, F1 Score

---

## 📤 Sample CSV Format
```csv
Gender,Age,CustomerId,Tenure,Balance,Satisfaction Score,EstimatedSalary
Female,42,15634602,2,1115.0,2,101348.88
Male,36,15647311,1,0.0,1,112542.58
```
👉 Use the format above when uploading your file.

---

## 📥 Installation (Run Locally)
```bash
# Clone the repository
$ git clone https://github.com/YOUR_USERNAME/churn-prediction-app.git
$ cd churn-prediction-app

# Install dependencies
$ pip install -r requirements.txt

# Run the Streamlit app
$ streamlit run churn_app.py
```

---

## 📌 Acknowledgements
Thanks to the open-source libraries and Streamlit team for making this project possible. 
Special shoutout to ChatGPT for guidance and debugging support 🚀

---



💬 Feedback or suggestions? Drop a comment or create an issue.

---

### 🔗 License
MIT License © SHRUTI
