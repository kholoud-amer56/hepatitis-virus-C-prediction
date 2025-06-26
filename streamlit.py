import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Hepatitis C Prediction",
    page_icon="🩺",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Hepatitis C Prediction Dashboard")
st.markdown("This app predicts Hepatitis C infection based on blood test results using different ML models.")

# Load Data
@st.cache_data
def load_data():
    file_path = "blood_test_virus_c_dataset.csv"
    df = pd.read_csv(file_path)

    if 'Category' not in df.columns:
        st.error("'Category' column not found in the dataset.")
        return pd.DataFrame()

    df['Hepatitis_C'] = df['Category'].apply(lambda x: 1 if "1=Virus C" in str(x) else 0)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    return df

df = load_data()

# Sidebar input
st.sidebar.header("Patient Information")
model_choice = st.sidebar.selectbox("Choose ML Model", ["XGBoost", "Random Forest"])

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex_encoded = 1 if sex == "Male" else 0

alb = st.sidebar.number_input("ALB (g/L)", min_value=20.0, max_value=60.0, value=40.0)
alp = st.sidebar.number_input("ALP (U/L)", min_value=20.0, max_value=400.0, value=80.0)
alt = st.sidebar.number_input("ALT (U/L)", min_value=5.0, max_value=300.0, value=30.0)
ast = st.sidebar.number_input("AST (U/L)", min_value=10.0, max_value=300.0, value=25.0)
bil = st.sidebar.number_input("BIL (μmol/L)", min_value=2.0, max_value=50.0, value=10.0)
che = st.sidebar.number_input("CHE (kU/L)", min_value=2.0, max_value=20.0, value=8.0)
chol = st.sidebar.number_input("CHOL (mmol/L)", min_value=2.0, max_value=10.0, value=5.0)
crea = st.sidebar.number_input("CREA (μmol/L)", min_value=30.0, max_value=200.0, value=80.0)
ggt = st.sidebar.number_input("GGT (U/L)", min_value=5.0, max_value=300.0, value=25.0)
prot = st.sidebar.number_input("PROT (g/L)", min_value=50.0, max_value=100.0, value=70.0)

# Input for prediction
input_data = pd.DataFrame([[age, sex_encoded, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]],
                          columns=['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])

# Train both models
@st.cache_resource
def train_models():
    X = df[['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]
    y = df['Hepatitis_C']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return xgb, rf, X_test, y_test

xgb_model, rf_model, X_test, y_test = train_models()

# Prediction
if st.sidebar.button("Predict"):
    model = xgb_model if model_choice == "XGBoost" else rf_model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader(f"Prediction using {model_choice}")
    col1, col2 = st.columns(2)

    with col1:
        label = "Virus C Positive" if prediction[0] == 1 else "Blood Donor (Negative)"
        confidence = f"{prediction_proba[0][1]*100:.1f}%" if prediction[0] == 1 else f"{prediction_proba[0][0]*100:.1f}%"
        st.metric("Prediction", label, f"{confidence} confidence")

    with col2:
        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
        st.metric("Model Accuracy", f"{accuracy:.1f}%", "on test dataset")

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_ if model_choice == "XGBoost" else model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots()
    feature_importance.sort_values('Importance').plot.barh(x='Feature', y='Importance', ax=ax)
    st.pyplot(fig)

    # Classification report
    st.subheader("Model Performance Metrics")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

# Data exploration
st.header("Data Exploration")
st.write("Explore the dataset used for training the models.")

if st.checkbox("Show raw data"):
    st.dataframe(df)

if st.checkbox("Show statistics"):
    st.write(df.describe())
