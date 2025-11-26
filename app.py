import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load model, scaler, and columns
model = joblib.load("temu_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")  # Columns used in training

#Streamlit UI 
st.title("Temu Purchase Likelihood Prediction App")
st.write("Fill in the customer details to predict purchase likelihood.")

#User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
residence = st.selectbox("Residence sector", ["Urban", "Rural", "Peri Urban"])
qualification = st.selectbox(
    "Qualification", 
    ["Secondary", "Diploma/NCE", "Bachelor’s degree/HND", "Master’s degree", "PhD"]
)
monthly_income = st.selectbox(
    "Monthly income",
    ["> ₦399,000", "₦100,000–199,999", "₦200,000–399,999", "₦50,000–99,999"]
)
recent_platform = st.selectbox(
    "Recent online purchase platform",
    ["Jumia", "Konga", "Jiji", "Physical store (with online payment)", "Social media seller", "Temu"]
)
temu_awareness = st.selectbox("Temu Awareness", ["Yes", "No"])
temu_purchased = st.selectbox("Temu purchased", ["No — never used", "Yes"])
qty_purchase = st.selectbox("Qty of purchase", ["1", "2–4", "5–9", "10"])
avg_spending = st.selectbox(
    "Average Spending Amount",
    ["<₦2,000", "₦2,000–4,999", "₦5,000–9,999", "₦10,000–19,999", ">₦20,000"]
)

#Binary numeric features (Yes=1 / No=0)
binary_features = {
    'Attractive promos/freebies': st.radio("Attractive promos/freebies", ["Yes", "No"]),
    'Better deals than Jumia/Konga': st.radio("Better deals than Jumia/Konga", ["Yes", "No"]),
    'Curiosity': st.radio("Curiosity", ["Yes", "No"]),
    'Lower prices': st.radio("Lower prices", ["Yes", "No"]),
    'Peer recommendation': st.radio("Peer recommendation", ["Yes", "No"]),
    'Social media ads': st.radio("Social media ads", ["Yes", "No"]),
    'Wide product range': st.radio("Wide product range", ["Yes", "No"]),
    'Products match Quality': st.radio("Products match Quality", ["Yes", "No"]),
    'Reasonable delivery time': st.radio("Reasonable delivery time", ["Yes", "No"]),
    'Better prices': st.radio("Better prices", ["Yes", "No"]),
    'Easy refund': st.radio("Easy refund", ["Yes", "No"]),
    'Accurate product description': st.radio("Accurate product description", ["Yes", "No"]),
    'Home broadband': st.radio("Home broadband", ["Yes", "No"]),
    'Mobile data': st.radio("Mobile data", ["Yes", "No"]),
    'Public Wi-Fi': st.radio("Public Wi-Fi", ["Yes", "No"]),
    'Work/school Wi-Fi': st.radio("Work/school Wi-Fi", ["Yes", "No"]),
    'Difficulty with returns': st.radio("Difficulty with returns", ["Yes", "No"]),
    'Fake or replica items': st.radio("Fake or replica items", ["Yes", "No"]),
    'Hidden charges/customs': st.radio("Hidden charges/customs", ["Yes", "No"]),
    'Language/communication barriers': st.radio("Language/communication barriers", ["Yes", "No"]),
    'Long shipping times': st.radio("Long shipping times", ["Yes", "No"]),
    'Payment security': st.radio("Payment security", ["Yes", "No"]),
    'Product safety/quality': st.radio("Product safety/quality", ["Yes", "No"])
}

#Prediction button
if st.button("Predict Purchase Likelihood"):

    #Features DataFrame 
    input_data = {
        'Age': age,
        'Gender': gender,
        'Residence sector': residence,
        'Qualification': qualification,
        'Monthly income': monthly_income,
        'Recent online purchase platform': recent_platform,
        'Temu Awareness': temu_awareness,
        'Temu purchased': temu_purchased,
        'Qty of purchase': qty_purchase,
        'Average Spending Amount': avg_spending
    }

    #Binary Yes/No features
    for key, value in binary_features.items():
        input_data[key] = 1 if value == "Yes" else 0

    #Converting to DataFrame
    input_df = pd.DataFrame([input_data])

    #One-hot encode categorical variables in the model_columns
    input_df_encoded = pd.get_dummies(input_df)

    #input_df_encoded
    for col in model_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    #Column Reordering
    input_df_encoded = input_df_encoded[model_columns]

    #Numeric Scaling
    numeric_cols = scaler.feature_names_in_
    input_df_encoded[numeric_cols] = scaler.transform(input_df_encoded[numeric_cols])

    #Prediction
    prediction = model.predict(input_df_encoded)[0]
    probability = model.predict_proba(input_df_encoded)[0][1]

    #Results
    st.subheader("Prediction Result")
    st.write(f"Purchase Likelihood: **{'Yes' if prediction == 1 else 'No'}**")
    st.write(f"Probability of Purchase: **{probability:.2f}**")

