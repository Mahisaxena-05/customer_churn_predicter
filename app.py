import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("Customer Churn Predictor")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")
contract = st.selectbox("Contract", [0,1,2])

if st.button("Predict"):
    pred = model.predict([[tenure, monthly, total, contract]])
    st.success("CHURN" if pred[0]==1 else "STAY")
