import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


with open('LinearRegressionModel.pkl', 'rb') as f:
    model = pickle.load(f)

transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year', 'kms_driven']),
        ('cat', OneHotEncoder(), ['company', 'fuel_type'])
    ])


pipe = Pipeline([
    ('preprocessor', transformer),
    ('model', model)
])

st.title("Car Price Prediction")


name = st.text_input("Car Name")
company = st.text_input("Company")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])


if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        'name': name,
        'company': company,
        'year': year,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type
    }])

    prediction = pipe.predict(input_data)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")