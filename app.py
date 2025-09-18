import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

# Load the pre-trained CatBoost model
with open('catboost_model.pkl', 'rb') as file:
    catboost_model = pickle.load(file)

# Streamlit interface for the user
st.title("Used Car Sales Prediction with CatBoost")

st.markdown("""
    This app predicts the selling price of a used car based on its features using a pre-trained CatBoost model.
    Please input the car details below to get a price prediction.
""")

# Inputs
year = st.number_input('Year of Manufacture', min_value=2000, max_value=2025, value=2015)
km_driven = st.number_input('Kilometers Driven (in KMs)', min_value=0, max_value=500000, value=50000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['1st Owner', '2nd Owner', '3rd Owner', '4th_Owner'])

# Prepare the input data for prediction (Make sure the column names match the trained model's expected feature names)
input_data = pd.DataFrame({
    'km_driven': [km_driven],  # Ensure this matches your training data feature name
	'fuel_CNG': [1 if fuel == 'CNG' else 0],   
    'fuel_Diesel': [1 if fuel == 'Diesel' else 0],
    'fuel_Electric': [1 if fuel == 'Electric' else 0],
	'fuel_LPG': [1 if fuel == 'LPG' else 0],
	'fuel_Petrol': [1 if fuel == 'Petrol' else 0],
	'seller_type_Dealer': [1 if seller_type == 'Dealer' else 0],
	'seller_type_Trustmark Dealer	': [1 if seller_type == 'Trustmark Dealer' else 0],	
    'seller_type_Individual': [1 if seller_type == 'Individual' else 0],   	
    'transmission_Automatic': [1 if transmission == 'Automatic' else 0],
	'transmission_Manual': [1 if transmission == 'Manual' else 0],
	'owner_4th_Owner': [1 if owner == '4th_Owner' else 0],
    'owner_First Owner': [1 if owner == '1st Owner' else 0],
    'owner_Second Owner': [1 if owner == '2nd Owner' else 0],
    'owner_Third Owner': [1 if owner == '3rd Owner' else 0],
    
})

# Add car age to the input data (make sure to add this if it was used in the model training)
car_age = 2025 - year
input_data['car_age'] = car_age  # Ensure this matches the feature name in your trained model

# Prediction button
if st.button('Predict Price'):
    # Predict the price using the loaded CatBoost model
    price_cb = catboost_model.predict(input_data)
    
    st.write(f"Predicted Price using CatBoost: ₹{round(price_cb[0], 2)}")

# Run the Streamlit app using: streamlit run app.py

