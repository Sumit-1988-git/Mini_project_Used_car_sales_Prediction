import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor

# Load the trained model
with open('catboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app header
st.title("Used Car Price Prediction")
st.markdown("This app predicts the price of a used car based on user inputs.")

# User input fields
year = st.number_input("Car Year", min_value=1990, max_value=2025, step=1, value=2020)
km_driven = st.number_input("Kilometers Driven", min_value=1000, max_value=300000, step=1000, value=50000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
owner = st.selectbox("Previous Owners", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner]
})

# Map categorical variables if needed (e.g., replacing 'Fourth & Above Owner' with '4th_Owner')
input_data['owner'] = input_data['owner'].replace({'Fourth & Above Owner': '4th_Owner'})

# Convert categorical columns to the proper type that CatBoost can handle
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
for col in categorical_cols:
    input_data[col] = input_data[col].astype('category')

# Prediction
if st.button('Predict Price'):
    # Make prediction using the CatBoost model
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Price: ₹{prediction[0]:,.2f}")

