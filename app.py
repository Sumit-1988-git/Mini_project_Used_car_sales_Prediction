import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from catboost import CatBoostRegressor

# Load the pre-trained CatBoost model
with open('catboost_model.pkl', 'rb') as f:
    catboost_model = pickle.load(f)

# Define function for data preprocessing
def preprocess_data(df):
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Replace 'Fourth & Above Owner' with '4th_Owner'
    df['owner'] = df['owner'].replace({'Fourth & Above Owner': '4th_Owner'})
    
    # Categorical features to one-hot encoding
    categorical_cols = ['fuel', 'seller_type', 'transmission','owner']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Remove outliers
    Q1 = df_encoded["selling_price"].quantile(0.25)
    Q3 = df_encoded["selling_price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_encoded = df_encoded[(df_encoded["selling_price"] >= lower_bound) & (df_encoded["selling_price"] <= upper_bound)]
    
    # Scale numeric columns
    numeric_cols = ['km_driven', 'car_age']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    return df_encoded

# Function to predict the price using CatBoost
def predict_price(model, input_data):
    return model.predict(input_data)

# Streamlit interface for the user
st.title("Used Car Sales Prediction")

st.markdown("""
    This app predicts the selling price of a used car based on its features such as year, km_driven, fuel type, transmission, and more.
    Please input the car details below to get a price prediction.
""")

# Inputs
year = st.number_input('Year of Manufacture', min_value=2000, max_value=2025, value=2015)
km_driven = st.number_input('Kilometers Driven (in KMs)', min_value=0, max_value=500000, value=50000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['1st Owner', '2nd Owner', '3rd Owner', '4th_Owner'])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel_Petrol': [1 if fuel == 'Petrol' else 0],
    'fuel_Diesel': [1 if fuel == 'Diesel' else 0],
    'fuel_CNG': [1 if fuel == 'CNG' else 0],
    'fuel_LPG': [1 if fuel == 'LPG' else 0],
    'fuel_Electric': [1 if fuel == 'Electric' else 0],
    'seller_type_Individual': [1 if seller_type == 'Individual' else 0],
    'seller_type_Dealer': [1 if seller_type == 'Dealer' else 0],
    'transmission_Manual': [1 if transmission == 'Manual' else 0],
    'transmission_Automatic': [1 if transmission == 'Automatic' else 0],
    'owner_1st Owner': [1 if owner == '1st Owner' else 0],
    'owner_2nd Owner': [1 if owner == '2nd Owner' else 0],
    'owner_3rd Owner': [1 if owner == '3rd Owner' else 0],
    'owner_4th_Owner': [1 if owner == '4th_Owner' else 0],
})

# Add car age
car_age = pd.Timestamp.now().year - year
input_data['car_age'] = car_age

# Prediction button
if st.button('Predict Price'):
    # Preprocess the input data for the model
    input_data_processed = preprocess_data(input_data)
    
    # Predict the price using CatBoost model
    price_catboost = predict_price(catboost_model, input_data_processed)
    
    st.write(f"Predicted Price using CatBoost: ₹{round(price_catboost[0], 2)}")
