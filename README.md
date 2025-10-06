# Used Car Sales Prediction

This project is a **Used Car Sales Prediction App** built with **Streamlit** and **CatBoost**. The app predicts the selling price of a used car based on features such as the year of manufacture, kilometers driven, fuel type, transmission, and more.

## Features
- **User Input:** The app allows users to input various features of the car such as year of manufacture, kilometers driven, fuel type, transmission type, and owner type.
- **Price Prediction:** The app uses a pre-trained **CatBoost Regressor** model to predict the selling price of the car.
- **Interactive Interface:** The app provides an easy-to-use interface for making predictions using Streamlit.

## Installation

### Prerequisites

Make sure you have **Python 3.6** or later installed. You also need to install the following Python libraries:

- `streamlit` – for creating the interactive web app
- `pandas` – for data manipulation
- `numpy` – for numerical operations
- `catboost` – for the machine learning model
- `scikit-learn` – for preprocessing and model evaluation

You can install the required libraries using pip:

```
pip install streamlit pandas numpy scikit-learn catboost

```

### Clone the repository

Clone the repository to your local machine using the following command:
```
git clone https://github.com/Sumit-1988-git/Mini_project_Used_car_sales_Prediction.git
cd Mini_project_Used_car_sales_Prediction
```

### CatBoost Model

Make sure to place the catboost_model.pkl file (the pre-trained model) in the root directory of the project.

### Running the App

To run the Streamlit app, use the following command in your terminal:

```
streamlit run app.py
```
### How to Use

Enter Car Details: On the app’s interface, input the car details such as year of manufacture, kilometers driven, fuel type, transmission type, and owner type.

Predict Price: Click the "Predict Price" button to get the predicted selling price of the car.

Results: The app will display the predicted price based on the CatBoost model.

### Project Structure

* app.py: Main Streamlit application that runs the app.

* catboost_model.pkl: Pre-trained CatBoost model for predicting car prices.

* README.md: Documentation for the project (this file).

* requirements.txt: List of all Python packages required for this project.
  
* Demo.webm : Contains the demo video of the application
