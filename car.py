import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Load and clean the dataset
car = pd.read_csv('quikr_car.csv')
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '', regex=False).astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '', regex=False)
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
car = car.reset_index(drop=True)
car = car[car['Price'] < 6000000]

# Add derived features
car['age'] = 2024 - car['year']
car['price_per_km'] = car['Price'] / car['kms_driven']

# Handle missing or infinite values
car.replace([np.inf, -np.inf], np.nan, inplace=True)
car.dropna(inplace=True)

# Model training
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type', 'age', 'price_per_km']]
y = car['Price']

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
initial_r2 = r2_score(y_test, y_pred)
initial_mae = mean_absolute_error(y_test, y_pred)
print(f'Initial R^2 score: {initial_r2}')
print(f'Initial MAE: {initial_mae}')

# Save the model
with open('LinearRegressionModel.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)

# Streamlit app
st.title('Car Price Prediction')

# Load the model once
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Display an image
st.image('image.png', caption='Car Image', use_column_width=True)

# Dropdown for car name and company
name_options = sorted(car['name'].unique())
company_options = sorted(car['company'].unique())

name = st.selectbox('Car Name', name_options)
company = st.selectbox('Company', company_options)
year = st.number_input('Year', 2000, 2024, 2019)
kms_driven = st.number_input('Kilometers Driven', 0, 1000000, 100)
fuel_type = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'))

# Additional Inputs
mileage = st.number_input('Mileage (km/l)', 0, 50, 15)
engine_size = st.number_input('Engine Size (cc)', 500, 5000, 1500)
transmission = st.selectbox('Transmission', ('Manual', 'Automatic'))

# Derived Inputs
age = 2024 - year
price_per_km = st.number_input('Price per KM', 0, 500, 5)

input_data = pd.DataFrame({
    'name': [name], 'company': [company], 'year': [year], 
    'kms_driven': [kms_driven], 'fuel_type': [fuel_type],
    'age': [age], 'price_per_km': [price_per_km]
})

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'The predicted price of the car is: ₹{prediction[0]:,.2f}')

# Visualizations
if st.checkbox('Show Data Visualization'):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x='company', y='Price', data=car, ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.swarmplot(x='year', y='Price', data=car, ax=ax, size=4)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(x='kms_driven', y='Price', data=car, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='fuel_type', y='Price', data=car, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(x='company', y='Price', data=car, hue='fuel_type', size='year', ax=ax)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    st.pyplot(fig)

# Performance Metrics
st.write(f'Model Performance:')
st.write(f'R² score: {initial_r2}')
st.write(f'Mean Absolute Error: {initial_mae}')
