# app.py
import streamlit as st
import requests

# Set the title of the app
st.title('Iris Flower Prediction App')

# Create input fields for the user to enter flower features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# Define the FastAPI backend URL
backend_url = 'http://127.0.0.1:8000/predict'

# Create a button to make a prediction
if st.button('Predict'):
    # Prepare the request payload
    payload = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    # Make the POST request to the FastAPI backend
    response = requests.post(backend_url, json=payload)
    
    # Get the prediction result
    result = response.json()
    
    # Display the prediction result
    st.write(f'Prediction: {result["prediction"]}')