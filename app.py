import streamlit as st
import requests
import tensorflow as tf
import datetime as dt
import os
from apikey import API_KEY

st.title('Cloud Burst Prediction ⛈️')


def get_data(CITY):
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
    url = BASE_URL + "appid=" + API_KEY + "&q=" + CITY

    response = requests.get(url).json()
    json_data = response

    return json_data  # Return the JSON data directly


def predict():
    CITY = st.text_input('Enter city name : ')

    if CITY:  # Check if CITY is not empty
        result = get_data(CITY)

        if result:
            # Extract relevant features from JSON data
            feature_names = ['coord.lat', 'coord.lon', 'main.temp', 'main.feels_like',
                             'main.pressure', 'main.humidity', 'wind.speed', 'wind.deg']

            extracted_features = [get_nested_value(
                result, name) for name in feature_names]

            # Load the model outside the function to avoid loading it with every prediction
            if 'model' not in st.session_state:
                st.session_state.model = tf.keras.models.load_model(
                    'api_model')

            model = st.session_state.model

            # Make the prediction
            pred = model.predict([extracted_features])
            st.success(f'Cloud Burst Prediction: {pred[0]}')


def get_nested_value(obj, key):
    keys = key.split('.')
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        elif isinstance(obj, list) and len(obj) > 0:
            obj = obj[int(k)]
        else:
            return None
    return obj


if __name__ == '__main__':
    predict()
