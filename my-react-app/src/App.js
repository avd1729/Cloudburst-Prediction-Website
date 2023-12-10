// App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [city, setCity] = useState('');
  const [prediction, setPrediction] = useState('');
  const [model, setModel] = useState(null);

  useEffect(() => {
    // Load the TensorFlow model during component initialization
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadGraphModel('C:/Users/Aravind/PROJECTS/Cloudburst-Prediction-Website/api_model');
        setModel(loadedModel);
      } catch (error) {
        console.error('Error loading TensorFlow model:', error);
      }
    };

    loadModel();
  }, []);

  const getWeatherData = async () => {
    try {
      const response = await axios.get(
        `http://api.openweathermap.org/data/2.5/weather?q=${city}&appid=87ce6387cccc97c824ed350534e68fb2`
      );

      const data = response.data;

      const featureNames = [
        'coord.lat',
        'coord.lon',
        'main.temp',
        'main.feels_like',
        'main.pressure',
        'main.humidity',
        'wind.speed',
        'wind.deg',
      ];

      const extractedFeatures = featureNames.map((name) => getNestedValue(data, name));

      if (model) {
        // Make the prediction
        const pred = model.execute(tf.tensor([extractedFeatures]));
        const scaledPred = pred.div(tf.scalar(3));
        
        setPrediction(`Chances of a CloudBurst in ${city} is ${scaledPred.arraySync()[0][0].toFixed(3)} %`);
      }
    } catch (error) {
      setPrediction('Please enter a valid city name!');
    }
  };

  const getNestedValue = (obj, key) => {
    const keys = key.split('.');
    let currentObj = obj;

    for (const k of keys) {
      if (currentObj && k in currentObj) {
        currentObj = currentObj[k];
      } else {
        return null;
      }
    }

    return currentObj;
  };

  return (
    <div>
      <h1>Baadal Jaagruk ⛈️</h1>
      <p>
        A cloud burst is a sudden, intense rainfall that happens over a short period, causing
        rapid flooding and other related disasters.
      </p>
      <p>
        Cloud burst prediction is the scientific process of forecasting sudden, intense rainfall
        events using weather data and computer models to help communities prepare for and respond to
        potential flooding and related disasters.
      </p>
      <p></p>
      <label>
        Enter city name:
        <input type="text" value={city} onChange={(e) => setCity(e.target.value)} />
      </label>
      <button onClick={getWeatherData}>Predict</button>
      <p>{prediction}</p>
    </div>
  );
}

export default App;


