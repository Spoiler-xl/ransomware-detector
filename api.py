from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained ransomware detection model
model = joblib.load('rf_model.pkl')

# Home route
@app.route('/')
def home():
    return "🚀 Ransomware Detection API is up and running!"

# Predict route
@app.route('/detect', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data['features'] 
        prediction = model.predict([np.array(features)])
        result = 'Benign' if prediction[0] == 1 else 'Ransomware'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Start the server
app.run()
import requests
