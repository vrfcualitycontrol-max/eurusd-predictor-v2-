from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# === ЗАГРУЗКА МОДЕЛИ V2 (5 лет, 44 признака) ===
with open('model_v2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler_v2.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app_data_v2.json', 'r') as f:
    app_data = json.load(f)

feature_names = app_data['feature_names']
scaler_params = app_data['scaler_params']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_from_features(features_dict):
    values = [float(features_dict.get(name, 0)) for name in feature_names]
    scaled = []
    for i, name in enumerate(feature_names):
        sp = scaler_params[name]
        scaled.append((values[i] - sp['mean']) / sp['std'])
    scaled = np.array(scaled).reshape(1, -1)
    prob_up = float(model.predict_proba(scaled)[0][1])
    prediction = 1 if prob_up >= 0.5 else 0
    return {
        'prediction': prediction,
        'prob_up': round(prob_up * 100, 2),
        'prob_down': round((1 - prob_up) * 100, 2)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    features = data.get('features', {})
    return jsonify(predict_from_features(features))

@app.route('/api/data')
def api_data():
    return jsonify(app_data)

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.get_json()
    base = {name: app_data['indicators'].get(name, 0) for name in feature_names}
    for key in data:
        if key in base:
            base[key] = float(data[key])
    result = predict_from_features(base)
    result['features_used'] = base
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
