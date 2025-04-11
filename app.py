from flask import Flask, request, jsonify, redirect, make_response
import urllib.parse
import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from flask_cors import CORS
from datetime import datetime

# Constants
NIGHTSCOUT_URL = os.getenv("NIGHTSCOUT_URL", "https://kidney-cgm-demo-32a6e80f3c55.herokuapp.com")

# Simulated CGM data for testing
def fetch_glucose_data_from_nightscout():
    return [160, 145, 170, 155, 165]

def estimate_hba1c_from_glucose(glucose_vals):
    if not glucose_vals:
        return None
    avg_glucose = sum(glucose_vals) / len(glucose_vals)
    return round((avg_glucose + 46.7) / 28.7, 2)

# Logging setup
sys.stdout = sys.stderr
os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "https://kidney-health-ui.onrender.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

@app.route('/')
def home():
    return "Hello from Kidney Health API!"

# Load XGBoost model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data, flush=True)

        age = float(data.get('age', 0))
        hba1c = float(data.get('hba1c', 0))
        albumin = float(data.get('albumin', 0))
        scr = float(data.get('scr', 0))
        egfr = float(data.get('egfr', 0))

        features = pd.DataFrame([[age, hba1c, albumin, scr, egfr]], 
                                columns=["age", "hba1c", "albumin", "scr", "egfr"])
        dmatrix = xgb.DMatrix(features)

        preds = model.predict(dmatrix)
        predicted_class = int(np.rint(preds[0]))

        # Adjust risk if HbA1c is very high
        if hba1c >= 9.0:
            if predicted_class == 0:
                predicted_class = 1
            elif predicted_class == 1 and egfr < 60:
                predicted_class = 2

        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        return jsonify({
            'risk': risk,
            'explanation': f"Your prediction is based on: eGFR ({egfr}), HbA1c ({hba1c}%), Creatinine ({scr} mg/dL), and Albumin ({albumin} g/dL).",
            'adjusted': hba1c >= 9.0
        })

    except Exception as e:
        print("Prediction error:", str(e), flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get-hba1c', methods=['GET'])
def get_hba1c():
    glucose_vals = fetch_glucose_data_from_nightscout()
    if not glucose_vals:
        return jsonify({"error": "No CGM glucose values found"}), 404

    estimated_hba1c = estimate_hba1c_from_glucose(glucose_vals)
    return jsonify({
        "estimated_hba1c": estimated_hba1c,
        "glucose_points_used": len(glucose_vals),
        "average_glucose": round(sum(glucose_vals) / len(glucose_vals), 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
