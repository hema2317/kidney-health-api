import os
import sys
import logging
import urllib.parse
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

# Setup logging and flush
sys.stdout = sys.stderr
os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Initialize app
app = Flask(__name__)

# Allow frontend origin
CORS(app, origins=["https://kidney-health-ui.onrender.com"])

# Load model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data, flush=True)

        # Extract and convert inputs
        age = float(data.get('age', 0))
        hba1c = float(data.get('hba1c', 0))
        albumin = float(data.get('albumin', 0))
        scr = float(data.get('scr', 0))
        egfr = float(data.get('egfr', 0))
        print("Parsed inputs:", age, hba1c, albumin, scr, egfr, flush=True)

        # Ensure feature names match training
        features = pd.DataFrame([[age, hba1c, albumin, scr, egfr]],
                                columns=["age", "hba1c", "albumin", "scr", "egfr"])
        dmatrix = xgb.DMatrix(features)

        # Predict
        preds = model.predict(dmatrix)
        print("Raw prediction:", preds, flush=True)

        predicted_class = int(np.rint(preds[0]))
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")
        print("Final risk:", risk, flush=True)

        return jsonify({'risk': risk})

    except Exception as e:
        print("Prediction error:", str(e), flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/connect-cgm')
def connect_cgm():
    dexcom_client_id = os.getenv("DEXCOM_CLIENT_ID")
    dexcom_redirect_uri = os.getenv("DEXCOM_REDIRECT_URI")
    dexcom_auth_url = "https://sandbox-api.dexcom.com/v2/oauth2/login"

    params = {
        "client_id": dexcom_client_id,
        "redirect_uri": dexcom_redirect_uri,
        "response_type": "code",
        "scope": "offline_access"
    }

    redirect_url = f"{dexcom_auth_url}?{urllib.parse.urlencode(params)}"
    return redirect(redirect_url)

@app.route('/cgm-callback')
def cgm_callback():
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Authorization code not found in callback"}), 400

    try:
        # Exchange code for access token
        token_url = "https://sandbox-api.dexcom.com/v2/oauth2/token"
        payload = {
            "client_id": os.getenv("DEXCOM_CLIENT_ID"),
            "client_secret": os.getenv("DEXCOM_CLIENT_SECRET"),
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": os.getenv("DEXCOM_REDIRECT_URI")
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        token_response = requests.post(token_url, data=payload, headers=headers)
        token_data = token_response.json()

        access_token = token_data.get("access_token")
        if not access_token:
            return jsonify({"error": "Failed to obtain access token", "details": token_data}), 500

        # Now fetch glucose values
        glucose_url = "https://sandbox-api.dexcom.com/v2/users/self/egvs"
        params = {
            "startDate": "2024-04-01T00:00:00",
            "endDate": "2024-04-07T23:59:59"
        }
        glucose_response = requests.get(glucose_url, headers={"Authorization": f"Bearer {access_token}"}, params=params)
        glucose_data = glucose_response.json()

        glucose_values = glucose_data.get("egvs", [])
        glucose_readings = [point["value"] for point in glucose_values if "value" in point]

        if not glucose_readings:
            return jsonify({"error": "No glucose readings found"}), 500

        avg_glucose = sum(glucose_readings) / len(glucose_readings)
        estimated_hba1c = (avg_glucose + 46.7) / 28.7
        print(f"Computed HbA1c: {estimated_hba1c}", flush=True)

        return jsonify({
            "average_glucose": avg_glucose,
            "estimated_hba1c": round(estimated_hba1c, 2)
        })

    except Exception as e:
        print("CGM callback error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
