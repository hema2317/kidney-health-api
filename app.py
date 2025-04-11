from flask import Flask, request, jsonify, redirect
import urllib.parse
import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from flask_cors import CORS
from datetime import datetime, timedelta

NIGHTSCOUT_URL = os.getenv("NIGHTSCOUT_URL", "https://kidney-cgm-demo-32a6e80f3c55.herokuapp.com")
NIGHTSCOUT_SECRET = os.getenv("NIGHTSCOUT_SECRET", "nightscout123")  # Set this in Heroku

def fetch_glucose_data_from_nightscout():
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)

        params = {
            "find[dateString][$gte]": start_time.isoformat(),
            "find[dateString][$lte]": end_time.isoformat()
        }

        headers = {
            "api-secret": NIGHTSCOUT_SECRET
        }

        url = f"{NIGHTSCOUT_URL}/api/v1/entries.json"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        glucose_vals = [entry["sgv"] for entry in data if "sgv" in entry]
        return glucose_vals
    except Exception as e:
        print("Error fetching Nightscout data:", str(e))
        return []

def estimate_hba1c_from_glucose(glucose_vals):
    if not glucose_vals:
        return None
    avg_glucose = sum(glucose_vals) / len(glucose_vals)
    return round((avg_glucose + 46.7) / 28.7, 2)

# Setup logging
sys.stdout = sys.stderr
os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://kidney-health-ui.onrender.com"}}, supports_credentials=True)

# Load the XGBoost model
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

        # Override logic if HbA1c is dangerously high (>=9.0)
        if hba1c >= 9.0:
            if predicted_class == 0:
                predicted_class = 1  # elevate from Low to Moderate
            elif predicted_class == 1 and egfr < 60:
                predicted_class = 2  # elevate to High if egfr also low

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

@app.route('/connect-cgm')
def connect_cgm():
    dexcom_client_id = os.getenv("DEXCOM_CLIENT_ID")
    dexcom_redirect_uri = os.getenv("DEXCOM_REDIRECT_URI")

    print("=== Dexcom Debug ===")
    print("Client ID:", dexcom_client_id)
    print("Redirect URI:", dexcom_redirect_uri)
    print("====================")

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
        return jsonify({"error": "Authorization code not found"}), 400

    try:
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
            return jsonify({"error": "Access token fetch failed", "details": token_data}), 500

        glucose_url = "https://sandbox-api.dexcom.com/v2/users/self/egvs"
        params = {
            "startDate": "2024-04-01T00:00:00",
            "endDate": "2024-04-07T23:59:59"
        }
        glucose_response = requests.get(glucose_url, headers={"Authorization": f"Bearer {access_token}"}, params=params)
        glucose_data = glucose_response.json()
        values = glucose_data.get("egvs", [])
        if not values:
            return jsonify({"error": "No glucose data found"}), 404

        glucose_vals = [v["value"] for v in values if "value" in v]
        if not glucose_vals:
            return jsonify({"error": "No glucose values found in data"}), 404

        avg_glucose = sum(glucose_vals) / len(glucose_vals)
        estimated_hba1c = round((avg_glucose + 46.7) / 28.7, 2)

        return jsonify({
            "estimated_hba1c": estimated_hba1c,
            "glucose_points_used": len(glucose_vals),
            "average_glucose": round(avg_glucose, 2)
        })
    except Exception as e:
        print("CGM callback error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500

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
