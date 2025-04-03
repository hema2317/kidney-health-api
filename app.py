from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import requests

app = Flask(__name__)
CORS(app)

# ✅ Load the trained model (string label version)
model = pickle.load(open("kidney_model.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Kidney Health API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Prepare input features
        features = [
            float(data["hba1c"]),
            float(data["albumin"]),
            float(data["creatinine"]),
            int(data["age"]),
            1 if data["sex"].lower() == "male" else 0
        ]

        columns = ["hba1c", "albumin", "creatinine", "age", "sex"]
        input_df = pd.DataFrame([features], columns=columns)

        # ✅ Predict
        prediction = model.predict(input_df)[0]

        # ✅ No number mapping needed – model returns strings like "High"
        return jsonify({"risk": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/cgm-callback")
def cgm_callback():
    code = request.args.get("code")
    if not code:
        return "No authorization code received.", 400

    # Dexcom credentials
    token_url = "https://sandbox-api.dexcom.com/v2/oauth2/token"
    client_id = "EjJmOsxReUCm2GojkJ37SoF3E0WnLu5"
    client_secret = "9LqRvUkZR4bK7Ijh"
    redirect_uri = "https://kidney-health-ui.vercel.app/cgm-callback"

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri
    }

    token_response = requests.post(token_url, data=payload)
    token_data = token_response.json()

    access_token = token_data.get("access_token")
    if not access_token:
        return "Failed to get access token", 400

    glucose_url = "https://sandbox-api.dexcom.com/v2/users/self/egvs?startDate=2024-01-01T00:00:00&endDate=2024-01-02T00:00:00"
    headers = {"Authorization": f"Bearer {access_token}"}
    glucose_response = requests.get(glucose_url, headers=headers)
    glucose_data = glucose_response.json()

    egvs = glucose_data.get("egvs", [])
    if not egvs:
        return "No CGM data found", 400

    values = [egv["value"] for egv in egvs if "value" in egv]
    if not values:
        return "No glucose values to calculate", 400

    avg_glucose = sum(values) / len(values)
    estimated_hba1c = round((avg_glucose + 46.7) / 28.7, 2)

    return jsonify({
        "estimated_hba1c": estimated_hba1c,
        "average_glucose": round(avg_glucose, 1),
        "source": "Dexcom CGM"
    })

# ✅ Render support: Bind to port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
