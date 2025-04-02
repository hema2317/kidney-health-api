from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = pickle.load(open("kidney_model_v2.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Kidney Health API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Prepare input features
    features = [
        float(data["hba1c"]),
        float(data["albumin"]),
        float(data["creatinine"]),
        float(data["egfr"]),
        int(data["age"]),
        1 if data["sex"].lower() == "male" else 0
    ]

    columns = ["hba1c", "albumin", "urine_creatinine", "egfr", "age", "sex"]
    input_df = pd.DataFrame([features], columns=columns)

    prediction = model.predict(input_df)[0]

    # Map result to label
    risk_levels = {0: "Low", 1: "Moderate", 2: "High"}
    return jsonify({"risk": risk_levels.get(prediction, "Unknown")})

# ✅ Fix for Render – bind to PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
