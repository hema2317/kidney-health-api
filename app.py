import os
if not os.path.exists("kidney_model.pkl"):
    import train_model

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("kidney_model.pkl")

@app.route("/")
def home():
    return "✅ Kidney Health Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["age"],
        data["hba1c"],
        data["albumin"],
        data["scr"],
        data["egfr"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    risk_map = {0: "High", 1: "Moderate", 2: "Low"}
    return jsonify({"risk": risk_map[int(prediction)]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)

