from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and encoder
model = joblib.load("kidney_model_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

@app.route("/")
def home():
    return "✅ Kidney Health API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("data...",data)

    # ✅ Only use the 4 features that the model expects
    required_keys = ["age", "creatinine", "albumin", "egfr","hba1c"]
    if not all(k in data for k in required_keys):
        return jsonify({"error": "Missing one or more required fields."}), 400

    features = np.array([[ 
        data["age"],
        data["creatinine"],
        data["albumin"],
        data["egfr"],
        data["hba1c"]
    ]])

    prediction = model.predict(features)
    risk = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"risk": risk})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # ✅ Required by Render
    app.run(host="0.0.0.0", port=port)
