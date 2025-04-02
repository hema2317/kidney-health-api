from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle


app = Flask(__name__)
CORS(app)


# Load trained model
model = pickle.load(open("kidney_model_v2.pkl", "rb"))


@app.route("/")
def home():
    return "✅ Kidney Health AI is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        float(data["hba1c"]),
        float(data["albumin"]),
        float(data["creatinine"]),
        float(data["egfr"]),
        int(data["age"]),
        1 if data["sex"].lower() == "male" else 0
    ]
    prediction = model.predict([features])[0]
    risk_levels = {0: "Low", 1: "Moderate", 2: "High"}
    return jsonify({"risk": risk_levels.get(prediction, "Unknown")})
