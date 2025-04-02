from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd  # ✅ Required for DataFrame input

app = Flask(__name__)
CORS(app)

# Load the retrained model
model = pickle.load(open("kidney_model_v2.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Kidney Health API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract and prepare features
    features = [
        float(data["hba1c"]),
        float(data["albumin"]),
        float(data["creatinine"]),
        float(data["egfr"]),
        int(data["age"]),
        1 if data["sex"].lower() == "male" else 0
    ]

    # 🧠 Fix warning by matching training column names
    columns = ["hba1c", "albumin", "urine_creatinine", "egfr", "age", "sex"]
    input_df = pd.DataFrame([features], columns=columns)

    # Predict risk
    prediction = model.predict(input_df)[0]

    # Map prediction to labels
    risk_levels = {0: "Low", 1: "Moderate", 2: "High"}
    return jsonify({"risk": risk_levels.get(prediction, "Unknown")})

if __name__ == "__main__":
    app.run(debug=True)
