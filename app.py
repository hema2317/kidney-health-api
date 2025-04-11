from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained XGBoost model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route("/")
def home():
    return "Hello from Kidney Health Predictor!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data, flush=True)

        age = float(data.get("age", 0))
        hba1c = float(data.get("hba1c", 0))
        albumin = float(data.get("albumin", 0))
        scr = float(data.get("scr", 0))
        egfr = float(data.get("egfr", 0))

        # Prepare input
        features = pd.DataFrame([[age, hba1c, albumin, scr, egfr]], 
            columns=["age", "hba1c", "albumin", "scr", "egfr"])
        dmatrix = xgb.DMatrix(features)

        # Predict
        prediction = model.predict(dmatrix)
        predicted_class = int(np.rint(prediction[0]))

        # Override risk based on key medical logic
        if hba1c >= 9 and egfr < 60:
            predicted_class = 2  # High risk
        elif hba1c >= 8:
            predicted_class = max(predicted_class, 1)  # At least moderate

        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        # Suggestions
        patient_plan = ""
        doctor_plan = ""

        if risk == "Low":
            patient_plan = "Maintain a balanced diet, stay hydrated, and monitor blood sugar regularly."
            doctor_plan = "Continue routine checks annually. Reinforce preventive care."
        elif risk == "Moderate":
            patient_plan = "Watch your sugar intake and consult a nutritionist if needed. Monitor kidney labs every 3–6 months."
            doctor_plan = "Repeat eGFR and HbA1c in 3 months. Consider early nephrology input."
        elif risk == "High":
            patient_plan = "Strict sugar control and renal-friendly diet required. Avoid alcohol and NSAIDs."
            doctor_plan = "Urgent nephrology referral. Evaluate for diabetic nephropathy or rapid decline."

        return jsonify({
            "risk": risk,
            "explanation": f"Your risk is based on eGFR ({egfr}) and HbA1c ({hba1c}%) along with Albumin ({albumin}) and Creatinine ({scr}).",
            "patient_plan": patient_plan,
            "doctor_plan": doctor_plan
        })

    except Exception as e:
        print("Prediction error:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
