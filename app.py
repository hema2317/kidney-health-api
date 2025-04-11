from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load XGBoost model
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

        features = np.array([[age, hba1c, albumin, scr, egfr]])
        dmatrix = xgb.DMatrix(features, feature_names=["age", "hba1c", "albumin", "scr", "egfr"])
        prediction = model.predict(dmatrix)
        predicted_class = int(np.rint(prediction[0]))

        if hba1c >= 9 and egfr < 60:
            predicted_class = 2
        elif hba1c >= 8:
            predicted_class = max(predicted_class, 1)

        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        patient_plan = doctor_plan = ""
        if risk == "Low":
            patient_plan = "Maintain a balanced diet, stay hydrated, and monitor blood sugar regularly."
            doctor_plan = "Continue routine checks annually. Reinforce preventive care."
        elif risk == "Moderate":
            patient_plan = "Watch your sugar intake and consult a nutritionist. Monitor kidney labs every 3–6 months."
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

# ✅ Simulated CGM route
@app.route("/get-hba1c", methods=["GET"])
def get_hba1c():
    glucose_values = [120, 140, 160, 150, 135, 155, 145]  # Simulated readings
    if not glucose_values:
        return jsonify({"error": "No glucose data available"}), 404

    avg_glucose = sum(glucose_values) / len(glucose_values)
    estimated_hba1c = round((avg_glucose + 46.7) / 28.7, 2)

    return jsonify({
        "estimated_hba1c": estimated_hba1c,
        "glucose_points_used": len(glucose_values),
        "average_glucose": round(avg_glucose, 2)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
