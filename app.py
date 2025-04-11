from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

# Simulated glucose values
def fetch_simulated_glucose():
    return [160, 145, 170, 155, 165]

def estimate_hba1c(glucose_values):
    if not glucose_values:
        return None
    avg_glucose = sum(glucose_values) / len(glucose_values)
    return round((avg_glucose + 46.7) / 28.7, 2)

@app.route("/")
def home():
    return "Hello from Kidney Health Predictor!"

@app.route("/connect-cgm")
def connect_cgm():
    # Simulate CGM by auto-redirecting to estimated HbA1c page
    return redirect("/get-hba1c")

@app.route("/get-hba1c")
def get_hba1c():
    glucose_vals = fetch_simulated_glucose()
    if not glucose_vals:
        return jsonify({"error": "No simulated CGM data found"}), 404
    estimated = estimate_hba1c(glucose_vals)
    return jsonify({
        "estimated_hba1c": estimated,
        "glucose_points_used": len(glucose_vals),
        "average_glucose": round(sum(glucose_vals) / len(glucose_vals), 2)
    })

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

        patient_plan = ""
        doctor_plan = ""

        if risk == "Low":
            patient_plan = "Maintain a balanced diet, stay hydrated, and monitor blood sugar regularly."
            doctor_plan = "Continue routine checks annually. Reinforce preventive care."
        elif risk == "Moderate":
            patient_plan = "Watch your sugar intake. Monitor kidney labs every 3–6 months."
            doctor_plan = "Repeat eGFR and HbA1c in 3 months. Consider nephrology input."
        elif risk == "High":
            patient_plan = "Strict sugar control and renal-friendly diet required. Avoid alcohol and NSAIDs."
            doctor_plan = "Urgent nephrology referral. Evaluate for diabetic nephropathy."

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
