from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

# ✅ Allow only your frontend domain (very important)
CORS(app, resources={r"/predict": {"origins": "https://kidney-health-ui.onrender.com"}})

# Load the XGBoost model from JSON
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = float(data['age'])
        hba1c = float(data['hba1c'])
        albumin = float(data['albumin'])
        scr = float(data['scr'])
        egfr = float(data['egfr'])

        # Prepare input
        features = np.array([[age, hba1c, albumin, scr, egfr]])
        dmatrix = xgb.DMatrix(features)
        preds = model.predict(dmatrix)
        predicted_class = int(np.argmax(preds))

        # Risk label mapping
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        return jsonify({'risk': risk})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
