import os
import sys
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging and flush
sys.stdout = sys.stderr
os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Initialize app
app = Flask(__name__)

# Allow frontend origin
CORS(app, origins=["https://kidney-health-ui.onrender.com"])

# Load model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data, flush=True)

        # Extract and convert inputs
        age = float(data.get('age', 0))
        hba1c = float(data.get('hba1c', 0))
        albumin = float(data.get('albumin', 0))
        scr = float(data.get('scr', 0))
        egfr = float(data.get('egfr', 0))
        print("Parsed inputs:", age, hba1c, albumin, scr, egfr, flush=True)

        # Ensure feature names match training
        features = pd.DataFrame([[age, hba1c, albumin, scr, egfr]],
                                columns=["age", "hba1c", "albumin", "scr", "egfr"])
        dmatrix = xgb.DMatrix(features)

        # Predict
        preds = model.predict(dmatrix)
        print("Raw prediction:", preds, flush=True)

        predicted_class = int(np.rint(preds[0]))
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")
        print("Final risk:", risk, flush=True)

        return jsonify({'risk': risk})

    except Exception as e:
        print("Prediction error:", str(e), flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
