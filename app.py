import sys
import os
sys.stdout = sys.stderr  # Ensures logs are captured
os.environ["PYTHONUNBUFFERED"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

app = Flask(__name__)
CORS(app, origins=["https://kidney-health-ui.onrender.com"])  # <-- Add this line

# Load model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data, flush=True)

        age = float(data.get('age', 0))
        hba1c = float(data.get('hba1c', 0))
        albumin = float(data.get('albumin', 0))
        scr = float(data.get('scr', 0))
        egfr = float(data.get('egfr', 0))

        print("Parsed inputs:", age, hba1c, albumin, scr, egfr, flush=True)

        features = np.array([[age, hba1c, albumin, scr, egfr]])
        dmatrix = xgb.DMatrix(features)

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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
