import sys
import os
sys.stdout = sys.stderr  # Ensures logs are captured
os.environ["PYTHONUNBUFFERED"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://kidney-health-ui.onrender.com"])  # <-- Add this line

# Load model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming data:", data)

        age = float(data['age'])
        hba1c = float(data['hba1c'])
        albumin = float(data['albumin'])
        scr = float(data['scr'])
        egfr = float(data['egfr'])

        features = np.array([[age, hba1c, albumin, scr, egfr]])
        dmatrix = xgb.DMatrix(features)
        preds = model.predict(dmatrix)
        print("Raw prediction:", preds)

        if len(preds.shape) == 1:
            predicted_class = int(np.round(preds[0]))  # for binary classifier
        else:
            predicted_class = int(np.argmax(preds))    # for multiclass

        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        return jsonify({'risk': risk})
    except Exception as e:
        print("Prediction error:", str(e))  # Print error to logs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
