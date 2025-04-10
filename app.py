from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load the model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = float(data['age'])             # RIDAGEYR
        hba1c = float(data['hba1c'])         # LBXGH
        albumin = float(data['albumin'])     # URXUMA
        scr = float(data['scr'])             # URXUCR
        egfr = float(data['egfr'])           # eGFR

        # Compute ACR = albumin / scr (mg/g) if scr > 0
        acr = albumin / scr if scr > 0 else 0.0

        # Arrange features to match model input
        features = np.array([[age, hba1c, albumin, scr, scr, egfr, acr]])
        dmatrix = xgb.DMatrix(features)

        preds = model.predict(dmatrix)
        predicted_class = int(np.argmax(preds))

        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk = label_map.get(predicted_class, "Unknown")

        return jsonify({'risk': risk})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
