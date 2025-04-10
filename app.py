from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Allow CORS for all origins (or restrict it to your frontend domain)
CORS(app, origins=["https://kidney-health-ui.onrender.com"])

# Load the XGBoost model
model = xgb.Booster()
model.load_model("kidney_model_xgb.json")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        age = float(data['age'])
        hba1c = float(data['hba1c'])
        albumin = float(data['albumin'])
        scr = float(data['scr'])
        egfr = float(data['egfr'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    # Prepare the input features for prediction
    features = np.array([[age, hba1c, albumin, scr, egfr]])
    dmatrix = xgb.DMatrix(features)

    # Predict
    preds = model.predict(dmatrix)
    predicted_class = int(np.argmax(preds))

    # Map prediction class to labels
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    risk = label_map.get(predicted_class, "Unknown")

    return jsonify({'risk': risk})

if __name__ == '__main__':
    app.run(debug=True)
