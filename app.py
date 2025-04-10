import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load XGBoost model
with open('kidney_model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get all 5 features from request
    age = float(data['age'])
    hba1c = float(data['hba1c'])
    albumin = float(data['albumin'])
    creatinine = float(data['creatinine'])
    egfr = float(data['egfr'])

    # Prepare for model
    features = np.array([[age, hba1c, albumin, creatinine, egfr]])
    
    prediction = model.predict(features)[0]
    return jsonify({'risk': prediction})

if __name__ == '__main__':
    app.run(debug=True)
