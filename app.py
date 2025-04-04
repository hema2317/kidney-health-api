from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# ✅ Load your trained model and label encoder
model = joblib.load("kidney_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ Extract input features
        age = float(data["age"])
        creatinine = float(data["creatinine"])
        albumin = float(data["albumin"])
        egfr = float(data["egfr"])

        # ✅ Create input DataFrame in correct order
        input_df = pd.DataFrame([[age, creatinine, albumin, egfr]],
                                columns=["age", "creatinine", "albumin", "egfr"])

        # ✅ Make prediction
        prediction = model.predict(input_df)[0]

        # ✅ Decode label (e.g., 0 → Low)
        readable_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({"risk": readable_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
