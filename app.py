from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("kidney_model_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON
        data = request.json

        # Expected keys: age, creatinine, albumin, egfr, hba1c
        required_keys = ["age", "creatinine", "albumin", "egfr", "hba1c"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400

        # Create DataFrame for prediction
        features = pd.DataFrame([{
            "age": data["age"],
            "creatinine": data["creatinine"],
            "albumin": data["albumin"],
            "egfr": data["egfr"],
            "hba1c": data["hba1c"]
        }])

        # Make prediction
        prediction = model.predict(features)
        risk_label = label_encoder.inverse_transform(prediction)[0]

        # Return result
        return jsonify({"predicted_risk": risk_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route (optional)
@app.route('/')
def home():
    return "🧠 Kidney Health Predictor API is running!"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
