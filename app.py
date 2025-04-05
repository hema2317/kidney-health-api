from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("kidney_model_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

# Health check route
@app.route("/", methods=["GET"])
def home():
    return "✅ Kidney Health Predictor is live!"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected keys
        required_keys = ["age", "creatinine", "albumin", "egfr", "hba1c"]

        # Check for missing keys
        if not all(key in data for key in required_keys):
            return jsonify({"error": f"Missing one or more required fields: {required_keys}"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([{
            "age": data["age"],
            "creatinine": data["creatinine"],
            "albumin": data["albumin"],
            "egfr": data["egfr"],
            "hba1c": data["hba1c"]
        }])

        # Predict
        prediction = model.predict(input_df)
        result = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app on Render (bind to 0.0.0.0 and port 10000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
