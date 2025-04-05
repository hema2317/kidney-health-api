from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("kidney_model_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")

@app.route("/", methods=["GET"])
def home():
    return "✅ Kidney Health Predictor is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected input keys
        required_fields = ["age", "creatinine", "albumin", "egfr", "hba1c"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing input fields"}), 400

        # Create a DataFrame from input
        df = pd.DataFrame([data])

        # Predict using the model
        prediction = model.predict(df)
        label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

