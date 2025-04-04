from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ✅ Load trained model
model = pickle.load(open("kidney_model.pkl", "rb"))

@app.route("/")
def home():
    return "✅ Kidney Health API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # ✅ Build input matching model's expected features (no sex)
    features = [
        float(data["hba1c"]),
        float(data["albumin"]),
        float(data["creatinine"]),
        int(data["age"])
    ]
    input_df = pd.DataFrame([features], columns=["hba1c", "albumin", "creatinine", "age"])

    # ✅ Predict
    prediction = model.predict(input_df)[0]

    # ✅ Return readable labels
    return jsonify({"risk": str(prediction)})

# ✅ For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
