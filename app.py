from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow frontend access

model = joblib.load("kidney_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_features = [[
        data["age"], data["hba1c"], data["albumin"],
        data["scr"], data["egfr"]
    ]]
    prediction = model.predict(input_features)[0]
    categories = ["High", "Moderate", "Low"]
    return jsonify({"risk": categories[prediction]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
