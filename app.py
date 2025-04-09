from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("kidney_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["age"],
        data["hba1c"],
        data["albumin"],
        data["scr"],
        data["egfr"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    risk_map = {0: "High", 1: "Moderate", 2: "Low"}
    return jsonify({"risk": risk_map[int(prediction)]})

if __name__ == "__main__":
    app.run(debug=True)
