from flask import Flask, request, jsonify
import mlflow.pyfunc
import numpy as np

# Load model đã đăng ký trong registry ở stage Production
model = mlflow.pyfunc.load_model("models:/BestClassifierModel/Production")

app = Flask(__name__)

# Route test nhanh: mở trình duyệt vào http://localhost:8000
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask app is running!"

# Route chính để dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
