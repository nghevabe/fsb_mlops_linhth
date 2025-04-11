from flask import Flask, request, jsonify, render_template_string
import mlflow.pyfunc
import numpy as np
import requests
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# Giao diện HTML với 2 form: dự đoán + chuyển stage
html_template = """
<!doctype html>
<html>
<head>
    <title>MLops Project</title>
</head>
<body>
    <h2>Dự đoán phân loại với mô hình MLflow</h2>
    <form method="post">
        <button type="submit" name="action" value="predict">Dự đoán với dữ liệu mẫu</button>
    </form>
    {% if prediction is not none %}
        <p><strong>Kết quả dự đoán:</strong> {{ prediction }}</p>
    {% endif %}

    <hr>

    <h2>Chuyển version sang Production</h2>
    <form method="post">
        <label>Nhập Version:</label>
        <input type="number" name="version" required>
        <button type="submit" name="action" value="transition">Chuyển sang Production</button>
    </form>
    {% if transition_msg %}
        <p><strong>{{ transition_msg }}</strong></p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    transition_msg = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "predict":
            input_data = {"features": [0.5] * 20}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
                prediction = response.json().get("prediction", "Không xác định")
            except Exception as e:
                prediction = f"Lỗi: {e}"

        elif action == "transition":
            version = request.form.get("version")
            model_name = "BestClassifierModel"
            try:
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=int(version),
                    stage="Production"
                )
                transition_msg = f"✅ Đã chuyển Version {version} sang Production."
            except Exception as e:
                transition_msg = f"❌ Lỗi khi chuyển version: {e}"

    return render_template_string(html_template, prediction=prediction, transition_msg=transition_msg)

# API /predict (được gọi bởi chính app)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        model = mlflow.pyfunc.load_model("models:/BestClassifierModel/Production")
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
