from flask import Flask, request, jsonify
import mlflow.pyfunc
import numpy as np

model = mlflow.pyfunc.load_model("models:/BestClassifierModel/Production")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)