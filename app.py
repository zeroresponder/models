from flask import Flask, request
from tensorflow import keras
from model2 import predict as predict_nn

model = keras.models.load_model("model.h5")

app = Flask(__name__)


@app.route('/predict')
def predict():
    args = request.args
    data = args.to_dict(flat=True)
    for (key, value) in data.items():
        data[key] = float(value)
    return {"result": str(predict_nn(model, data)[0][0])}

if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
