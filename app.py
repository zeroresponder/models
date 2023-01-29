from flask import Flask, request
from tensorflow import keras
from model2 import predict, predict_lite

model = keras.models.load_model("model.h5")
model_lite = keras.models.load_model("model_lite.h5")
app = Flask(__name__)


@app.route('/predict')
def predict():
    data = request.args
    print(data)
    return {"result": str(predict(model, data)[0][0])}

@app.route('/predict_lite')
def predict_lite():
    data = request.args
    return {"result": str(predict_lite(model, data)[0][0])}

if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
