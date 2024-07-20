from flask import Flask, request, jsonify
from PIL import Image
from src.model_train_inference.inference import inference
import os
import tensorflow as tf


app = Flask(__name__)

global model

model_path = os.path.join("model/best_model.keras")
# model_path = "model/best_model.keras"
model = tf.keras.models.load_model(model_path)


@app.route("/predict_file", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # img_path = file
    # print(img_path)

    img = Image.open(file)

    prediction = inference(img, model=model, mode="file")

    # result = prediction[0].tolist()
    return jsonify({"predictions": prediction})


# @app.route('/predict_path', methods=['POST'])
# def predict():
#     if 'file' not in request.json:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.json['file']
#     if file == '':
#         return jsonify({'error': 'No file selected'}), 400

#     img_path = '/Users/jiazhenli/Desktop/solar panel/data' + file
#     # file.save(img_path)
#     print(img_path)
#     prediction = inference(img_path)

#     result = prediction[0].tolist()
#     return jsonify({'predictions': result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
