import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, jsonify

app = Flask(__name__)


model = load_model('best_model.h5')

def load_and_preprocess_image(img_path, target_size=(244, 244)):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Preprocess the image using VGG16 preprocess_input
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def inference(img_path):
    img = load_and_preprocess_image(img_path)
    return model.predict(img)


@app.route('/predict_file', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_path = file
    print(img_path)

    prediction = inference(img_path)
    
    result = prediction[0].tolist()  
    return jsonify({'predictions': result})


# @app.route('/predict', methods=['POST'])
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



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
