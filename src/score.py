import os
import logging
import json
import numpy
import tensorflow as tf
from model_train_inference.inference import inference

from PIL import Image
from io import BytesIO
import base64

# from requests_toolbelt.multipart import decoder


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    # )
    # deserialize the model file back into a sklearn model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "best_model.keras")
    # model_path = "model/best_model.keras"
    model = tf.keras.models.load_model(model_path)

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    # logging.info("model 1: request received")
    # prediction = inference(img, model=model, mode="file")
    # logging.info("Request processed")
    # return prediction.tolist()

    try:
        # raw_data is expected to be multipart form data
        logging.info("model 1: request received")

        # Parse the JSON data
        data = json.loads(raw_data)
        # Decode the Base64 encoded image data
        img_data = base64.b64decode(data["image"])
        # multipart_data = decoder.MultipartDecoder(img)

        img = Image.open(BytesIO(img_data))
        logging.info("model 1: input image loaded, starting inference")
        prediction = inference(img, model=model, mode="file")
        logging.info("Request processed")

        return json.dumps({"predicted_class": prediction})
    except Exception as e:
        error = str(e)
        print("Error:", error)
        return json.dumps({"error": error})
