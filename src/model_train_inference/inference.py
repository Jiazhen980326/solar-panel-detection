import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image


def load_and_preprocess_image(img, mode="path", target_size=(244, 244)):
    # Load the image
    if mode == "path":
        img = image.load_img(img, target_size=target_size)
    else:
        img = img.resize(target_size)
        print(img.size)
        # img = np.squeeze(img, axis=0)

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Preprocess the image using VGG16 preprocess_input
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def inference(img_path, model, mode="path"):
    img = load_and_preprocess_image(img_path, mode=mode)
    logits = model.predict(img)
    cls_idx = np.argmax(logits)

    cls_names = [
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
    ]
    return cls_names[cls_idx]


if __name__ == "__main__":
    img_path = "data/Dusty/Dust (58).jpg"
    img = Image.open(img_path)
    prediction = inference(img, mode="file")
    print(prediction)
