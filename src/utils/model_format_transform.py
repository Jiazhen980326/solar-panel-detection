import tensorflow as tf
import tf2onnx
import onnx


def convert_model_to_onnx(model_path, output_path):
    # Load your Keras model
    model = tf.keras.models.load_model(model_path)  # or .h5 if it's in HDF5 format

    # Convert the model
    spec = (tf.TensorSpec((None,) + model.input_shape[1:], tf.float32),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
