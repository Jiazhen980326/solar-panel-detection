import tensorflow as tf
from azureml.core import Workspace
from azureml.core.model import Model

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Define and save the TensorFlow model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(10, input_shape=(3,), activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.save("path_to_your_model/my_model")

# Define the model path and name
model_path = "path_to_your_model/my_model"
model_name = "your_model_name"

# Register the model
model = Model.register(workspace=ws, model_path=model_path, model_name=model_name)

# Verify the model registration
models = Model.list(ws)
for m in models:
    print(m.name, m.id, m.version)
