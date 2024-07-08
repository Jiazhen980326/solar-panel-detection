import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import warnings

warnings.filterwarnings("ignore")


def load_image(image_folder, image_file):
    """Load and display a single image"""
    img_path = os.path.join(image_folder, image_file)
    print(f"Attempting to load image from: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Error reading image {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Image: {image_file}")
    plt.axis("off")
    plt.show(block=False)


def load_images(image_folder):
    """Load all images from the specified folder"""
    images = []
    for img_path in glob(f"{image_folder}/*"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return images


def prepare_datasets(
    image_folder, img_height=244, img_width=244, batch_size=32, seed=42
):
    """Prepare training and validation datasets"""
    # Add a Rescaling layer to normalize pixel values
    # normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        image_folder,
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        image_folder,
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    # Apply the normalization to the dataset
    # train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds


def visualize_sample_images(train_ds, class_names):
    """Visualize sample images"""
    plt.figure(figsize=(15, 15))
    for images, labels in train_ds.take(1):
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def build_model(img_height=244, img_width=244):
    """Build the model"""
    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(6)(x)
    outputs = tf.keras.activations.softmax(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def train_model(model, train_ds, val_ds, epochs=15):
    """Train the model"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=3,
        verbose=1,
        restore_best_weights=True,
    )

    # Define ModelCheckpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
    )

    # Put the callbacks in a list
    callbacks = [early_stopping, model_checkpoint]

    # Train the model with the callbacks
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )

    return history
