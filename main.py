import os
import tensorflow as tf
from model import load_image, prepare_datasets, visualize_sample_images, build_model, train_model

if __name__ == "__main__":
    image_folder = './data'
    image_file = 'Bird-drop/Bird (1).jpeg'
    
    img_path = os.path.join(image_folder, image_file)
    print(f"Loading image from path: {img_path}")

    load_image(image_folder, image_file)

    img_height = 244
    img_width = 244
    train_ds, val_ds = prepare_datasets(image_folder, img_height, img_width)
    
    class_names = train_ds.class_names
    print(class_names)
    visualize_sample_images(train_ds, class_names)

    model = build_model(img_height=img_height, img_width=img_width)

    # Call the train_model function to train the model
    history = train_model(model, train_ds, val_ds, epochs=2)








