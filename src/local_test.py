# local_test.py
import score  # This imports your score.py script
import json
import requests
from PIL import Image
import base64


def test_local_deployment():
    # Initialize the model
    score.init()

    img_path = "data/Electrical-damage/Electrical (2).JPG"

    # # Send the request with the image as binary data
    # with open(img_path, "rb") as img_file:
    #     # files = {"file": img_file}
    #     files = img_file.read()
    # # img = Image.open(img_path)

    # # result = score.run(files)

    # Read the image and encode it in Base64
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # # Run the model with the test input
    input_json = json.dumps({"image": img_base64})
    result = score.run(input_json)

    # Print the result
    print("Result:", result)


if __name__ == "__main__":
    test_local_deployment()
