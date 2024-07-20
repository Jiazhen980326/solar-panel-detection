# Solar panel fault detection

The presence of dust, snow, bird droppings, and other debris on the surface of solar panels can significantly reduce their efficiency and, consequently, the amount of energy they produce. Therefore, it is essential to regularly monitor and clean these panels to maintain optimal performance. Developing an effective procedure for monitoring and cleaning solar panels is crucial for enhancing module efficiency, lowering maintenance costs, and conserving resources.

The objective of this project is to evaluate the effectiveness of various machine learning classifiers in accurately detecting the presence of dust, snow, bird droppings, and other physical and electrical issues on solar panel surfaces. By identifying the best-performing classifiers, we aim to implement a reliable and efficient monitoring system that maximizes energy production and minimizes maintenance efforts.

## Dataset

Dataset used is from: <https://www.kaggle.com/datasets/pythonafroz/solar-panel-images?resource=download>

It contians 6 classes of the following:

```
Clean: This directory has images of clean solar panels
Dusty: This directory has images of dusty solar panels
Bird-drop: This directory has images of bird-drop on solar panels
Electrical-damage: This directory has images of electrical-damage solar panels
Physical-Damage: This directory has images of physical-damage solar panels
Snow-Covered: This directory has images of snow-covered on solar panels

```

## Library installation

```bash
git clone https://github.com/Jiazhen980326/solar-panel-detection.git
cd solar-panel-detection
python -m venv env
source env/bin/activate  # .\env\Scripts\activate
pip install -r requirements.txt
```

## Usage instruction

### 1. Local post request on flask app

```bash
python app.py 

curl -X POST -F "file=@/Users/yusali/dev/solar-panel-detection/data/Dusty/Dust (18).jpg" http://127.0.0.1:5001/predict_file
```

### 2. AML Local Docker Deployment

Follow ./notebooks/azure_ml_v2.ipynb Local deployment section.

make sure to Turn on the local = True parameter. `ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)`

### 3. Deploy to AML online deployment

Follow ./notebooks/azure_ml_v2.ipynb Online deployment section.

## Note

1. Apple chip acceleration:
The latest Python supported tensorflow-metal is python==3.11. [ref](https://discuss.tensorflow.org/t/tensorflow-on-apple-m2/14804/3)

2. There should be multiple ways to pass an image to online inference endpoint.

    TODO: Though only the base64 encode worked in my case, this needs further investigation.

> * Multipart Form Data: Ideal for web applications where users upload images through a form.
> * Base64 Encoded String: Useful when integrating with JSON-based APIs, where sending files is not convenient.
> * Image URL: Convenient when images are hosted online and accessible via URLs.
> * Raw Binary Data: Suitable for applications where images are sent as raw binary streams, such as in some IoT use cases.

3. Further score.py script debugging techniques can be found at [Debugging scoring script with Azure Machine Learning inference HTTP server](<https://learn.microsoft.com/en-us/azure/machine-learning/how-to-inference-server-http?view=azureml-api-2#debug-your-scoring-script-locally>)

4. The inference setup followed this page [Deploy and score a machine learning model by using an online endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python)

[AML Example repo](https://github.com/Azure/azureml-examples/tree/main/cli/endpoints/online/managed)
