import json
import numpy as np
import os
from tensorflow.keras.models import load_model

from azureml.core.model import Model

def init():
    global model
    
    model_root = Model.get_model_path('food-identifier')
    # load model
    model = load_model(model_root)


def run(raw_data):
    data = np.array(json.loads(raw_data)['image'])
    # make prediction
    # y_hat = np.argmax(model.predict(data), axis=1)
    response = model.predict(data)

    return response.tolist()