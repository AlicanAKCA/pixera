import os
import cv2
import logging
import json
import numpy
import joblib
import numpy as np
from PIL import Image

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")

def run(filename):
    gif = Image.open(filename)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = Image.new('RGB', gif.size)
        frame.paste(gif)
        frame = np.array(frame)
        frames.append(frame)
    result = pixl.toThePixL(frames, 6)
    frames = []
    for frame in result:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)    
        frames.append(frame)
    frames[0].save('new.gif', append_images=frames[1:], save_all=True, loop=1)
    frames = None
    gif = None
    result = None
    return Image.open('new.gif')