import os

import cv2 as cv
import numpy as np
from flask import jsonify, request
from flask_restful import Resource
from tensorflow.keras.models import load_model
import tensorflow as tf


# set GPU memory usage limitation
tf.config.gpu.set_per_process_memory_fraction(0.75)
tf.config.gpu.set_per_process_memory_growth(True)

# set model_path
model_path = "model.h5"

if model_path:
    model = load_model(model_path)

if not model_path:
    raise ValueError("Set your model path first.")

class Model(Resource):
    def get(self):
        return "You should use 'POST'"

    def post(self):
        """Get predictions of image"""
        image = request.files['image'].read()
        image_buffer = process_image(image)
        prediction = model.predict_classes(image_buffer)[0]
        result = {
            "prediction": str(prediction)
        }
        return jsonify(result)

def process_image(image_buffer):
    """Make you image to fit your input_shape of model
    
    Parameters
    ------------
    image_buffer: str
        Buffer string of image

    Returns:
    ------------
    The shape you want.

    """
    # image shape
    input_shape = (-1, 224, 224, 1)

    # >0 Return a 3-channel color image,
    # =0 Return a grayscale iamge,
    # <0 Return the loaded image as is (with alpha channel)
    flag = 0

    # read buffer string to array in flattened shape.
    image_array = np.frombuffer(image_buffer, dtype=np.uint8)
    image = cv.imdecode(image_array, flag)
    image = image.reshape(input_shape)
    
    return image
