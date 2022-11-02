# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Krish Naik
"""
from __future__ import division, print_function
# coding=utf-8
import sys
# ----------------------------------------------------------------------------------------------------------------------
import glob
import re
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
# Keras
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
# ----------------------------------------------------------------------------------------------------------------------
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from gevent.pywsgi import WSGIServer
# ----------------------------------------------------------------------------------------------------------------------
# TENSORFLOW WITH COMPILER FLAGS - TEST
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ----------------------------------------------------------------------------------------------------------------------
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_vgg.h5'

# Load your trained model
model = load_model(MODEL_PATH)


# ----------------------------------------------------------------------------------------------------------------------

def model_predict(img_path, model):
    img = keras.utils.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = keras.utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Person is Infected With Pneumonia"
    else:
        preds = "The Person is not Infected With Pneumonia"

    return preds


# ----------------------------------------------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# ----------------------------------------------------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


# ----------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#   app.run(debug=True)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
# ----------------------------------------------------------------------------------------------------------------------
