# import json
# import plotly
# import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
# from plotly import graph_objects

# import joblib
import cv2
import numpy as np
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint  
from keras.preprocessing import image                  
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions

from extract_bottleneck_features import extract_InceptionV3
from dog_classifier import *


app = Flask(__name__)


# load model
# model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    img_path = request.args.get('query', '') 

    # use model to predict classification for query
    classification_result = classify_dog_breed(img_path)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        img_path=img_path,
        classification_result=classification_result
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()