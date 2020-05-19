from importlib import import_module
import os
import jsonpickle
from flask import Flask,Response , request , flash , url_for,jsonify
import logging
from logging.config import dictConfig
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask 
from tensorflow.keras.preprocessing import image
app = Flask(__name__)

@app.route('/') 
def hello_world():
    return "Hello, World!"

@app.route('/classifier/run',methods=['POST'])
def classify():
    app.logger.debug('Running classifier')
    upload = request.files['data']
    #load_image() is to process image : 
    image = load_image(upload)
    print('image ready')
    classifier = load_model('my_model_multiclass10.h5') #load the model that was created using cnn_multiclass.py
    result = classifier.predict(image) # returns array

    if result[0][0] == 1:
    	prediction = 'bridge' #predictions in array are in alphabetical order
    elif result[0][1] == 1:
    	prediction = 'childspose'
    elif result[0][2] == 1:
        prediction = 'downwarddog'
    elif result[0][3] == 1:
        prediction = 'mountain'
    elif result[0][4] == 1:
        prediction = 'plank'
    elif result[0][5] == 1:
        prediction = 'seatedforwardbend'
    elif result[0][6] == 1:
        prediction = 'tree'
    elif result[0][7] == 1:
        prediction = 'trianglepose'
    elif result[0][8] == 1:
        prediction = 'warrior1'
    elif result[0][9] == 1:
        prediction = 'warrior2'
    return prediction


def load_image(filename):    
    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
