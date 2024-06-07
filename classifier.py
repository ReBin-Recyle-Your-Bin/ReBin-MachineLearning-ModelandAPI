import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Load Model Image Classification
MY_MODEL = 'modelDenseNet121.h5'
model_detec = tf.keras.models.load_model(MY_MODEL)
# Label kategori sampah yang dapat diprediksi
class_name = [
    "Ban",
    "Botol Plastik",
    "Bungkus Plastik",
    "Cup Gelas",
    "Galon",
    "Kaca",
    "Kaleng",
    "Kardus",
    "Kertas",
    "Sampah Organik"
]
# Function to predict trash categories from images
def model_predict(img):
    # Preprocessing the image and resizing it to 224x224
    i = np.asarray(img.resize((224, 224))) / 255.0
    i = i.reshape(1, 224, 224, 3)
    # Make predictions using models
    predict_class = model_detec.predict(i)
    # Get the waste category label with the highest probability value
    class_result = class_name[np.argmax(predict_class)]
    # Calculates prediction accuracy in percentage
    accuracy_result = np.max(predict_class) * 100
    return class_result, accuracy_result
