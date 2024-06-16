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
# Label or class name
kelas = [
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
class_name = [
    "Tire",
    "Plastic Bottles",
    "Plastic Packaging Waste",
    "Cups",
    "Gallon",
    "Glass",
    "Can",
    "Cardboard",
    "Paper",
    "Organic Waste"
]
# Function to preprocess image
def preprocess_image(img):
    try:
        # Ensure image is in RGB format
        img = img.convert('RGB')
        # Resize the image to 224x224
        img = img.resize((224, 224))
        # Normalize the image
        img_array = np.asarray(img) / 255.0
        # Add batch dimension
        img_array = img_array.reshape(1, 224, 224, 3)
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Function to predict trash categories from images (English labels)
def model_predict_EN(img):
    img_array = preprocess_image(img)
    if img_array is None:
        return "Error", 0.0
    predict_class = model_detec.predict(img_array)
    class_result = class_name[np.argmax(predict_class)]
    accuracy_result = np.max(predict_class) * 100
    return class_result, accuracy_result

# Function to predict trash categories from images (Indonesian labels)
def model_predict_ID(img):
    img_array = preprocess_image(img)
    if img_array is None:
        return "Error", 0.0
    predict_class = model_detec.predict(img_array)
    class_result = kelas[np.argmax(predict_class)]
    accuracy_result = np.max(predict_class) * 100
    return class_result, accuracy_result
