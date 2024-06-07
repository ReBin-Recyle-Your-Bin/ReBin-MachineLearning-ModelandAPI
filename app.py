import json
from flask import Flask, request, jsonify
import io
from PIL import Image
from keras.models import load_model
from classifier import model_predict
from recycleRecomendation import get_recommendation
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get('file')
    if f is None or f.filename == "":
        return jsonify({"error": "No file provided."}), 400
    
    extension = f.filename.split(".")[-1].lower()
    if extension not in ("jpg", "jpeg", "png"):
        return jsonify({"error": "Image must be in jpg, jpeg, or png format!"}), 400
    
    try:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224), Image.BILINEAR)
        
        pred_label, pred_accuracy = model_predict(img)
        recommendation_result = get_recommendation(pred_label)
        
        return jsonify({
            "label": pred_label,
            "accuracy": f"{pred_accuracy:.2f}%",
            "recommendation": recommendation_result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
