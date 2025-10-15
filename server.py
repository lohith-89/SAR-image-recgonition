import base64
import numpy as np
import io
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import os

from flask import request, jsonify, Flask, send_from_directory, json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class_labels = {
    0: '2S1',
    1: 'BMP2',
    2: 'BRDM2',
    3: 'BTR60',
    4: 'BTR70',
    5: 'D7',
    6: 'SLICY',
    7: 'T62',
    8: 'T72',
    9: 'ZIL132',
    10: 'ZSU_23_4'
}

def get_model():
    global model
    MODEL_FILENAME = "model1.h5"  # Update with your model filename
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" * Model loaded!")

def decode_image(encoded_image):
    try:
        image = tf.image.decode_image(encoded_image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
        return image
    except tf.errors.InvalidArgumentError:
        return None

print("Loading Keras model...")
get_model()

@app.route('/')
def index():
    return send_from_directory('.', 'predict.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        print("Received image data:", decoded[:100])
        image = decode_image(decoded)
        if image is None:
            response = {'error': 'Unsupported image format. Please provide an image in JPEG, PNG, GIF, or BMP format.'}
            return jsonify(response), 400
        else:
            image = tf.expand_dims(image, axis=0)
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = class_labels[predicted_class_index]
            print("Predicted Class:", predicted_class)
            response = {
                'predicted_class': predicted_class,
                'classified_image': encoded
            }
            return jsonify(response)
    except Exception as e:
        print("Error:", e)
        response = {'error': 'An error occurred while processing the request.'}
        return jsonify(response), 500

if __name__ == "__main__":
    app.run(debug=True)
