import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image # for image manipulation
# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')
# Class labels dictionary
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
# Function to preprocess and predict with error handling
def predict_image(inp):
 # Convert PIL image to NumPy array
 image_array = np.array(inp)
 # Resize the image to match model's expected shape (optional based on your model)
 # Assuming your model expects 224x224, uncomment the following line if needed
 image_array = tf.image.resize(image_array, (224, 224))
 # Preprocess the image based on your model's requirements (replace with your specific steps)
 # This example assumes normalization (divide by 255)
 image_array = tf.cast(image_array, tf.float32) / 255.0
 # Expand dimension for batch processing
 image_array = tf.expand_dims(image_array, axis=0)
 # Make prediction and handle potential errors
 try:
    prediction = model.predict(image_array)
    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_labels.get(predicted_class_index, 'Unknown')
    return predicted_class
 except Exception as e:
    print("Error:", e)
    return "An error occurred during prediction." # Return error message
# Create Gradio interface
interface = gr.Interface(
 fn=predict_image,
 inputs=gr.Image(type="pil"),
 outputs=gr.Label(num_top_classes=1),
 title="Military Vehicle Classifier",
 description="Upload an image of a military vehicle to classify it."
)
# Launch the Gradio app
interface.launch()

