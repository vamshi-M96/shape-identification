import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
MODEL_PATH = "C:/Users/Nitro V 15/OneDrive/Desktop/kaggel project/archive (1)/my_model.h5"  # Path to your saved model
model = load_model(MODEL_PATH)

# Define class names
class_names = ["Ciircle", "Square", "Triangle"]  # Replace with your actual class names

# Preprocess the input image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)  # Resize image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app layout
st.title("Image Classification App")
st.write("Upload an image, and the model will predict its class.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the result
    st.write(f"### Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")