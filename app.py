import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # path to the trained model

st.title("Forest Fire Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("Detecting...")

    # Perform detection
    results = model(image)

    # Display detection results
    results.show()

    # Convert results to an image with bounding boxes and display it
    st.image(results.render()[0], caption='Detection Result', use_column_width=True)
