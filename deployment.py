import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2

st.title("Damage Detector")

# Image uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Open image and display original image
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Original Image", use_container_width=True)
    
    # Load YOLO model (make sure best.pt is in the same folder or provide correct path)
    model = YOLO("best.pt")
    
    # Perform inference
    results = model(pil_image)
    
    # Get the annotated image from results[0].plot()
    # Note: results[0].plot() returns a NumPy array in BGR format; convert it to RGB.
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Display annotated image
    st.image(annotated_image, caption="Processed Image", use_container_width=True)
