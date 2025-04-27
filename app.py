import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import requests
import tempfile

# 🔥 Updated model loading with HuggingFace link
@st.cache_resource
def load_tflite_model():
    model_url = "https://huggingface.co/Vasu10khanna/deepcrack-model/resolve/main/deepcrack_vgg16_unet.tflite"
    r = requests.get(model_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as temp_file:
        temp_file.write(r.content)
        temp_file.flush()
        interpreter = tf.lite.Interpreter(model_path=temp_file.name)
        interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Prediction function
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Title
st.title("🛣️ DeepCrack Detection App (Optimized with TFLite)")

# Upload Image
uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read Image using PIL
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    pred = predict_tflite(interpreter, input_tensor)[0].squeeze()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Crack stats
    cracked_pixels = np.sum(pred_mask > 0)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    crack_percent = (cracked_pixels / total_pixels) * 100

    # Decision logic
    if crack_percent > 3:
        action = "⚠️ Immediate Repair Needed"
    elif crack_percent > 2:
        action = "🛠️ Monitor - Not Urgent"
    else:
        action = "✅ No Repair Needed"

    # Create overlay
    overlay = np.array(img_resized.copy())
    overlay[pred_mask == 255] = [255, 0, 0]  # Mark cracks in blue

    # Display
    st.subheader("Original Image")
    st.image(img_resized, use_column_width=True)

    st.subheader("Predicted Crack Mask")
    st.image(pred_mask, clamp=True, channels="GRAY")

    st.subheader("Overlayed Result")
    st.image(overlay, use_column_width=True)

    # Display Metrics
    st.success(f"🧮 Crack Coverage: {crack_percent:.2f}%")
    st.info(f"🚨 {action}")
