import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

import requests
import os


# Download model from Google Drive link if not already downloaded

import tempfile


# 🔥 Updated model loading with HuggingFace link
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/Vasu10khanna/deepcrack-model/resolve/main/deepcrack_vgg16_unet.h5"
    r = requests.get(model_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        temp_file.write(r.content)
        temp_file.flush()
        model = keras.models.load_model(temp_file.name, compile=False)
    return model

model = load_model()


# Title
st.title("🛣️ DeepCrack Detection App")

# Upload Image
uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read Image using PIL
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(input_tensor)[0].squeeze()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Crack stats
    cracked_pixels = np.sum(pred_mask > 0)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]


    if crack_percent > 3:
        action = "⚠️ Immediate Repair Needed"
    elif crack_percent > 2:
        action = "🛠️ Monitor - Not Urgent"
    else:
        action = "✅ No Repair Needed"

    overlay = cv2.resize(img_rgb, (256, 256))  # Match size to 256x256
    overlay[pred_mask == 255] = [255, 0, 0]  # Blue cracks


    # Display
    st.subheader("Original Image")
    st.image(img_rgb, use_column_width=True)

    st.subheader("Predicted Crack Mask")
    st.image(pred_mask, clamp=True, channels="GRAY")

    st.subheader("Overlayed Result")
    st.image(overlay, use_column_width=True)

    # Display Metrics
    st.success(f"🧮 Crack Coverage: {crack_percent:.2f}%")
    st.info(f"🚨 {action}")
