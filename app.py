import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_inception_model():
    model = load_model("best_model_InceptionV3.h5")
    return model

model = load_inception_model()

# Load class labels
@st.cache_data
def load_labels():
    with open("tumor_labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    return class_labels

class_labels = load_labels()

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure it's in RGB mode
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Predict
def predict(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]
    if top_idx >= len(class_labels):
        raise ValueError(f"Top index {top_idx} out of bounds for class_labels of length {len(class_labels)}")
    return class_labels[top_idx], confidence, predictions

# Streamlit UI
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI image, and the model will classify the tumor type with confidence scores.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")  # Handle grayscale and RGBA
        st.image(image, caption="Uploaded MRI Scan")

        st.info("⏳ Predicting tumor type...")
        label, confidence, all_scores = predict(image)

        st.success(f"🧾 Prediction: **{label}**")
        st.write(f"🔬 Confidence: **{confidence * 100:.2f}%**")

        # Confidence scores bar chart
        st.subheader("📊 Confidence for All Classes:")
        conf_data = {label: float(f"{score * 100:.2f}") for label, score in zip(class_labels, all_scores)}
        st.bar_chart(conf_data)

    except UnidentifiedImageError:
        st.error("❌ Error: The uploaded file is not a valid image.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<small>Developed By Janani Ravi</small>", unsafe_allow_html=True)
