import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Configure page styling
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    # Reconstruct the exact same architecture as used in training
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(150, 150, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    reconstructed_model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

    # Resolve path to weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We prioritize native keras file, fallback to weights if needed
    keras_model_path = os.path.join(current_dir, 'effnet.keras')
    legacy_model_path = os.path.join(current_dir, 'brain_tumor_model.h5')

    if os.path.exists(keras_model_path):
        # Load weights from the best model checkpoint (.keras file contains model structures and weights)
        reconstructed_model.load_weights(keras_model_path)
    elif os.path.exists(legacy_model_path):
        # Fallback to loading the legacy model structure + weights
        reconstructed_model = tf.keras.models.load_model(legacy_model_path)
    else:
        st.error("Model file not found! Please ensure 'effnet.keras' or 'brain_tumor_model.h5' is uploaded and exists in the same directory as app.py.")
        st.stop()
        
    return reconstructed_model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# App UI
st.title("🧠 Brain Tumor Classification from MRI")
st.write("Upload a brain MRI scan image (JPG, JPEG, or PNG) to classify it into one of the four categories.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image.', use_container_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (150, 150))
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing MRI scan...'):
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_idx] * 100

    # Format and present predicted class name
    predicted_class = labels[class_idx].replace('_', ' ').title()
    
    if labels[class_idx] == 'no_tumor':
        st.success(f"Result: **No Tumor Detected** ({confidence:.2f}% confidence)")
    else:
        st.warning(f"Result: **{predicted_class} Detected** ({confidence:.2f}% confidence)")
