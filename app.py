import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

@st.cache_resource
def load_model():
    # Reconstruct the exact same architecture
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(150, 150, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    reconstructed_model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
<<<<<<< HEAD
    # Load the pure weights
    reconstructed_model.load_weights('effnet.weights.h5')
=======
    
    # Resolve the absolute path of the weights file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'effnet.weights.h5')
    
    # Robust check for existence
    if not os.path.exists(weights_path):
        st.error(f"Weights file not found! Expected at: {weights_path}\nPlease make sure you have uploaded 'effnet.weights.h5' to GitHub.")
        st.stop()
        
    # Load the pure weights
    reconstructed_model.load_weights(weights_path)
>>>>>>> 605a03f74dba119eaae6578d6204a91266cd56ab
    return reconstructed_model

model = load_model()
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

st.title("Brain Tumor Classification from MRI")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (150, 150))
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][class_idx] * 100

<<<<<<< HEAD
    st.success(f"Prediction: **{labels[class_idx]}** ({confidence:.2f}% confidence)")
=======
    st.success(f"Prediction: **{labels[class_idx]}** ({confidence:.2f}% confidence)")
>>>>>>> 605a03f74dba119eaae6578d6204a91266cd56ab
