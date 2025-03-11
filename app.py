import streamlit as st
import numpy as np
import time
from PIL import Image
from keras.models import load_model

# Load Model
model = load_model('model1_cifar_10epoch.h5')

# Class Labels
classes = {
    0: 'Aeroplane✈️', 1: 'Automobile🚘', 2: 'Bird🐦', 3: 'Cat🐈', 4: 'Deer🦌',
    5: 'Dog🐶', 6: 'Frog🐸', 7: 'Horse🐎', 8: 'Ship🚢', 9: 'Truck🚒'
}

# Streamlit UI
st.set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 50px;'>✨CIFAR-10 Image Classifier✨</h1>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='text-align: center;'>Upload⬆️ Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("<h3 style='text-align: center;'>Prediction❔</h3>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        classify_button = st.button("Classify Image")
        st.markdown("</div>", unsafe_allow_html=True)

        if classify_button:
            with st.spinner("Classifying..."):
                time.sleep(1)
                image = image.resize((32, 32))
                image = np.expand_dims(image, axis=0)
                image = np.array(image)
                prediction = np.argmax(model.predict(image), axis=-1)[0]
                label = classes[prediction]
            st.success(f"Predicted Class: {label}")