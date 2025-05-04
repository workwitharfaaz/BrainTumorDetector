import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("BrainTumorDetector.keras")

# Define image preprocessing function
def preprocess_image(img, target_size=(224, 224)):  # adjust size as needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Streamlit UI
st.title("Brain Tumor Prediction")

uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)

    # Display result
    st.subheader("Prediction:")
    st.write(prediction)


    for row in prediction:
        if row[1] > row[0]:
            st.markdown("<h2 style='color:red;'>Tumor Detected</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>No Tumor Detected</h2>", unsafe_allow_html=True)