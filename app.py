import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Medical Image Diagnosis",
    layout="centered"
)

# Title
st.title("Medical Image Diagnosis Using Deep Learning")
st.write("Upload a Chest X-Ray image to predict whether it is NORMAL or PNEUMONIA.")

# Load Model
@st.cache_resource
def load_trained_model():
    model = load_model("pneumonia_detection_model.h5")
    return model

model = load_trained_model()

# File Upload
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Image
    img = Image.open(uploaded_file)

    # Convert to RGB (3 channels)
    img = img.convert("RGB")

    st.image(
        img,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    # Preprocessing
    img = img.resize((224,224))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(img_array)

    probability = prediction[0][0]

    if probability > 0.5:

        st.error("Prediction: PNEUMONIA")

        st.write(
            f"Confidence: {probability*100:.2f}%"
        )

    else:

        st.success("Prediction: NORMAL")

        st.write(
            f"Confidence: {(1-probability)*100:.2f}%"
        )