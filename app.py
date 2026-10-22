import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page configuration
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_cnn_model.keras")

model = load_model()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Application title
st.title("🩺 COVID-19 Detection Using CNN")

st.write(
    "Upload a chest X-ray image to classify it "
    "as COVID-19, Normal, or Viral Pneumonia."
)



# Upload X-ray image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width="stretch"
    )

    # Preprocess image
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32")

    # Normalize pixel values to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict button
    if st.button("Predict"):

        prediction = model.predict(image_array)

        predicted_class = np.argmax(prediction[0])

        predicted_label = class_names[predicted_class]

        confidence = np.max(prediction[0]) * 100

        # Display prediction
        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {predicted_label}"
        )

        st.write(
            f"Model Confidence: {confidence:.2f}%"
        )

        # Display class probabilities
        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = float(prediction[0][i])

            st.write(
                f"{class_name}: {probability * 100:.2f}%"
            )

            st.progress(probability)
