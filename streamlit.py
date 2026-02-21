from tkinter import Image

import streamlit as st
import fastai.vision.all import *

st.title("Pet Breed Classification")
st.text("Built by David Shaun")

def extract_breed_name(file_name):
    p = Path(file_name)
    breed_name_parts = p.stem.split("_")
    final_breed_name = ""
    length_parts = len(breed_name_parts)-1
    for i in range(length_parts):
        breed_name_part = breed_name_parts[i]
        final_breed_name += breed_name_part
        if i != length_parts - 1:
            final_breed_name += "_"
    return final_breed_name

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = breed_model.predict(img)
    return pred_class


breed_model = load_learner("cst_dog_breed_model_3.8.6.pkl")

uploaded_file = st.file_uploader("Upload an image...", type =["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)

    st.subheader(f"Predicted Breed: {prediction}")

    st.text("Built with Fastai and Streamlit")