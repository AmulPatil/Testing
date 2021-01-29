
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.basic_train import load_learner
from fastai.vision import open_image



# load the learner
learn = load_learner(path='./', file='trained_model.pkl')
classes = learn.data.classes



st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    #prediction = import_and_predict(image, model)
    
    prediction = learn.predict(open_image(image))
    probs_list = prediction[2].numpy()

    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write({
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    })

