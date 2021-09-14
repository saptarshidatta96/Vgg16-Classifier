import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import cv2

objects = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

st.title("Image Classifier")
st.write('We have developed an web application for our Deep Learning Application.')
st.write('This Application can classify images based on the below 20 classes:')
st.markdown('aeroplane, bicycle, bird, boat, bottle, bus, car,cat, chair,cow, diningtable, dog, horse, motorbike,person, pottedplant, sheep, sofa, train, tvmonitor')
@st.cache()
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0).numpy()
    return image_tensor

def load_model(weights_file):
    model = keras.models.load_model(weights_file)
    return model
    

def load_predict(uploaded_file ,weights_file):
    data = preprocess_image(uploaded_file)
    model = load_model(weights_file)
    prediction = model.predict(data)
    return np.argmax(prediction)


if __name__ == "__main__":
    uploaded_file = st.file_uploader("Pls upload an image", type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded IMAGE', use_column_width=True)
        if st.button("Predict"):
            st.write("")
            st.write("Classifying...")
            label = load_predict(uploaded_file,  'trained_model.h5')
            st.write('The image uploaded belongs to:')
            st.text(objects[label])