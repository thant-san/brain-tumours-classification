import streamlit as st

from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.model import load_model
from tensorflow.keras.activation import softmax
import os
st.title("Brain tumer classification")
st.header("insert ur  mri file below")
def main():
     file_upload=st.file_uploader("choose the mri file",type=['jpg','png','jpeg'])
     image=Image.open(file_upload)
     figure=plt.figure()
     plt.imshow(image)
     plt.axis('off')
     result=predict_class(image)
     st.write(result)
     st.pyplot(figure)
#def predict_class(image):
     classifier_model=tf.keras.models.load_model('cnn.h5')
     shape={(100,100,3)}
     tf.keras.Sequential(hub[hub.KerasLayer(classifier_model,input_shape=shape)])
     test_image=image.resize((128,128))
     test_image=preprocessing.image.img_to_array(test_image)
     test_image=test_image/255.0
     test_image=np.expand.dims(test_image,axis=0)
     class_names=['glioma_tumor', 'pituitary_tumor', 'meningioma_tumor', 'no_tumor']
