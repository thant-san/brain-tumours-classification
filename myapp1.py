import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
#st.sidebar.write("Setting")
model=load_model('neural_networks.h5')

st.title("brain tumer classification",)
st.header("insert ur  mri image",)

#st.set_option('deprecation.showfileUploaderEncoding', False)
file_upload=st.file_uploader("choose the mri file",type=['jpg','png','jpeg'])

image = Image.open(file_upload)
size=(100,100)
image=ImageOps.fit(image,size,Image.ANTIALIAS)
img=np.asarray(image)
img_reshape=img[np.newaxis,...]
st.image(img_reshape, caption='your mri image')
prediction=model.predict(img_reshape)
class_names=['glioma', 'pituitary', 'meningioma', 'no']
string=class_names[np.argmax(prediction)]
st.success("you have",string)
