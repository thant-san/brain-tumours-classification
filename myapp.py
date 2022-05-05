import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model=load_model('cnn_model_65eps.h5')

st.title("brain tumer classification",)
st.header("insert ur  mri image",)

st.set_option('deprecation.showfileUploaderEncoding', False)
file_upload=st.file_uploader("choose the mri file",type=['jpg','png','jpeg'])

image = Image.open(file_upload)
size=(227,227,3)
image=ImageOps.fit(image,size,Image.ANTIALIAS)
img=np.asarray(image)
img_reshape=img[np.newaxis,...]
st.image(img_reshape, caption='your mri image')
prediction=model.predict(img_reshape)
class_names=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
string=class_names[np.argmax(prediction)]
st.write("you have",string)
option=st.selectbox('Select your tumor_type',('none','glioma_tumor','meningioma_tumor','pituitary_tumor'))
st.write('You have selected',option)
if option=(glioma_tumor):
    st.write('''Common signs and symptoms of gliomas include:
               Headache.
                Nausea or vomiting.
                Confusion or a decline in brain function.
                 Memory loss.
                Personality changes or irritability.
                Difficulty with balance.
                Urinary incontinence.
                Vision problems, such as blurred vision, double vision or loss of peripheral vision.''')
else st.write('please'):
