import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from imp import load_compiled

import json
import requests
from streamlit_lottie import st_lottie


model=load_model('cnn_model_65eps.h5')

st.title("Brain tumer classification",)
st.header("Insert ur  mri image",)
def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
st.set_option('deprecation.showfileUploaderEncoding', False)
file_upload=st.file_uploader("Choose the mri file",type=['jpg','png','jpeg'])
if file_upload is None:
    lottie_coding=load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_iarc855d.json")
    st_lottie(lottie_coding,height=100,width=100,key=None)
    st.write("you haven't put any images yet!")
else:    
    image = Image.open(file_upload)
    size=(227,227)
    image=ImageOps.fit(image,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    st.image(img_reshape, caption='your mri image')
    prediction=model.predict(img_reshape)
    class_names=['glioma','meningioma','notumor','pituitary']
    string=class_names[np.argmax(prediction)]
    st.write("you have ",string)


expander=st.expander(" you can also check symptons of tumor")
with expander:
    option=st.selectbox('Select your tumor_type',('glioma_tumor','meningioma_tumor','pituitary_tumor'))
    st.write('You have selected',option)
    if option=='glioma_tumor':
        st.write("""
                      Common symptoms of Gliomas:Headache.
                      Nausea or vomiting.
                      Confusion or a decline in brain function.
                      Memory loss.
                      Personality changes or irritability.
                      Difficulty with balance.
                      Urinary incontinence.
                      Vision problems, such as blurred vision, double vision or loss of peripheral vision
                  """)
    elif option=='meningioma_tumor':
         st.write ("""Common symptoms of Meningioma_tumor:
                       Changes in vision, such as seeing double or blurriness.
                       Headaches, especially those that are worse in the morning.
                       Hearing loss or ringing in the ears.
                       Memory loss.
                       Loss of smell.
                       Seizures.
                       Weakness in your arms or legs.
                       Language difficulty.""") 
    elif option=='pituitary_tumor':
         st.write(""" Common symptonms of pituitary_tumor:
                     Nausea and vomiting.
                     Weakness.
                     Feeling cold.
                     Less frequent or no menstrual periods.
                     Sexual dysfunction.
                     Increased amount of urine.
                     Unintended weight loss or gain.""")
    else :
         st.write('please select')
st.sidebar.write('Developed by INTEL AI')
lottie_contact=load_lottieurl_1("https://assets7.lottiefiles.com/packages/lf20_zj3qnsfs.json")
ani=st_lottie(lottie_contact,height=60,width=60,key=None)
st.sidebar.write("""If you have any issues contact us
                 thanthtoosan.@mechatronic@gmail.com""",ani)
