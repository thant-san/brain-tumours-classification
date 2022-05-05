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
size=(227,227)
image=ImageOps.fit(image,size,Image.ANTIALIAS)
img=np.asarray(image)
img_reshape=img[np.newaxis,...]
st.image(img_reshape, caption='your mri image')
prediction=model.predict(img_reshape)
class_names=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
string=class_names[np.argmax(prediction)]
st.write("you have",string)

with st.expander("KNOW MORE ABOUT SYMPTONS OF TUMOR"):
option=st.selectbox('Select your tumor_type',('glioma_tumor','meningioma_tumor','pituitary_tumor'))
st.write('You have selected',option)
if option=='glioma_tumor':
        st.write("""Common symptoms of Gliomas:
                Headache.
                Nausea or vomiting.
                Confusion or a decline in brain function.
                 Memory loss.
                Personality changes or irritability.
                Difficulty with balance.
                Urinary incontinence.
                Vision problems, such as blurred vision, double vision or loss of peripheral vision""")
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
         st.write("""Nausea and vomiting.
                Weakness.
                Feeling cold.
                Less frequent or no menstrual periods.
                Sexual dysfunction.
                Increased amount of urine.
                Unintended weight loss or gain.""")
else :
         st.write('please select')
    
