import streamlit as st


st.title("Brain tumer classification")
st.header("insert ur  mri file below")

     file_upload=st.file_uploader("choose the mri file",type=['jpg','png','jpeg'])
     image=Image.open(file_upload)
     
