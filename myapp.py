import streamlit as st
st.title("Brain tumer classification")
st.header("insert ur file below")
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)
model=load_model('cnn.h5')
