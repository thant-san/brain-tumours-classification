# Brain Tumours Classification
- A brain tumor is an abnormal growth of tissue in the brain or central spine that can disrupt proper brain function. It is the abnormal growth of tissues in brain. If the        tumor originates in the brain, it is called a primary brain tumor. Primary brain tumors can be benign or malignant. Benign brain tumors are not cancerous.
>Dataset link:https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

>Download datasets directky and unzip it

>Train images 7023 with images size (100,100)
### Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 30000)             0         
                                                                 
 dense (Dense)               (None, 200)               6000200   
                                                                 
 dense_1 (Dense)             (None, 20)                4020      
                                                                 
 dense_2 (Dense)             (None, 4)                 84        
                                                                 
=================================================================
- Total params: 6004304 (22.90 MB)
- Trainable params: 6004304 (22.90 MB)
- Non-trainable params: 0 (0.00 Byte)
### Requirement
- opencv
- tensorflow
### Training accuracy
![Untitled](Untitled.png)
### Training loss
![Untitled](training_loss.png)

### Streamlit
>https://thant-san-prj-mri-myapp-fg8wgb.streamlit.app/
### Acknowledgement
- The model is still not perfect and it is also our first hand on project.

### Contributer
- Thant htoo san
- Htoo lynn htet
- Aung moe kyaw
### Refernce
- https://github.com/dheerajnbhat/Brain-Tumor-Detection
