import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import drive
drive.mount("/content/gdrive/")
test_dir='/content/gdrive/MyDrive/Brain_tumor/Testing'
train_dir='/content/gdrive/MyDrive/Brain_tumor/Training'
import tensorflow as tf
model= tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (100, 100, 3)),
        #tf.keras.layers.Dense(1000,activation=tf.nn.relu),#hidden
        #tf.keras.layers.Dense(500,activation=tf.nn.relu),
        tf.keras.layers.Dense(200,activation=tf.nn.relu),
        tf.keras.layers.Dense(20,activation=tf.nn.relu),
        tf.keras.layers.Dense(4,activation=tf.nn.softmax) # output layers
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1.0/255.)
test_datagen=ImageDataGenerator(rescale=1.0/255.)
import os
os.listdir(train_dir)
os.listdir(train_dir+'/glioma')
imort cv2
image=cv2.imread(train_dir+'/glioma/'+'Tr-gl_1273.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.imshow(image)  
batch_size=20
training_set=train_datagen.flow_from_directory(train_dir,
                                               target_size=(100,100),
                                               batch_size=batch_size,
                                               class_mode='categorical')
testing_set=test_datagen.flow_from_directory(test_dir,
                                             target_size=(100,100),
                                             batch_size=batch_size,
                                             class_mode='categorical')
import matplotlib.pyplot as plt
acc = history.history['acc']
#val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
#val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) 
plt.plot  ( epochs,     acc ,label = 'Accuracy')
#plt.plot  ( epochs, val_acc)
#plt.title ('Training and validation accuracy')
plt.title ('Training Accuracy')
plt.figure()

plt.plot  ( epochs,     loss )
#plt.plot  ( epochs, val_loss )
#plt.title ('Training and validation loss')
plt.title ('Training Loss')

plt.show()
classes=model.predict(testing_set)
folder_names=['glioma', 'pituitary', 'meningioma', 'no']
y_pred=[]
for folder in folder_names:
    path = test_dir+"/"+folder
    path_fnames = os.listdir(path)
    for i in path_fnames:
        path2 = path+'/'+i
        img = tf.keras.preprocessing.image.load_img(path2, target_size=(100,100))#target size ကို သတိထားပါ
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x /= 255.0
        images = np.vstack([x])# [1 2 3 4 5 6]
        classes = model.predict(x)
        y_classes=classes.argmax(axis=-1)
        y_pred.append(y_classes[0])
    print()
    y_true = testing_set.classes.tolist()
print(len(y_true))
class_dictionary = testing_set.class_indices
print('Labels dictionary',class_dictionary)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_true,y_pred)
print('confusion_matrix')
print(confusion_matrix)
y_true
y_pred
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print('Accuracy Score',accuracy_score(y_true, y_pred)*100,'%')
print('Precision Macro Score ',precision_score(y_true, y_pred,average = 'macro')*100,'%')
print('Recall_Score',recall_score(y_true, y_pred, average = 'macro')*100,'%')
print('F1_Score',f1_score(y_true, y_pred, average = 'macro')*100,'%')
model.save('/neural_networks.h5')
