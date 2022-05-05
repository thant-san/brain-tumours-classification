%tensorflow_version 2.x
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
drive.mount("/content/gdrive/")
train_dir = '/content/gdrive/MyDrive/Project/Brain_tumor/Training'
val_dir = '/content/gdrive/MyDrive/Project/Brain_tumor/Testing'

test_dir = '/content/gdrive/MyDrive/Project/Brain_tumor22/Testing'
import os
os.listdir(train_dir+'/notumor')
import cv2
import numpy as np
img1 = cv2.imread(train_dir+'/meningioma/Tr-me_1182.jpg')
img1_array = np.array(img1)
img1_array.shape
img2 = cv2.imread(train_dir+'/notumor/Tr-no_1477.jpg')
img2_array = np.array(img2)
img2_array.shape
import matplotlib.pyplot as plt
plt.imshow(img1)
plt.imshow(img2)
half = cv2.resize(img2, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(img2, (128, 128))
 
wanted = cv2.resize(img2, (227, 227),
               interpolation = cv2.INTER_NEAREST)
 
 
Titles =["Original", "Half", "128", "Wanted"]
images =[img2, half, bigger, wanted]
count = 4
 
for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])
 
plt.show()
print(f'wanted shape : ',wanted.shape)
print(f'Original shape : ',img2.shape)
print(f'Half shape : ',half.shape)
print(f'128 shape : ',bigger.shape)
train_imgGen = ImageDataGenerator(rescale = 1.0/255,
                                  rotation_range = 30,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                            shear_range= 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
                            fill_mode = 'nearest'
                            )
test_imgGen = ImageDataGenerator(rescale = 1.0/255)

test_img = ImageDataGenerator(rescale = 1.0/255)
training_set = train_imgGen.flow_from_directory(train_dir, target_size = (277,227),
                                         batch_size = 30,
                                         class_mode = 'categorical')

val_set = test_imgGen.flow_from_directory(val_dir, target_size = (277,227),
                                               batch_size = 30,
                                               class_mode = 'categorical')

test_set = test_img.flow_from_directory(test_dir, target_size = (277, 277),
                                               batch_size = 30,
                                               class_mode = 'categorical')

training_set.image_shape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Input(shape = (227,227,3)))
#1 input and convo
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (6,6), strides = 2, activation = 'relu'))


cnn.add(tf.keras.layers.BatchNormalization())

#3 maxpool
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.MaxPool2D((2,2), strides = 2))

#4 convo
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.Conv2D(filters = 96, kernel_size = (6,6), strides = 2, padding = 'valid', activation = 'relu'))

#5 relu
cnn.add(tf.keras.layers.Activation('relu'))

#6 maxpool
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.MaxPool2D((2,2), strides = 1))

#7 convo
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.Conv2D(filters = 24, kernel_size = (6,6), strides = 2, padding = 'valid', activation = 'relu'))


#9 maxpool
cnn.add(tf.keras.layers.ZeroPadding2D(padding = (2,2)))
cnn.add(tf.keras.layers.MaxPool2D((2,2), strides = 2))

#10 convo
cnn.add(tf.keras.layers.Conv2D(filters = 24, kernel_size = (6,6), padding = 'valid', strides = 2,
                               activation = 'relu'))

#maxpool
cnn.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2))

cnn.add(tf.keras.layers.BatchNormalization(-1))

#13flatten
cnn.add(tf.keras.layers.Flatten())

#14fully connected
cnn.add(tf.keras.layers.Dense(512, activation = 'relu'))

cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Dense(512, activation = 'relu'))
#15
cnn.add(tf.keras.layers.Dropout(0.4))

#16
cnn.add(tf.keras.layers.Dense(256, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(120, activation = 'relu'))


cnn.add(tf.keras.layers.Dropout(rate=0.3))

cnn.add(tf.keras.layers.Dense(4, activation = tf.nn.softmax))

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(cnn.summary())
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.utils.vis_utils import plot_model
plot_model(cnn)
r = cnn.fit(x =training_set, validation_data = val_set, epochs = 110, verbose = 1)
cnn.save_weights('/content/gdrive/MyDrive/Deep Learning/ANN/cnn2_weights_110eps.h5')
tf.keras.models.save_model(cnn, '/content/gdrive/MyDrive/Deep Learning/ANN/cnn_model_110eps.h5')
folder_names = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']


y_true = test_set.classes.tolist()

y_pred = []
for folder in folder_names:
    path = test_dir+"/"+folder
    path_fnames = os.listdir(path)
    for i in path_fnames:
        path2 = path+'/'+i
        img = tf.keras.preprocessing.image.load_img(path2, target_size=(128,128))#target size ကို သတိထားပါ
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x /= 255.0
        images = np.vstack([x])# [1 2 3 4 5 6]
        classes = cnn.predict(x)
        y_classes=classes.argmax(axis=-1)
        y_pred.append(y_classes[0])
        from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_true,y_pred)

print(confusion_matrix)



