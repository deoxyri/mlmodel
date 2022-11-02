import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg19 import VGG19

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.layers import MaxPooling2D
import scipy
# ----------------------------------------------------------------------------------------------------------------------
# LOADING IMAGES
IMAGE_SIZE = [224, 224]
train_path = 'G:/Machine Learning/Dataset/Train'
test_path = 'G:/Machine Learning/Dataset/Test'
# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING LAYER - IMAGENET WEIGHTS
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
vgg.summary()
# ----------------------------------------------------------------------------------------------------------------------
# DONT TRAIN EXISTING WEIGHTS
for layer in vgg.layers:
    layer.trainable = False
# ----------------------------------------------------------------------------------------------------------------------
# GETTING CLASSES
folders = glob('G:/Machine Learning/Dataset/Train/*')  # LOAD IN FUTURE THE IMAGES REQUIRED (BODY PARTS)
# ----------------------------------------------------------------------------------------------------------------------
# LAYER 1 - FLATTENING THE OUTPUT OF VGG
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# ----------------------------------------------------------------------------------------------------------------------
# MODEL - TRANSFER LEARNING
model = Model(inputs=vgg.input, outputs=prediction)
# MODEL STRUCTURE
model.summary()

# COST AND OPTIMISATION METHOD
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING IMAGES FROM DATASET
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# ----------------------------------------------------------------------------------------------------------------------
training_set = train_datagen.flow_from_directory('G:/Machine Learning/Dataset/Train', target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('G:/Machine Learning/Dataset/Test', target_size=(224, 224), batch_size=32,
                                            class_mode='categorical')
# ----------------------------------------------------------------------------------------------------------------------
# # MODEL - FROM SCRATCH
# model = Sequential()
# # ADDING A LAYER
#
# model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(224, 224, 3)))
#
# # POOLING
# model.add(MaxPooling2D(pool_size=2))
#
# # LAYERS 2 AND 3
# model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
# model.add(MaxPooling2D(pool_size=2)
#
# # FLATTENING
# model.add(Flatten())
#
# # FULLY CONNECTED LAYER
# model.add(Dense(500,activation="relu"))
# # OUTPUT LAYER
# model.add(Dense(2,activation="softmax"))
#
# # COMPILING CNN MODEL
# model.compile(loss=’categorical_crossentropy’,optimizer=’adam’,metrics=[‘accuracy’])
#
# # FITTING MODEL ON THE TRAINING SET
# model.fit_generator(training_set,validation_data=test_set,epochs=50, steps_per_epoch=len(training_set), validation_steps=len(test_set) )
# ----------------------------------------------------------------------------------------------------------------------
# FITTING THE MODEL
r = model.fit(training_set, validation_data=test_set, epochs=2, steps_per_epoch=len(training_set),
              validation_steps=len(test_set))
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING LOSS GRAPH
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# ----------------------------------------------------------------------------------------------------------------------
# PLOTTING ACCURACY GRAPH
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
# ----------------------------------------------------------------------------------------------------------------------
model.save('G:\Machine Learning\model_vgg2.h5')
# ----------------------------------------------------------------------------------------------------------------------
