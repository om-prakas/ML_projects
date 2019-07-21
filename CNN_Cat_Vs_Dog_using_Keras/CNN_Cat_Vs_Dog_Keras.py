#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:06:53 2019

@author: omprakash
"""

from keras.models import Sequential   # initialise neural network
from keras.layers import Convolution2D  # convolutional layer add
from keras.layers import MaxPool2D   
from keras.layers import Flatten
from keras.layers import Dense    # to add fully connected layer in a cnn 

se = Sequential()

# Convolution step - 32 filter with dimension 3*3, 
se.add(Convolution2D(32,3,3,input_shape =(64,64,3),activation = 'relu'))
#pooling
se.add(MaxPool2D(pool_size= (2,2)))

#u can add more convilution layer or fully connected layer to increase accuracy
se.add(Convolution2D(32,3,3,activation = 'relu'))
se.add(MaxPool2D(pool_size= (2,2)))


#flattening
se.add(Flatten())

#Fully connected
se.add(Dense(output_dim =128,activation = 'relu'))
#when outcome more than 2 category then use softmax activation function
se.add(Dense(1,activation = 'sigmoid'))

#compiling the CNN
#more 2 outcome use categorical_crossentropy
se.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])

#fitting CNN to image 
#do image augmentation 
from keras.preprocessing.image import ImageDataGenerator

#resclae pixel value from 255 to 0-1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#know ur code loction
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

#go upto privious folder 
training_set = train_datagen.flow_from_directory('dataset/training_set/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
from PIL import Image
se.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)


