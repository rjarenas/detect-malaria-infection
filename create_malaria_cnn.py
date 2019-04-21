# -*- coding: utf-8 -*-
"""Malaria CNN Classifier 

This script creates a CNN to classify whether or not a cell is infected
by malaria.  The script depends on image data available from Kaggle:
    https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

The resulting CNN is explored by considering the effect of the CNN's layers
on various images.

Created on Fri Apr 19 16:20:42 2019

@author: Ruben
"""

# Imports
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import math

# Custom module for visualization
from plot_from_generator import plot_image_from_gen, plot_activation_layers_from_gen

# Resize all images to SIZE by SIZE
SIZE = 100

# Training data will flow from a directory and be scaled, rotated, flipped, etc.
# to augment the existing data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range = 30, width_shift_range = 0.25,
                                   height_shift_range = 0.25, shear_range = 0.25, zoom_range = 0.25,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Directories where train and test images are located
# The script create_test_directories.py can be used to move files 
# into a test directory after download from Kaggle
TRAIN_DATA_DIR = 'cell_images/train'
TEST_DATA_DIR = 'cell_images/test'

# Create the iterators that flow data from the above directories
train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                    target_size = (SIZE,SIZE),
                                                    batch_size = 50,
                                                    class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(TEST_DATA_DIR,
                                                    target_size = (SIZE,SIZE),
                                                    batch_size = 50,
                                                    class_mode = 'binary')

# Plot one of the test images from an iterator
plot_image_from_gen(test_generator)

# Define CNN architecture       
model = Sequential()
model.add(Convolution2D(40, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))
model.add(Convolution2D(40, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(40, (3, 3), activation = 'relu'))
model.add(Convolution2D(40, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(40, (3, 3), activation = 'relu'))
model.add(Convolution2D(40, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

# Fit the CNN
epochs = 60
steps_per_epoch = math.ceil(train_generator.n / train_generator.batch_size)
validation_steps = math.ceil(test_generator.n / test_generator.batch_size)

history = model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, 
                              epochs = epochs, validation_data = test_generator,
                              validation_steps = validation_steps)

# Evaluate model against the test_generator
model.evaluate_generator(test_generator,steps = validation_steps)

# Plot the activation layers for one of the test images
plot_activation_layers_from_gen(test_generator, model, 9)
