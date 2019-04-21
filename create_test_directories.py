# -*- coding: utf-8 -*-
"""Script to Create Test Folders for Kaggle Malaria Images

Download the directory cell_images from
    https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
and unzip locally and move all files into a directory called 'train'. 
This script will take the images and create new folders
for a test set of specified size.  This is useful for using an ImageDataGenerator.

Created on Fri Apr 19 19:15:36 2019

@author: Ruben
"""
import os
from random import sample

# Directory where the directory train is located
PATH = "cell_images"

infected_files = [f for f in os.listdir(os.path.join(PATH,"train","Parasitized")) if os.path.isfile(os.path.join(PATH,"train","Parasitized", f))]
uninfected_files = [f for f in os.listdir(os.path.join(PATH,"train","Uninfected")) if os.path.isfile(os.path.join(PATH,"train","Uninfected", f))]

# Create folder for test images
os.makedirs(os.path.join(PATH,"test","Parasitized"))
os.makedirs(os.path.join(PATH,"test","Uninfected"))

# Randomly select files for the test set
TEST_SIZE = 0.2

infected_indices = sample(list(range(1,len(infected_files))),int(TEST_SIZE*len(infected_files)))
uninfected_indices = sample(list(range(1,len(uninfected_files))),int(TEST_SIZE*len(uninfected_files)))

for i, file in enumerate(infected_files):
    if(i in infected_indices):
        os.rename(os.path.join(PATH,"train", "Parasitized",file), os.path.join(PATH,"test", "Parasitized",file))
        
for i, file in enumerate(uninfected_files):
    if(i in uninfected_indices):
        os.rename(os.path.join(PATH,"train", "Uninfected",file), os.path.join(PATH,"test", "Uninfected",file))