# -*- coding: utf-8 -*-
""" Functions for Plotting Images Related to a Keras Image Iterator

Created on Sat Apr 20 16:24:58 2019

@author: Ruben
"""
import numpy as np
from keras.models import Model 
import matplotlib.pyplot as plt

def plot_image_from_gen(gen):
    """Plots the next image from a Keras image iterator such as DirectoryIterator

    Parameters
    ----------
    gen : keras.preprocessing.Iterator
        An image iterator from which to pull images
    
    Returns
    -------
    None
    """
    plt.imshow(gen.next()[0][0,:,:])
    plt.show()
    
def plot_activation_layers_from_gen(gen, model, layer_count, images_per_row = 16):
    """Plots the activation layers associated with the next image from a Keras image iterator 
    such as DirectoryIterator.

    Parameters
    ----------
    gen : keras.preprocessing.Iterator
        An image iterator from which to pull images
    model : keras.models.Model
        The network model derived from a Model class such as Sequential
    layer_count : int
        The first layer_count layers to be plotted
    images_per_row : int
        The number of filters to plot per row
        
    Returns
    -------
    None
    """
    gen_image = gen.next()[0]
    
    plt.imshow(gen_image[0,:,:])
    
    layer_outputs = [layer.output for layer in model.layers[:layer_count]]
    activation_model = Model(inputs = model.input, output = layer_outputs)
    activations = activation_model.predict(gen_image)
    
    layer_names = []
    for layer in model.layers[:layer_count]:
        layer_names.append(layer.name)
        
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        
        size = layer_activation.shape[1]
        
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, : , : , col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1)*size, row*size : (row + 1) * size] = channel_image
        
        scale = 1.0 / size
        plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
