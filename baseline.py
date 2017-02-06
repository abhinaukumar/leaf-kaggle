# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 00:34:23 2017

@author: nownow
"""

import keras
from convnetskeras.convnets import preprocess_image_batch, convnet
from keras.utils.visualize_util import plot
from keras.layers.core import Activation, Dense

#Third party bug fix. Use only if required.
def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

model = convnet('vgg_16', weights_path="/home/nownow/Documents/projects/pretrained_models/vgg16_weights.h5", heatmap=False)

#Skip modifications for primary testing with orignal VGG 16. Use 99 class SVM from 4096 vectors

model.layers.pop()
model.layers.pop()
model.add(Dense(99))
model.add(Activation("softmax"))

#All modified
