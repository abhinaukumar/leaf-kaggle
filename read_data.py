# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:56:19 2017

@author: nownow
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

sess = tf.InteractiveSession()

file_dir = "/home/nownow/Documents/projects/leaf/data/images/"
files = os.listdir(file_dir)
files = [file_dir+file for file in files]
m = len(files)
#filename_queue = tf.train.string_input_producer(files)

#reader = tf.WholeFileReader()
#key, value = reader.read(filename_queue)

images = [];

for f in files:
    #im = Image.open(f)
    #images.append(np.reshape(list(im.getdata()),im.size))
    #images.append(np.roll(cv2.imread(f),1,-1))
    images.append(cv2.imread(f,0))

#images = [255 - image for image in images]

print("Read images into matrix");

im = tf.placeholder(tf.uint8,[None,None,1])
im_reshaped = tf.image.resize_image_with_crop_or_pad(im,1024,1024);

images_reshaped = []

print("Set up graph for reshaping")

for image in images:
    image.shape = (image.shape[0],image.shape[1],1)
    images_reshaped.append(im_reshaped.eval(feed_dict = {im: image}))

print("Reshaped all images")

image_batch = tf.placeholder(tf.uint8,[None,1024,1024,1])
im_resized = tf.image.resize_images(image_batch,[256,256])

print("Set up graph for resizing to 256x256")

images_preprocessed = im_resized.eval(feed_dict = {image_batch: images_reshaped})

print("Preprocessed all images")
#mean normalization

#for image in images_reshaped:
#    image[0, :, :] -= 103.939
#    image[1, :, :] -= 116.779
#    image[2, :, :] -= 123.68
#    image = np.expand_dims(image, axis=0);

