# -*- coding: utf-8 -*-
"""Partaq4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/167Io-gfMOU8pbiR6htvn97SIjETxOJJU
"""

!pip install tensorflow-gpu
!nvidia-smi

pip install -q pyyaml h5py  # Required to save models in HDF5 format

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, optimizers, activations
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

#to test on small dataset
from google.colab import drive
drive.mount('/content/drive')
import os
# from pathlib import Path
# sum(p.stat().st_size for p in Path("inaturalist_12K").rglob('*'))
!ls

os.chdir('drive/My Drive/dl_cs6910/Assignment2')
!ls

# cnn_model = tf.keras.models.load_model('best_model')
# Recreate the exact same model, including its weights and the optimizer
cnn_model = tf.keras.models.load_model('model-best.h5')
# Show the model architecture
cnn_model.summary()



def generate_data(directory, data_desc, data_aug, batch_size, image_shape):
    if(data_desc == "test"):
        data = ImageDataGenerator(rescale=1./255)
        data = data.flow_from_directory(
            directory,
            target_size=image_shape,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        return data
    else:
        if data_aug:
            data = ImageDataGenerator(featurewise_center = True,
                                      brightness_range=None,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2, 
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      rotation_range=30,
                                      fill_mode='reflect',
                                      rescale=1./255,
                                      validation_split=0.1
            )
        else:
            data = ImageDataGenerator(rescale=1./255, validation_split=0.1)
            
        train_data = data.flow_from_directory(
            directory,
            target_size=image_shape,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset="training"
        )
        
        val_data = data.flow_from_directory(
            directory,
            target_size=image_shape,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
            subset="validation"
        )
        return train_data, val_data

#from give dataset link
# !curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > nature_12K.zip
# !unzip nature_12K.zip
# !ls

data_aug = [True]
batch_size = 64
image_shape = (227,227)
test_data_dir = 'inaturalist_12K/val/'
test = generate_data(test_data_dir, "test", data_aug[0], batch_size, image_shape)

test_loss, test_acc = cnn_model.evaluate(test, verbose=2)

# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/images.ipynb#scrollTo=crs7ZjEp60Ot
# import pathlib
# import PIL
# import PIL.Image
# !ls {"inaturalist_12K/val/Fungi"}
# train_data_dir = pathlib.Path("inaturalist_12K/train/")
# test_data_dir = pathlib.Path("inaturalist_12K/val")
# data_dir = pathlib.Path("inaturalist_12K/val/")
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
# fungi = list(data_dir.glob('Fungi/*'))
# PIL.Image.open(str(fungi[0]))

# Inside my model training code
!pip install wandb
import wandb

wandb.init(project='partA_Q4', entity='shreekanti')

# sample image and prediction from the test dataset
class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
plt.figure(figsize=(20, 60))
wandb_img = []
wandb_title = []
for i in range(10*3):
    ax = plt.subplot(10, 3, i + 1)
    image, label = test.next()
    predicted_label = cnn_model.predict(image)
    title_ = str("Actual Label : ") + class_names[np.where(label[0,:] == 1)[0][0]] + "\n" + str("Predicted Label : ") + class_names[np.where(predicted_label == np.amax(predicted_label))[1][0]]
    wandb_img.append(image[0])
    wandb_title.append(title_)
    plt.imshow(image[0])
    plt.title(title_)
for j in range(len(wandb_img)):
  wandb.log({"sample image and prediction from the test dataset": [wandb.Image(wandb_img[j], caption=wandb_title[j])]})



filter_ = Model(inputs= cnn_model.inputs, outputs=cnn_model.layers[0].output)
image, label = test.next()
plt.imshow(image[0])
title_ = str("Actual Label : ") + class_names[np.where(label[0,:] == 1)[0][0]]
plt.title(title_)
plt.show()
wandb.log({"Input image for visualization": [wandb.Image(image[0], caption=title_)]})

for i in range(len(cnn_model.layers)):
    layer = cnn_model.layers[i]
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    print(i, layer.name, filters.shape, layer.input.shape, layer.output.shape)

first_layer = cnn_model.layers[0]
filters, biases = first_layer.get_weights()
no_of_filters = filters.shape[-1]
print(no_of_filters)
feature_maps = filter_.predict(image)
plt.figure(figsize=(20,20))
row = int(no_of_filters/8)
wandb_img = []
wandb_title = [] 
for i in range(no_of_filters):
    ax = plt.subplot(row, 8, i + 1)
    wandb_img.append(feature_maps[0,:,:,i])
    wandb_title.append("filter"+str(i))
    plt.imshow(feature_maps[0,:,:,i])
    plt.axis("off")

for j in range(len(wandb_img)):
  wandb.log({"Filter visualization": [wandb.Image(wandb_img[j], caption=wandb_title[j])]})

