#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:42:13 2020

@author: tapanpatro
"""

from __future__ import absolute_import, division, print_function, unicode_literals
# from google_images_download import google_images_download

# try:
#   # The %tensorflow_version magic only works in colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


print("Current tensorflow version {}".format(tf.__version__))


import os
from tqdm import tqdm
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense,AveragePooling2D,Conv2D,Input,Flatten,Activation,Dropout,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from time import time
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
# import cv2

LOAD_PRETRAINED_SAVED_MODEL = False

if tf.test.is_gpu_available():
    LOAD_PRETRAINED_SAVED_MODEL = True
else:
    LOAD_PRETRAINED_SAVED_MODEL = False

class FeatureExtractModel:
    
    
    def __init__(self, model_given="MobileNetV2"):
        self.model_name = model_given
        self.IMG_SHAPE = 512
        
    
    def __select_model(self):
        """
    
        Returns
        -------
        base_model : Load the given model from tensorflow keras.

        """
        
        if self.model_name == "MobileNetV2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.IMG_SHAPE, 
                include_top=False, 
                weights='imagenet')
        else:
            return None
            
        self.__define_working_layers(base_model, 20)
        
        return base_model
    
    
    
    
    def __define_working_layers(self, base_model, fine_tune_at_layer):
        """
        

        Parameters
        ----------
        base_model : Working on Traning Layers.
        fine_tune_at_layer :  FINE tuning Layers.

        Returns
        -------
        None.

        """
        
        base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))
        
        if fine_tune_at_layer > len(base_model.layers) :
            return
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at_layer]:
          layer.trainable =  False
          
          
          

    def build_compiled_model(self, print_summary=False, build_new=False):
        """

        Parameters
        ----------
        print_summary : Prinint Summary and parameters of the model. The default is False.
        build_new : Checking if we need to create a model or we can load existing model. The default is False.

        Returns
        -------
        model : Returns the model.

        """
        
        
        
        
        if build_new:
            
            if not LOAD_PRETRAINED_SAVED_MODEL:
                print("As this system does not support GPU, Please use trained Model.")
                return
            
            base_model_with_tuned = self.__select_model()
    
            model =  tf.keras.Sequential([
                  base_model_with_tuned,
                  tf.keras.layers.Conv2D(64, 3, activation='relu'),
                  tf.keras.layers.Dropout(0.2),
                  tf.keras.layers.GlobalAveragePooling2D(),
                #   tf.keras.layers.Dense(125, activation='softmax')
                ])
            
            model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
            
            if print_summary:
                model.summary()
                
                
            return model
        else:
            if os.path.exists("MobileNetV2_model_20l.h5"):
                model = load_model("MobileNetV2_model_20l.h5")
                return model
            else:
                print("Model not found")
                return None
            
            
            
    
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        