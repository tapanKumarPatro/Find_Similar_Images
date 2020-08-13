#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:44:33 2020

@author: tapanpatro
"""


import os
from tqdm import tqdm
from time import time
# import cv2
from PIL import Image
from . import building_model 
import matplotlib.pyplot as plt
import numpy as np
from . import Util

class ImageProcessing:
    
    def __init__(self, path="./dataset"):
        super().__init__(path)
        self.IMAGE_SIZE = 512
        self.all_images_names = None
        self.dataset_path = path
        
    def show_files_dataset(self):
        """

        Returns
        -------
        returning the file list.

        """
        os.chdir(self.dataset_path)
        self.all_images_names = os.listdir()
        return self.all_images_names
    
    def show_images(self, count):
        """
        

        Parameters
        ----------
        count : Showing Some images.

        Returns
        -------
        None.

        """
        for i, val in enumerate(self.all_images_names[:count]):
            plt.subplot(1, count, i+1)
            image_data = Image.open(val)
            plt.imshow(image_data)
            plt.show()
        
    
    def cal_feature(self, image_data):
        """
        

        Parameters
        ----------
        image_data : Feature calculating for the given image.

        Returns
        -------
        feature : A vector what represents the images data.

        """
        image = Image.open(image_data)
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = building_model.FeatureExtractModel.build_compiled_model().predict(image)
        return feature


    def extract_feature_and_load_pickel(self):
        """

        Returns
        -------
        None.

        """
        precompute_features = []

        for image_name in tqdm(self.all_images_names):
            name = image_name
            features = self.cal_feature(image_name)
            precompute_features.append({"name": name, "features": features})
            
            
        if not self.check_if_pickel_exit():
            Util.pickle_stuff("precompute_img_features.pickle", precompute_features)
        else:
            print("File already Present")
        

    def check_if_pickel_exit(self):
        
        PATH = "precompute_img_features.pickle"
        
        if os.path.isfile(PATH):
            return True
        else:
            return False
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        