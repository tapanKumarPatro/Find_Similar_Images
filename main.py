#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:27:54 2020

@author: tapanpatro
"""

# from Avantari_model import building_model, cluster_images,load_process_images,matcher,Util


from Avantari_model.building_model import FeatureExtractModel 
from Avantari_model.cluster_images import Cluster_images
from Avantari_model.load_process_images import ImageProcessing 
from Avantari_model.matcher import Find_Similar_images
from Avantari_model.Util import *

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

model = None


# First We need to Build Model
ava_model = FeatureExtractModel(model_given="MobileNetV2")
model_builder = ava_model.build_compiled_model(print_summary=True)

matcher = Find_Similar_images()
cluster = Cluster_images()

def main(image_path, count):
    
    if not LOAD_PRETRAINED_SAVED_MODEL:
    image_list = matcher.return_similar_images(image_path, 3)
    
    return image_list
    
def cluster_image():
    # Cluster images and save into folder
    cluster.cluster_kmean_and_save_into_folder()
    
    # Create cluster images
    cluster.cluster_tsne_and_plot_cluster()
    


if int(str(tf.__version__).split(".")[0]) < 2:
    print("Please use tensorflow version 2.")
    print("")
    return
else:
    main()
    cluster_image()
    