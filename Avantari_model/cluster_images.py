#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:56:06 2020

@author: tapanpatro
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as PathEffects
import shutil
# import Util
# from load_process_images import ImageProcessing
# Avantari_model import

from . import Util
from . import load_process_images as processImage



class Cluster_images:
    
    def __init__(self, pickel_path="precompute_img_features.pickle"):
        super().__init__(pickel_path)
        self.IMAGE_SIZE = 512
        self.pickel_file_path = pickel_path
        self.precompute_features = Util.load_stuff("precompute_img_features.pickle")
        
    def __collect_all_image_feature(self):
        # Collecting all features for Clustering
        all_features = []
        
        for each_image_data in self.precompute_features:
            image_feature = each_image_data.get("features")
            all_features.append(image_feature[0])
            
        return all_features
        
        
    def cluster_kmean_and_save_into_folder(self):
        kmeans = KMeans(n_clusters=10, random_state=0).fit(np.array(self.__collect_all_image_feature()))
        all_images = processImage.ImageProcessing("./dataset").show_files_dataset()
        
        # categorizing images into clusters and saving in dataset folder
        
        self.cluster_labels = kmeans.labels_
        
        print("\n")
        for i, m in enumerate(self.cluster_labels):
            print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
            shutil.copy(all_images[i], str(m) + "_" + str(i) + ".jpg")
            
    def cluster_tsne_and_plot_cluster(self):
        """
        Choosing T-SNE Clusters because It's neighbourhood embedding and cloud work with massive data.'

        Returns
        -------
        None.

        """
        
        
        # Fit the model using t-SNE randomized algorithm
        digits_proj = TSNE(random_state=25111993).fit_transform(self.__collect_all_image_feature())
        
        # Ploting the graph.
        print(list(range(0,18)))
        sns.palplot(np.array(sns.color_palette("hls", 18)))
        self.scatter(digits_proj, list(self.cluster_labels))
        plt.savefig('animal_cluster.png', dpi=120)

    def scatter(self, x, colors):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 18))
    
        # We create a scatter plot.
        f = plt.figure(figsize=(32, 32))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                        c=palette[colors.astype(np.int)])
        
        ax.axis('off')
        ax.axis('tight')
    
        # We add the labels for each cluster.
        txts = []
        for i in range(18):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=50)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    
        return f, ax, sc, txts
        
        
        
        
        
        
        
        