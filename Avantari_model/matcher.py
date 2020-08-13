#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:06:52 2020

@author: tapanpatro
"""

from . import Util
# from building_model import FeatureExtractModel as feModel
from . import building_model 
# from load_process_images import ImageProcessing 
from . import load_process_images as processImage
import scipy as sp



class Find_Similar_images:
    
    def __init__(self, pickel_path="precompute_img_features.pickle"):
        super().__init__(pickel_path)
        self.IMAGE_SIZE = 512
        self.pickel_file_path = pickel_path
        self.precompute_features = None
        self.model = None
        
    def __load_required_files(self):
        self.precompute_features = Util.load_stuff(self.pickel_file_path)
        self.model = building_model.FeatureExtractModel.build_compiled_model()
    
    def return_similar_images(self, image_path, count=3):
        """

        Parameters
        ----------
        image_path : Path of the Image.
        count : No. of image we want to see,  The default is 3.

        Returns
        -------
        top_n_list : Retruns the count list of similar images.

        """
        given_image_feature = processImage.ImageProcessing.cal_feature(image_path)
        top_n_list = self.__return_top_img(feature=given_image_feature, count=count)
        return top_n_list
    
    
    def __return_top_img(self, feature, count):
        
        distances = []
        
        for each_image_data in self.precompute_features:
            image_feature = each_image_data.get("features")
            eucl_dist = sp.spatial.distances.euclidean(image_feature, feature)
            distances.append(eucl_dist)
    
        return self.__get_images(distances, count)
        
    def __get_images(self, distances, count):
        distances.sort()
        image_name_list = []
        
        for dis in distances[:count]:
            each_index = distances.index(dis)
            image_name = self.precompute_features[each_index].get("name")
            image_name_list.append(image_name)
            
        return image_name_list
        
        










        