#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:41:15 2021

@author: khushalmodi
"""
import os
import constants
import numpy as np
from scipy.stats import skew
from PIL import Image
from skimage import feature

class ColorMoments:
    
    def __init__(self, data_path):
        self.data_path = data_path
        return
    
    def compute_image_color_moments(self, image_ids):
        images_CM = []
        for image_id in image_ids:
            features =  []
            image_np_arr = Image.open(os.path.join(self.data_path, image_id))
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            M = 8
            N = 8
            tiles = [image_np_arr[x:x+M,y:y+N] for x in range(0,image_np_arr.shape[0],M) for y in range(0,image_np_arr.shape[1],N)]
            nptiles = np.array(tiles)
            CM_mean = nptiles.mean(axis=(1,2))
            CM_SD = nptiles.std(axis=(1,2))
            CM_skew = [skew(tile.flatten()) for tile in nptiles]
            images_CM.append(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()]))
        ##TODO how to combine the mean, std and skew to make a feature vector
        return np.array(images_CM)

    def compute_image_ELBP(self, image_ids):
        ELBP = []
        for image_id in image_ids:
            image_np_arr = Image.open(os.path.join(self.data_path, image_id))
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            i_min = np.min(image_np_arr)
            i_max = np.max(image_np_arr)
            if ( i_max - i_min != 0 ):
                image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
            
            lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
            ELBP.append(lbp.flatten())
        return np.array(ELBP)
    
    def compute_image_hog(self, image_ids):
        HOG = []
        for image_id in image_ids:
            image_np_arr = Image.open(os.path.join(self.data_path, image_id))
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
            fd = np.array(fd)
            HOG.append(fd.flatten())
        return np.array(HOG)
    
    def compute_single_image_color_moments(self, image_path):
        images_CM = []
        image_np_arr = Image.open(image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        M = 8
        N = 8
        tiles = [image_np_arr[x:x+M,y:y+N] for x in range(0,image_np_arr.shape[0],M) for y in range(0,image_np_arr.shape[1],N)]
        nptiles = np.array(tiles)
        CM_mean = nptiles.mean(axis=(1,2))
        CM_SD = nptiles.std(axis=(1,2))
        CM_skew = [skew(tile.flatten()) for tile in nptiles]
        ##TODO how to combine the mean, std and skew to make a feature vector
        images_CM.append(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()]))
        return np.array(images_CM)

    def compute_single_image_ELBP(self, image_path):
        ELBP = []
        image_np_arr = Image.open(image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        i_min = np.min(image_np_arr)
        i_max = np.max(image_np_arr)
        if ( i_max - i_min != 0 ):
            image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
        
        lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
        return lbp.flatten()
    
    def compute_single_image_hog(self, image_path):
        HOG = []
        image_np_arr = Image.open(image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        fd = np.array(fd)
        return fd.flatten()
    
    def compute_single_image_color_moments_from_image(self, image_vec):
        images_CM = []
        image_np_arr = np.array(image_vec)
        M = 8
        N = 8
        tiles = [image_np_arr[x:x+M,y:y+N] for x in range(0,image_np_arr.shape[0],M) for y in range(0,image_np_arr.shape[1],N)]
        nptiles = np.array(tiles)
        CM_mean = nptiles.mean(axis=(1,2))
        CM_SD = nptiles.std(axis=(1,2))
        CM_skew = [skew(tile.flatten()) for tile in nptiles]
        ##TODO how to combine the mean, std and skew to make a feature vector
        images_CM.append(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()]))
        return np.array(images_CM)

    def compute_single_image_ELBP_from_image(self, image_vec):
        image_np_arr = np.array(image_vec)
        i_min = np.min(image_np_arr)
        i_max = np.max(image_np_arr)
        if ( i_max - i_min != 0 ):
            image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
        
        lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
        return lbp.flatten()
    
    def compute_single_image_hog_from_image(self, image_vec):
        image_np_arr = np.array(image_vec)
        fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        fd = np.array(fd)
        return fd.flatten()



