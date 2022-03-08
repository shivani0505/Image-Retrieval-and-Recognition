#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import constants
import numpy as np
from scipy.stats import skew
from PIL import Image
from skimage import feature

class Q_Img_FM:
    
    def __init__(self):
        return
    
    def compute_image_color_moments(self, image_name):
        images_CM = []
        image_np_arr = Image.open(image_name)
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
        return np.array(images_CM)

    def compute_image_ELBP(self, image_name):
        ELBP = []
        image_np_arr = Image.open(image_name)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        i_min = np.min(image_np_arr)
        i_max = np.max(image_np_arr)
        if ( i_max - i_min != 0 ):
            image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
            
        lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
        ELBP.append(lbp.flatten())
        return np.array(ELBP)
    
    def compute_image_hog(self, image_name):
        HOG = []
        image_np_arr = Image.open(image_name)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        fd = np.array(fd)
        HOG.append(fd.flatten())
        return np.array(HOG)



