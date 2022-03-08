#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:48:48 2021

@author: khushalmodi
"""
from datetime import datetime

#Dimension of the input image
image_dimension = (64,64)

#Mongo db collection for the images
image_collection = "images_db"

#path where images are stored
input_path = "/Users/khushalmodi/Desktop/ASU/Fall_2021/MWDB/Phase-2/Presentation/Multimedia-and-Web-Databases/all"

#path where outputs are stored
output_path = "/Users/khushalmodi/Desktop/ASU/Fall_2021/MWDB/Phase-2/Presentation/Multimedia-and-Web-Databases/outputs"

#available reduction techniques
list_reduction_techniques = ["PCA", "SVD", "LDA", "k-means"]

#available image types
list_image_types = ["cc", "con", "detail", "emboss", "jitter", "neg", "noise01",
                    "noise02", "original", "poster", "rot", "smooth", "stipple"]

#available feature models
list_feature_models = ["Color_Moments", "HOG", "ELBP"]

output_file_datetime = datetime.now().strftime("%m-%d-%y-%h-%m-%s")

#number of clustering iterations in k means
k_means_iter = int(100)

Y_range = 40

Y = "Subject"

X = "Type"

weight = "Weight"
"""
output file format will be task-k-model-reduction_tech-X/Y

for example:
    task 1, k = 2, reduction tech = PCA, model = HOG, X = cc
        will be stored as 3 files 
            1. task1-PCA-HOG-cc-2-rev.npy (task1 executed with PCA, HOG, type cc and storing the right eigen vector here)
            2. task1-PCA-HOG-cc-2-ll.npy (task1 executed with PCA, HOG, type cc and storing the latent semantic importance here)
            3. task1-PCA-HOG-cc-2-lev.npy (task1 executed with PCA, HOG, type cc and storing the left eigen vector here)

for example:
    task 2, k = 6, reduction tech = K-means, model = ELBP, Y = 30
        will be stored as 3 files 
            1. task2-k-means-ELBP-30-6-rev.npy (task2 executed with K-means, ELBP, subject 30, k=6 and storing the right eigen vectors here)
            2. task2-k-means-ELBP-30-6-lev.npy  (task2 executed with K-means, ELBP, subject 30, k=6 and storing the left eigen vectors here)
"""