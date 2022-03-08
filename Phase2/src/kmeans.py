#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:12:07 2021

@author: khushalmodi
"""
import numpy as np
import constants
import random

class KMeans:
    def get_k_clusters(self, feature_matrix, k):
        
        centroids = np.random.rand(k, feature_matrix.shape[1]) #k random centroid to start with
        labels = np.zeros((feature_matrix.shape[0], 1), dtype=np.int64)       #labels for each row of the feature matrix
        cluster_size = np.zeros((k,1), dtype=np.int64)                        #number of points in each cluster
        
        for i in range(0,k):
            rand_int = random.randint(0, feature_matrix.shape[0]-1)
            centroids[i] = feature_matrix[rand_int]
        
        for i in range(0, constants.k_means_iter):            # running iterations
            cluster_size = np.zeros((k,1), dtype=np.int64)    
            
            for j in range(0, feature_matrix.shape[0]):       #going over each data point
                euclidean = np.empty((k))                  #stored the distance of this point from all centroids
                
                for l in range(0, k):                         #going over all centroids
                    euclidean[l] = np.linalg.norm( feature_matrix[j] - centroids[l] )
                
                labels[j] = int(np.argmin(euclidean))              #update the label of the point with argmax
                cluster_size[int(labels[j])]+=1
            
            #print(cluster_size.reshape(1,k))
            #print(labels.reshape(1, feature_matrix.shape[0]))
            
            for l in range(0,k):
                if ( cluster_size[l] != 0 ):
                    centroids[l] = 0.0
            
            for j in range(0, feature_matrix.shape[0]):       #update the centroids for the new iteration
                centroids[int(labels[j])] += (feature_matrix[j]/cluster_size[int(labels[j])])
        
        new_feature_matrix = np.empty((feature_matrix.shape[0], k))
        for i in range(0, feature_matrix.shape[0]):
            euclidean = []
            for j in range (0, k):
                euclidean.append(np.linalg.norm(feature_matrix[i] - centroids[j]))
            new_feature_matrix[i] = np.array(euclidean)
        
        return new_feature_matrix, centroids