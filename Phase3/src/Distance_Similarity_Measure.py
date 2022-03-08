#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import constants
import numpy as np
from scipy.stats import skew
from PIL import Image
from skimage import feature
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import chebyshev
from scipy.stats import wasserstein_distance

class Dis_Measure:
    
    def __init__(self):
        return
    
    def euclidean(self, q_latent, left_matrix):
    	eucli_dict = {}
    	for i in range(len(left_matrix)):
    		e_dis = np.linalg.norm(q_latent-left_matrix[i],ord = 2)
    		eucli_dict[i] = round(e_dis,4)
    	eucli_dict = dict(sorted(eucli_dict.items(), key=lambda item: item[1]))
    	return eucli_dict
    	
    def hamming(self, q_latent, left_matrix):
    	hamming_dict = {}
    	for i in range(len(left_matrix)):
    		ham_dic = hamming(q_latent,left_matrix[i])
    		hamming_dict[i] = ham_dic
    	hamming_dict = dict(sorted(hamming_dict.items(), key=lambda item: item[1]))
    	return hamming_dict
    	
    def manhattan(self, q_latent, left_matrix):
    	manhattan_dict = {}
    	for i in range(len(left_matrix)):
    		man_dic = cityblock(q_latent,left_matrix[i])
    		manhattan_dict[i] = man_dic
    	manhattan_dict = dict(sorted(manhattan_dict.items(), key=lambda item: item[1]))
    	return manhattan_dict
    	
    def chebyshev(self, q_latent, left_matrix):
    	chebyshev_dict = {}
    	for i in range(len(left_matrix)):
    		chebyshev_dic = chebyshev(q_latent,left_matrix[i])
    		chebyshev_dict[i] = chebyshev_dic
    	chebyshev_dict = dict(sorted(chebyshev_dict.items(), key=lambda item: item[1]))
    	return chebyshev_dict
    
    def earth_mover(self, q_latent, left_matrix):
    	earth_mover_dict = {}
    	q_latent = q_latent.flatten()
    	for i in range(len(left_matrix)):
    		feature = left_matrix[i]
    		feature = feature.flatten()
    		cost = wasserstein_distance(q_latent,feature)
    		earth_mover_dict[i] = cost
    	earth_mover_dict = dict(sorted(earth_mover_dict.items(), key=lambda item: item[1]))
    	return earth_mover_dict
    


