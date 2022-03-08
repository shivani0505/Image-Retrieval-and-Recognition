#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 04:31:09 2021

@author: khushalmodi
"""
import numpy as np
import constants
from eigen_decomposition import EigenDecomposition

class SVD:
    
    def __init__(self):
        return
    
    def get_svd_from_sklearn(self):
        return

    def get_svd_from_scratch(self, data_matrix, k):
        D = np.array(data_matrix)
        D_left =np.matmul(D, D.T)
        D_right = np.matmul(D.T, D)
        
        decompose = EigenDecomposition()
        left_eigen_components = decompose.get_eigen_decomposition_top_k(D_left, k)
        right_eigen_components = decompose.get_eigen_decomposition_top_k(D_right, k)
        
        U = left_eigen_components[0]
        sigma = np.diag(left_eigen_components[1])
        V = right_eigen_components[0]
        
        return (U, sigma, V.T)
