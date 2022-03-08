#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:03:50 2021

@author: khushalmodi
"""
import numpy as np

class EigenDecomposition:
    def get_eigen_decomposition_top_k(self, matrix, k):
        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real
        
        idx = np.argsort(eigen_values)[::-1]
        
        eigen_vectors = eigen_vectors[:,idx]
        
        eigen_values = eigen_values[idx]
        
        eigen_vectors = eigen_vectors[:, 0:k]
        
        eigen_values = eigen_values[:k]
        
        return (eigen_vectors, eigen_values)