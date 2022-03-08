#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:13:40 2021

@author: khushalmodi
"""
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class LDA:
    def get_LDA_components(self, feature_matrix, k):
        feature_matrix = np.array(feature_matrix)
        feature_matrix = feature_matrix + 1
        feature_matrix = feature_matrix/feature_matrix.sum(axis=1, keepdims=True)
        lda = LatentDirichletAllocation(n_components = int(k), learning_method='online', max_iter = 20, random_state = 42)
        lda_f = lda.fit(feature_matrix)
        lda_weights = lda_f.transform(feature_matrix)
        return lda_weights, lda_f.components_
