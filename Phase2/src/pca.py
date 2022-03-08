#!/usr/bin/env python3
"""
Created on Tue Oct  5 16:57:19 2021
@author: rchityal
"""
import numpy as np
from eigen_decomposition import EigenDecomposition
from sklearn.decomposition import PCA as pca_lib

class PCA:
    def get_reduced_dimensions(self, features, principalComponents):
        return np.matmul(features, principalComponents)
    
    
    def get_PCA_library(self, features, K):
        pca = pca_lib(n_components=K)
        principalComponents = pca.fit_transform(features)
        return principalComponents
    
    
    def get_PCA(self, features, K_principal_components):
        
        # find covariance matrix
        #print (features.shape)
        mean_vec = np.mean(features, axis=0)
        
        covariance_matrix = (features - mean_vec).T.dot((features - mean_vec)) / (features.shape[0]-1)
        #print(covariance_matrix.shape)
        
        # find eigen values and vectors for the covariance matrix
        decompose = EigenDecomposition()
        eigen_vectors, eigen_values = decompose.get_eigen_decomposition_top_k(covariance_matrix, K_principal_components)
        
        return np.dot(eigen_vectors.T, features.T).T, (np.diag(eigen_values)), eigen_vectors.T
    
    
# =============================================================================
#     eigen_vectors = eigen_vectors.real
#     
#     # find inverse of the eigen vectors matrix
#     eigen_vectors_inverse = np.linalg.inv(eigen_vectors)
#     
#     print(eigen_vectors.shape)
#     
#     # Make a list with eigen values and eigen vectors to sort the
#     # list based on eigen values    
#     eigen_value_vectors = []
# 
#     for indx, eigen in enumerate(eigen_values):
#         eigen_value_vectors.append([np.abs(eigen), eigen_vectors[:, indx], eigen_vectors[indx, :]])
#     
#     # Sort the above eigen value-vector pair based on eigen value (descending order)
#     print(eigen_value_vectors[0][1].shape)
#     eigen_value_vectors.sort(key = lambda val: val[0], reverse = True)
# 
#     # Get the sorted eigen vectors into a separate list and store them
#     # numpy arrays
#     
#     eigen_vectors_list = []
#     eigen_values = []
#     eigen_vectors_inverse_list = []
#     for value in eigen_value_vectors:
#         eigen_values.append(value[0])
#         eigen_vectors_list.append(value[1])
#         eigen_vectors_inverse_list.append(value[2])
# 
#     eigen_vectors = np.asarray(eigen_vectors_list)
#     eigen_vectors_inverse = np.asarray(eigen_vectors_inverse_list)
#     
#     
#     # eigen_vectors = np.transpose(eigen_vectors)
#     print(eigen_vectors.shape)
#     
#     eigen_vectors = eigen_vectors[:, : -(eigen_vectors.shape[0] - K)]
#     eigen_vectors_inverse = eigen_vectors_inverse[: -(eigen_vectors.shape[0] - K), :]
#     
#     
#     print(eigen_vectors.shape)
#     
#     # PCA dimensions : n * k.. Data matrix = m * n 
#     # D = U V M. (m * n) ( n * k ) = m * n
#     
#     reduced_features = get_reduced_dimensions(features, eigen_vectors)
#     
#     
#     return  eigen_vectors, eigen_values,  eigen_vectors_inverse, reduced_features
# =============================================================================