#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:38:30 2021

@author: khushalmodi
"""

import constants
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import task6
from Top_K_Images import Top_K_Img
import personalizePageRank

class PPR_C:
    def __init__(self, X, Y, images, X_test, feature_model, images_folder, binary_class=False):
        #globals
        self.binary_class = binary_class
        self.regularization_strength = 10000
        self.learning_rate = 0.0001
        self.max_epochs = 20000
        
        #generate a classifier for each type value of image
        self.W = {}
        self.n_classes = []
        self.class_dict = {}
        self.labels = self.create_multi_label_classifier(X, Y, images, X_test, feature_model, images_folder)
        return
    
    def get_labels(self):
        return self.labels
    
    def create_multi_label_classifier(self, X, Y, images, X_test, feature_model, images_folder):
        test_images = np.array(images)
        labels = []
        
        self.n_classes = np.unique(Y)
        X_features = pd.DataFrame(X)
        X_normalized = MinMaxScaler().fit_transform(X_features.values)
        X_features = pd.DataFrame(X_normalized)
        X_features['image_names'] = test_images

        print("splitting dataset into train and test sets...")
        X_type_train, X_type_test, Y_type_train, Y_type_test = tts(X_features, Y, test_size=0.2, random_state=42)
        X_type_train.reset_index(drop=True, inplace=True)
        X_type_test.reset_index(drop=True, inplace=True)
        
        X_train_image_names = X_type_train['image_names'].to_numpy()
        X_test_image_names = X_type_test['image_names']
        X_type_train.drop(['image_names'], axis = 1, inplace = True)
        X_type_test.drop(['image_names'], axis = 1, inplace = True)
        a = X_type_train.to_numpy()
        
        for i in range(0, len(self.n_classes)):
            self.class_dict[i] = self.n_classes[i]
        
        print("Setting up the classifier...")
        
        top_subject_ids = []
        for i in range(0, X_test.shape[0]):
            Task5 = Top_K_Img(images_folder, None)
            closer_images = Task5.compute_score(X_test[i], X_type_train.to_numpy(), test_images, feature_model, min(100, len(images)))
            closer_images_x = np.argwhere(np.isin(test_images, closer_images)).ravel()
            X_type_train_temp = X_type_train.to_numpy()
            X_type_train_temp = X_type_train_temp[list(closer_images_x)]
            images_type_train_temp = test_images[closer_images_x]
            
            X_type_train_temp = np.vstack((X_type_train_temp, X_test[i]))
            
            sim_matrix = np.matmul(X_type_train_temp, X_type_train_temp.T)
            ppr_p2 = personalizePageRank.ppr()
            similarityGraph = ppr_p2.convertToGraph(sim_matrix,5,3)
            result = ppr_p2.findPersonalizedRank(X_type_train_temp.shape[0],
                                                 X_type_train_temp.shape[0], 
                                                 X_type_train_temp.shape[0], 
                                                 similarityGraph, 
                                                 0.75)
            idx = np.argsort(result)[::-1]
            if ( idx[0] == X_type_train_temp.shape[0]-1 ):
                # print("blah")
                idx[0] = idx[1]
            label = images_type_train_temp[idx[0]]
            labels.append(label)
        return labels
    
    def predict_class(self, X_test):
        return self.predict_multi_label_class(X_test)
    
    def predict_multi_label_class(self, X_test, internal=False):
        labels = []
        X_features = pd.DataFrame(X_test)
        if ( internal == False ):
            print("Inside internal false")
            X_normalized = MinMaxScaler().fit_transform(X_features.values)
            X_features = pd.DataFrame(X_normalized)
            X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        X_test_local = X_features.to_numpy()
        print("shape: ", X_test_local.shape)
        for i in range(0, X_test_local.shape[0]):
            yp = []
            for j in range(0, len(self.n_classes)):
                yyp = np.dot(X_test_local[i], self.W[j])
                yp.append(yyp)
            max_ind = np.argmax(np.array(yp))
            labels.append(self.class_dict[max_ind])
        return np.array(labels).reshape(X_test.shape[0], 1)
    
    def compute_cost(self, W, X, Y):
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)
        
        cost = (1/2)*np.dot(W, W) + hinge_loss
        return cost
    
    def compute_cost_gradient(self, W, X, Y):
        if ( type(Y) == np.float64):
            Y = np.array([Y])
            X = np.array([X])
        
        distance = 1 - (Y * np.dot(X, W))
        dw = np.zeros(len(W))
        
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength * Y[ind] * X[ind])
            dw += di
        
        dw = dw/len(Y)
        return dw
    
    def sgd(self, features, outputs):
        weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01
        
        for epoch in range(1, self.max_epochs):
            X, Y = shuffle(features, outputs)
            X = np.array(X)
            Y = np.array(Y)
            
            counter = 0
            for each_row in X:
                ascent = self.compute_cost_gradient(weights, each_row, Y[counter])
                weights = weights - (self.learning_rate * ascent)
                counter+=1
            
            if ((epoch == 2**nth) or (epoch == self.max_epochs - 1)):
                cost = self.compute_cost(weights, features, outputs)
                print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                """if (abs(prev_cost - cost) < cost_threshold*prev_cost):
                    return weights"""
                prev_cost = cost
                nth +=1
        return weights
    


