#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:12:49 2021

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
from sklearn.metrics import confusion_matrix

class SVM:
    def __init__(self, X, Y, binary_class=False):
        #globals
        self.binary_class = binary_class
        self.regularization_strength = 1000
        self.learning_rate = 0.000001
        self.max_epochs = 10000
        
        #generate a classifier for each type value of image
        self.W = {}
        self.n_classes = []
        self.class_dict = {}
        
        if ( self.binary_class ):
            self.create_binary_classifier(X, Y)
        else:
            self.create_multi_label_classifier(X, Y)
        return
    
    def create_multi_label_classifier(self, X, Y):
        self.n_classes = np.unique(Y)
        X_features = pd.DataFrame(X)
        #self.remove_correlated_features(X_features)
        #self.remove_less_significant_features(X_features, Y_type)
        X_normalized = MinMaxScaler().fit_transform(X_features.values)
        X_features = pd.DataFrame(X_normalized)
        # insert 1 in every row for intercept b
        X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        print("splitting dataset into train and test sets...")
        X_type_train, X_type_test, Y_type_train, Y_type_test = tts(X_features, Y, test_size=0.2, random_state=42)
        
        for i in range(0, len(self.n_classes)):
            self.class_dict[i] = self.n_classes[i]
        
        print("Classifying data into ", self.n_classes, " classes")
        
        for i in range(0, len(self.n_classes)):            
            y_type_train = np.zeros(Y_type_train.shape[0])
            ind_type = np.where(Y_type_train == self.class_dict[i])
            ind_other_type = np.where(Y_type_train != self.class_dict[i])
            y_type_train[ind_type[0]] = 1.0
            y_type_train[ind_other_type[0]] = -1.0
            
            print("training model...")
            W_type = self.sgd(X_type_train, y_type_train)
            print("training finished for svd model for label ",self.class_dict[i])
            self.W[i] = W_type
            print("\n\n\n")
        
        labels = self.predict_multi_label_class(X_type_test, internal=True)
        confu_matrix = confusion_matrix(Y_type_test, labels)
        fp = confu_matrix.sum(axis=0) - np.diag(confu_matrix)  
        fn = confu_matrix.sum(axis=1) - np.diag(confu_matrix)
        tp = np.diag(confu_matrix)
        tn = confu_matrix.sum() - (fp + fn + tp)
        false_pos_rate = fp/(tn+fp+fn+tp)
        miss_rate = fn/(fn+tp)
        
        print("False positive rates ad miss rates for each class--> ")
        for i in range(0, len(self.n_classes)):
            print ("Class: ", self.n_classes[i], " fp rate: ", false_pos_rate[i], " miss rate: ", miss_rate[i])
        print("accuracy on test dataset: {}".format(accuracy_score(Y_type_test, labels)*100), " percentage")
        #print("recall on test dataset: {}".format(recall_score(Y_type_test, labels, pos_label='positive', average='micro')))
        #print("precision on test dataset: {}".format(precision_score(Y_type_test, labels, pos_label='positive', average='micro')))
        return
    
    def create_binary_classifier(self, X, Y):
        self.n_classes = np.unique(Y)
        X_features = pd.DataFrame(X)
        #self.remove_correlated_features(X_features)
        #self.remove_less_significant_features(X_features, Y_type)
        X_normalized = MinMaxScaler().fit_transform(X_features.values)
        X_features = pd.DataFrame(X_normalized)
        # insert 1 in every row for intercept b
        X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        print("splitting dataset into train and test sets...")
        X_type_train, X_type_test, Y_type_train, Y_type_test = tts(X_features, Y, test_size=0.2, random_state=42)
        
        for i in range(0, len(self.n_classes)):
            self.class_dict[i] = self.n_classes[i]
        
        print("Classifying data into ", self.n_classes, " classes")
        
        y_type_train = np.zeros(Y_type_train.shape[0])
        ind_type = np.where(Y_type_train == self.class_dict[0])
        ind_other_type = np.where(Y_type_train != self.class_dict[0])
        y_type_train[ind_type[0]] = 1.0
        y_type_train[ind_other_type[0]] = -1.0
        
        print("training model...")
        W_type = self.sgd(X_type_train, y_type_train)
        print("training finished for svd model for label ",self.class_dict[i])
        self.W[0] = W_type
        print("\n\n\n")
        
        labels = self.predict_multi_label_class(X_type_test, internal=True)
        confu_matrix = confusion_matrix(Y_type_test, labels)
        fp = confu_matrix.sum(axis=0) - np.diag(confu_matrix)  
        fn = confu_matrix.sum(axis=1) - np.diag(confu_matrix)
        tp = np.diag(confu_matrix)
        tn = confu_matrix.sum() - (fp + fn + tp)
        false_pos_rate = fp/(tn+fp+fn+tp)
        miss_rate = fn/(fn+tp)
        
        print("False positive rates ad miss rates for each class--> ")
        for i in range(0, len(self.n_classes)):
            print ("Class: ", self.n_classes[i], " fp rate: ", false_pos_rate[i], " miss rate: ", miss_rate[i])
        print("accuracy on test dataset: {}".format(accuracy_score(Y_type_test, labels)))
        #print("recall on test dataset: {}".format(recall_score(Y_type_test, labels, pos_label='positive', average='micro')))
        #print("precision on test dataset: {}".format(precision_score(Y_type_test, labels, pos_label='positive', average='micro')))
        return
    
    def predict_class(self, X_test):
        if ( self.binary_class ):
            return self.predict_binary_class(X_test)
        return self.predict_multi_label_class(X_test)
    
    def predict_multi_label_class(self, X_test, internal=False):
        labels = []
        X_features = pd.DataFrame(X_test)
        if ( internal == False ):
            X_normalized = MinMaxScaler().fit_transform(X_features.values)
            X_features = pd.DataFrame(X_normalized)
            X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        X_test_local = X_features.to_numpy()
        #print("shape: ", X_test_local.shape)
        for i in range(0, X_test_local.shape[0]):
            yp = []
            for j in range(0, len(self.n_classes)):
                yyp = np.dot(X_test_local[i], self.W[j])
                yp.append(yyp)
            max_ind = np.argmax(np.array(yp))
            labels.append(self.class_dict[max_ind])
        return np.array(labels).reshape(X_test.shape[0], 1)
    
    def predict_binary_class(self, X_test, internal=False):
        labels = []
        X_features = pd.DataFrame(X_test)
        X_normalized = MinMaxScaler().fit_transform(X_features.values)
        X_features = pd.DataFrame(X_normalized)
        X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        X_test_local = X_features.to_numpy()
        
        for i in range(X_test_local.shape[0]):
            yp = np.sign(np.dot(X_test_local[i], self.W[0]))
            if ( yp > 0 ):
                labels.append(self.class_dict[0])
            else:
                labels.append(self.class_dict[1])
        return np.array(labels).reshape(X_test.shape[0], 1)
    
    def remove_correlated_features(self, X):
        corr_threshold = 0.9
        corr = X.corr()
        drop_columns = np.full(corr.shape[0], False, dtype =  bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= corr_threshold:
                    drop_columns[j] = True
        
        columns_dropped = X.columns[drop_columns]
        print("Dropping columns ", columns_dropped)
        X.drop(columns_dropped, axis = 1, inplace = True)
        return columns_dropped
    
    
    def remove_less_significant_features(self, X, Y):
        sl = 0.05
        regression_ols = None
        columns_dropped = np.array([])
        for itr in range(0, len(X.columns)):
            regression_ols = sm.OLS(Y, X).fit()
            max_col = regression_ols.pvalues.idxmax()
            max_val = regression_ols.pvalues.max()
            if max_val > sl:
                X.drop(max_col, axis = 'columns', inplace = True)
                columns_dropped = np.append(columns_dropped, [max_col])
            else:
                break
        regression_ols.summary()
        print("Dropping columns ", columns_dropped)
        return columns_dropped
    
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
                if (abs(prev_cost - cost) < cost_threshold*prev_cost):
                    return weights
                prev_cost = cost
                nth +=1
        return weights
    
    def get_top_nn(self, X, X_image_name, m):
        X_features = pd.DataFrame(X)
        X_normalized = MinMaxScaler().fit_transform(X_features.values)
        X_features = pd.DataFrame(X_normalized)
        X_features.insert(loc=len(X_features.columns), column='intercept', value=1)
        X_test_local = X_features.to_numpy()
        
        indx = []
        dist = []
        for i in range(0, X_test_local.shape[0]):
            yp = np.dot(X_test_local[i], self.W[0])
            dist.append(yp)
        dist = np.array(dist)
        idx = np.argsort(dist)[::-1]
        idx = idx[:m]
        return X_image_name[idx]

