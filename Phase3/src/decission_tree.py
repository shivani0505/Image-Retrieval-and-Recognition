# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:40:55 2021

@author: rchityal
"""
from sklearn.datasets import load_iris
from pprint import pprint
import utils
import math
import numpy as np
import inquirer
import os
from color_moments import ColorMoments
from pca import PCA
from sklearn.utils import shuffle
import re
import constants
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




class Node:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.feature_num = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.input_output_map = {}
        self.output_input_map = {}

    def fit(self, X_train, Y_train):
        
        self.classes_number = len(set(Y_train))
        classes = set(Y_train)
        classes = list(classes)
        classes.sort()
        c = 0
        for i in classes:
            self.input_output_map[i] = c
            c = c + 1
        for i in classes:
            self.output_input_map[self.input_output_map[i]] = i
        new_Y_train = []
        for i in Y_train:
            new_Y_train.append(self.input_output_map[i])
        new_Y_train = np.asarray(new_Y_train)
        self.features_number = X_train.shape[1]
        self.main_tree = self.building_the_tree(X_train, new_Y_train)

    def predict(self, X_train):
        results = []
        for val in X_train:
            node = self.main_tree
            while node.left:
                if val[node.feature_num] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            results.append(node.num_classes)
        new_results = []
        for i in results:
            new_results.append(self.output_input_map[i])
        new_results = np.asarray(new_results)
        return new_results
            
        # return [self._predict(inputs) for inputs in X]

    def find_best_split(self, X, y):
        
        # print("****************************")
        # print(y.shape)
        # print(self.classes_number)
        
        
        min_index = None
        num_samples_per_class = []
        for i in range(self.classes_number):
            c = 0
            for y_val in y:
                if (y_val == i):
                    c = c + 1
            num_samples_per_class.append(c)
        if y.size <= 1:
            return None, None
        
        min_gini = 1.0 - sum((n / y.size) ** 2 for n in num_samples_per_class)
        
        
        threshold = None
        for index in range(self.features_number):
            threshold_sorted, Y_sorted = zip(*sorted(zip(X[:, index], y)))
            
            left_y = [0] * self.classes_number
            right_y = num_samples_per_class.copy()
            for i in range(1, y.size):
                
                c = Y_sorted[i - 1]
                # print(len(left_y))
                # print(c)
                # print(left_y)
                # print(c)
                left_y[c] += 1
                right_y[c] -= 1
                
                temp = 0
                for j in range(self.classes_number):
                    temp = temp + ((left_y[j] / i) ** 2)
                left = 1.0 - temp
                
                temp = 0
                for j in range(self.classes_number):
                    temp = temp + ((right_y[j] /(y.size - i)) ** 2)
                right = 1.0 - temp
                
                
                gini = (i * left + (y.size - i) * right) / y.size
                if threshold_sorted[i] == threshold_sorted[i - 1]:
                    continue
                
                if gini < min_gini:
                    min_gini = gini
                    min_index = index
                    threshold = (threshold_sorted[i] + threshold_sorted[i - 1]) / 2
        return min_index, threshold

    
    def building_the_tree(self, X, y, depth=0):
        # num_samples_per_class = [np.sum(y == i) for i in range(self.classes_number)]
        
        num_samples_per_class = []
        for i in range(self.classes_number):
            c = 0
            for y_val in y:
                if (y_val == i):
                    c = c + 1
            num_samples_per_class.append(c)
        
        num_classes = np.argmax(num_samples_per_class)
        # print(num_classes)
        
        node = Node(num_classes=num_classes)
        
        if depth < self.max_depth:
            index, threshold = self.find_best_split(X, y)
            
            if index is not None:
                left_index = X[:, index] < threshold
                # print(left_index)
                # print(threshold)
                X_left, y_left = X[left_index], y[left_index]
                X_right, y_right = X[~left_index], y[~left_index]
                node.feature_num = index
                node.threshold = threshold
                node.left = self.building_the_tree(X_left, y_left, depth + 1)
                node.right = self.building_the_tree(X_right, y_right, depth + 1)
        return node



    


def extract_all_images(images_path):
    files = os.listdir(images_path)
    all_files = [i for i in files]
    return all_files


def extract_all_X_images(data_path, X):
    files = os.listdir(data_path)
    X_files = [i for i in files if re.search(r"\w*("+str(X)+")\w*", i)]
    return X_files

def main_():

    iris = load_iris()
    
    x = iris.data
    y = iris.target
    
    clf = DecisionTreeClassifier(max_depth=7)
    m = clf.fit(x, y, iris)
    
    pprint(m)
def transform_image_vectors(image_vectors):
    image_vectors = np.array(image_vectors)
    min_vectors = image_vectors.min(axis=0)
    min_value = np.min(min_vectors)
    if ( min_value < 0 ):
        image_vectors = image_vectors + (-1*min_value)
    return image_vectors

def run_PCA(features, k):
    #TODO run SVD and save the output file in proper format
    pca_instance = PCA()
    decomposed = pca_instance.get_PCA(features, int(k))
    return decomposed
    


def pre_process(images_path):
    red_tech = {}
    feature_model = {}
    feature_model['feature_model'] = 'Color_Moments'
    red_tech['red_tech'] = 'PCA'
    k = 10
    # images_path = inquirer.prompt("enter image folder path")
    # images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//3_phase//Phase3//src//100//100"
    
    images = extract_all_images(images_path)
    
    
    # X_train = []
    Y_train = []
    
    images = []
    count = 0
    for indx, img_type in enumerate(constants.list_image_types):        
        type_images = extract_all_X_images(images_path, img_type)
        if (len(type_images) == 0):
            # print(img_type)
            continue;
        images.extend(type_images)
        for i in range(len(type_images)):
            # count = count + 1
            Y_train.append(indx)
            
        count = count + 1

    # print(len(images))
    # print(len(Y_train))
    if (len(images) == 0 ):
        print("No such images present in the DB")
        return
    
    #create a feature model object and image vectors
    FM = ColorMoments(images_path)
    image_vectors = []
    
    if ( feature_model['feature_model'] == 'Color_Moments'):
        image_vectors = FM.compute_image_color_moments(images)
    
    elif ( feature_model['feature_model'] == 'HOG' ):
        image_vectors = FM.compute_image_hog(images)
    
    elif ( feature_model['feature_model'] == 'ELBP' ):
        image_vectors = FM.compute_image_ELBP(images)
    
    #create a decomposed object
    image_vectors = transform_image_vectors(image_vectors)
    decomposed = []
    #apply proper reduction technique
    decomposed = run_PCA(image_vectors, k)
    
    # iris = load_iris()
    X_train = []
    # Y_train = []
    for indx, features in enumerate(decomposed[0]):
        X_train.append(features)
        # Y_train.append(indx)
    
    # x = iris.data
    # y = iris.target
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    
    return X_train, Y_train
def main():
# =============================================================================
#     feature_model = inquirer.prompt(self.feature_models)
#     image_x = inquirer.prompt(self.image_type)
#     red_tech = inquirer.prompt(self.reduction_tech)
#     k = input("input the value of k: ")
#     
# =============================================================================
        
    #get all the X images
    # images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//3_phase//Phase3//src//100//100"
  
    # clf = DecisionTreeClassifier(max_depth = 100)
    
    # m = clf.fit(X_train, Y_train)
    
    for i in range(10):
        
        # images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//final//all"
        images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//3_phase//Phase3//src//1000//1000"
        X_train, Y_train = pre_process(images_path)
        
        
        
        
        clf = DecisionTreeClassifier(max_depth = i + 1)
        m = clf.fit(X_train, Y_train)
        
        # sk_model = DecisionTreeClassifier(max_depth = i)
        # sk_model.fit(X_train, Y_train)
        
    
        # pprint(m)
        
        # X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        # print(Y_train)
        
        images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//3_phase//Phase3//src//100//100"
        # images_path = "C://Users//rchityal//Desktop//ASU//ASU//sem2//MWDB//final//all"
        
        
        X_train, Y_train = pre_process(images_path)
        
        
        result = clf.predict(X_train)
        
        print(i)
        print(accuracy_score(Y_train, result))
        
        # print(Y_train)
        # print(result)
        
        # sk_preds = sk_model.predict(X_train)
    
        # print(Y_train)
        # print(sk_preds)
        
        # print(accuracy_score(Y_train, sk_preds))
    
    
if __name__=="__main__":
    main()