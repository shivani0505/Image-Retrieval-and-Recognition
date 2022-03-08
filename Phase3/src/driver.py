#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 09:06:19 2021

@author: khushalmodi
"""
import os
import re

from scipy.spatial import distance
from image_mcq import MCQ
import constants
import task4_phase3
from color_moments import ColorMoments
from FM_QueryImage import Q_Img_FM
from Top_K_Images import Top_K_Img
import inquirer
from svd import SVD
from svm import SVM
from ppr_classifier import PPR_C
from PIL import Image
import numpy as np
from pca import PCA
import sys
from lda import LDA
from kmeans import KMeans
import pandas as pd
from tabulate import tabulate
from Task8 import Task8
import personalizePageRank
import utils
import os
import task6, task7, task5
import pickle
from datetime import datetime
from Top_K_Images import Top_K_Img
from sklearn.metrics import accuracy_score
from decission_tree import DecisionTreeClassifier
from Task7_Feedback import Task7_Feedback
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class TaskDriverPhase3:
    
    def __init__(self):
        self.data_path = constants.input_path
        self.out_path = constants.output_path
        self.Y_feedback = {}
        self.task1 = "Task1: Associate label X to a folder of images."
        self.task2 = "Task2: Associate label Y to a folder of images."
        self.task3 = "Task3: Associate label Z to a folder of images."
        self.task4 = "Task4: Implement a LSH tool, use LSH tool to find nearby t images."
        self.task5 = "Task5: Implement a VA-file index tool, use the tool to find nearby images."
        self.task6 = "Task6: Implement a decision tree based relevance feedback system"
        self.task7 = "Task7: Implement a SVM classifier based relevance feedback system"
        self.questions = [
                              inquirer.List('task',
                                            message="What task do you need?",
                                            choices=[self.task1, self.task2, 
                                                     self.task3, self.task4,
                                                     self.task5, self.task6,
                                                     self.task7,
                                                     "exit", "change data path",
                                                     "change output path"],
                                        ),
                              ]
        self.reduction_tech = [
                              inquirer.List('reduction_technique',
                                            message="What reduction technique do you need?",
                                            choices=constants.list_reduction_techniques,
                                        ),
                              ]
        self.image_type = [
                              inquirer.List('image_type',
                                            message="What image type do you need?",
                                            choices=constants.list_image_types,
                                        ),
                              ]
        self.feature_models = [
                              inquirer.List('feature_model',
                                            message="What feature model do you need?",
                                            choices=constants.list_feature_models,
                                        ),
                              ]
        self.classifier_models = [
                              inquirer.List('class_model',
                                            message="Choose a classifier",
                                            choices=constants.list_classifiers,
                                        ),
                              ]
        self.use_reduction = [
            inquirer.List('use_reduction',
                          message="Do you want to use reduction method? ",
                          choices=constants.yes_no_choice,
                          ),
        ]
        
        self.file_name_dict = {}
        if ( os.path.exists(constants.dict_path) ):
            afile = open(constants.dict_path, "rb")
            self.file_name_dict = pickle.load(afile)
    
    def generate_a_folder_name(self):
        return str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    
    def save_file_dict(self):
        if (os.path.exists(constants.dict_path)):
            os.remove(constants.dict_path)
        
        afile = open(constants.dict_path, "wb")
        pickle.dump(self.file_name_dict, afile)
        return
        
    """
    extract all images of type X from the database
    Output: List of all the images of type X
    """
    def extract_all_X_images(self, X):
        files = os.listdir(self.data_path)
        X_files = [i for i in files if re.search(r"\w*("+str(X)+")\w*", i)]
        return X_files
    
    """
    extract all images of subject ID Y from the database
    Output: List of all the images of subject ID Y
    """
    def extract_all_Y_images(self, Y):
        files = os.listdir(self.data_path)
        Y_files = [i for i in files if re.search(r"(image)-\w*-("+str(Y)+")-\w*", i)]
        return Y_files
    
    """
    Changes the database path in the driver
    """
    def change_output_path(self):
        new_data_path = input("Input new output path: ")
        self.out_path = new_data_path
        return
    
    """
    Changes the database path in the driver
    """
    def change_source_path(self):
        new_data_path = input("Input new data path: ")
        self.data_path = new_data_path
        return
    
    """
    reads all the images from the given image_ids
    Output: numpy array of all images flattened
    """
    def read_all_input_images_names_from_folder(self, data_path):
        image_ids = os.listdir(data_path)
        return image_ids
    
    """
    reads all the images from the given image_ids
    Output: numpy array of all images flattened
    """
    def read_all_images(self, image_ids):
        images_np_arr = []
        for image_id in image_ids:
            img = Image.open(os.path.join(self.data_path, image_id))
            img = img.convert("L")
            img = np.array(img)
            images_np_arr.append(img.flatten())
        return np.array(images_np_arr)
    
    def read_given_single_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert("L")
        img = np.array(img)
        return img
    
    def get_Z_from_image_list(self, images):
        labels = []
        for image in images:
            labels.append(self.get_Z_from_image_name(image))
        return labels
    
    def get_Z_from_image_name(self, image):
        image_token = image.split("-")
        image_tokenized = image_token[3].split(".")
        return image_tokenized[0]
    
    def get_X_from_image_list(self, images):
        labels = []
        for image in images:
            labels.append(self.get_X_from_image_name(image))
        return labels
    
    def get_Y_from_image_list(self, images):
        labels = []
        for image in images:
            labels.append(self.get_Y_from_image_name(image))
        return labels
    
    def get_Y_from_image_name(self, image_name):
        image_token = image_name.split("-")
        return image_token[2]
    
    def get_X_from_image_name(self, image_name):
        image_token = image_name.split("-")
        return image_token[1]
    
    def check_if_folder_exists(self, folder):
        if ( folder in list(self.file_name_dict.keys())):
            return True
        return False
    
    def create_folder_in_file_dict(self, folder_path):
        if ( folder_path not in list(self.file_name_dict.keys())):
            self.file_name_dict[folder_path] = self.generate_a_folder_name()
        return
    
    def check_if_file_exists(self, sub_file_name, folder_name):
        orig_folder_name = self.file_name_dict[folder_name]
        orig_folder_path = os.path.join(constants.output_path, orig_folder_name)
        orig_file_path = os.path.join(orig_folder_path, sub_file_name)
        if ( os.path.exists(orig_file_path) ):
            return True
        return False
    
    def save_file_in_dict_in_df(self, folder_path, df, file_name):
        orig_folder_name = self.file_name_dict[folder_path]
        orig_folder_path = os.path.join(constants.output_path, orig_folder_name)
        orig_file_path = os.path.join(orig_folder_path, file_name)
        if ( os.path.exists(orig_folder_path) ):
            df.to_csv(orig_file_path)
        else:
            os.mkdir(orig_folder_path)
            df.to_csv(orig_file_path)
        return
    
    def save_file_in_dict(self, folder_path, data, file_name):
        orig_folder_name = self.file_name_dict[folder_path]
        orig_folder_path = os.path.join(constants.output_path, orig_folder_name)
        orig_file_path = os.path.join(orig_folder_path, file_name)
        if ( os.path.exists(orig_folder_path) ):
            np.save(orig_file_path, data)
        else:
            os.mkdir(orig_folder_path)
            np.save(orig_file_path, data)
        return
    
    def get_file_contents(self, file_name, images_folder):
        orig_folder_name = self.file_name_dict[images_folder]
        orig_folder_path = os.path.join(constants.output_path, orig_folder_name)
        orig_file_path = os.path.join(orig_folder_path, file_name)
        numpy_arr = self.load_numpy_array(orig_file_path)
        return numpy_arr
    
    def perf_measure(self, y_actual, y_predicted):
        cnf_matrix = confusion_matrix(y_actual, y_predicted)
        # print(cnf_matrix)
        
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        
        FPR = FP/(FP+TN)
        X = np.average(FPR)
        print("FALSE Positive Rate:")
        print(X)
        Miss_rate = FN / (FN + TP)
        X = np.average(Miss_rate)
        print("MISS RATE:")
        print(X)
    
    def visualize_k_images(self, images, query_image):        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(len(images)+1,
                                    figsize=(20,20),
                                    subplot_kw={'xticks': [], 'yticks': []})
        
        
        axis[0].imshow(np.array(query_image.reshape(64,64)), cmap='gray')
        counter = 1
        
        image_arr = []
        for image_id in images:
            image_np_arr = Image.open(os.path.join(self.data_path, image_id))
            image_np_arr = np.array(image_np_arr)
            image_arr.append(image_np_arr.flatten())
        
        pyto = []
        
        image_arr = np.array(image_arr)
        for i in range(0,len(images)):
            image_y = image_arr[i]
            axis[counter].imshow(np.array(image_y).reshape(64,64), cmap='gray')
            axis[counter].yaxis.set_label_position("right")
            axis[counter].set_ylabel("("+str(counter)+") "+str(images[i]),
                                     rotation=0,
                                     labelpad=80)
            counter+=1
        my_file = 'last_feedback_file.png'
        plt.savefig(os.path.join(constants.output_path, my_file))  
        
        plt.show()

    """
    Drives the flow for task 1
    """
    def driver_task1(self):
        images_folder =input("Input the image folder for latent semantics: ")
        k = input("Input the value of k: ")
        feature_model = inquirer.prompt(self.feature_models)
        classifier_model = inquirer.prompt(self.classifier_models)
        class_folder = input("Input the image folder for classification: ")
        self.create_folder_in_file_dict(images_folder)
        
        lev = []
        rev = []
        core = []
        lev_name = "lev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        rev_name = "rev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        core_name = "core-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        flag = True
        if ( self.check_if_folder_exists(images_folder) and k!="*" ):
            if ( self.check_if_file_exists(lev_name, images_folder)):
                rev = self.get_file_contents(rev_name, images_folder)
                core = self.get_file_contents(core_name, images_folder)
                flag = False
                print("No Need to run the dimensionality reduction...")
        
        FM = ColorMoments(str(images_folder))
        image_vectors = []
        
        images = self.read_all_input_images_names_from_folder(images_folder)
        
        if ( feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            image_vectors = FM.compute_image_hog(images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            image_vectors = FM.compute_image_ELBP(images)
        
        if ( flag and k!="*" ):
            #create a feature model object and image vectors
            
            
            #create a decomposed object
            image_vectors = self.transform_image_vectors(image_vectors)
            decomposed = []
            #apply proper reduction technique
            print("Running PCA...")
            decomposed = self.run_PCA(image_vectors, k)
            lev = decomposed[0]
            rev = decomposed[2]
            core = decomposed[1]
            self.save_file_in_dict(images_folder, lev, lev_name)
            self.save_file_in_dict(images_folder, rev, rev_name)
            self.save_file_in_dict(images_folder, core, core_name)
        
        #create a feature model object and image vectors
        class_images = self.read_all_input_images_names_from_folder(class_folder)
        class_FM = ColorMoments(class_folder)
        class_image_vectors = []
        if ( feature_model['feature_model'] == 'Color_Moments'):
            class_image_vectors = class_FM.compute_image_color_moments(class_images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            class_image_vectors = class_FM.compute_image_hog(class_images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            class_image_vectors = class_FM.compute_image_ELBP(class_images)
        
        #apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        class_image_vectors = self.transform_image_vectors(class_image_vectors)
        X_train = []
        X_test = []
        
        if ( k != "*" ):
            X_train = np.matmul(image_vectors, rev.T)
        else:
            X_train = image_vectors
        Y_train = np.array(self.get_X_from_image_list(images))
        Y_train = Y_train.reshape(Y_train.shape[0],1)

        if ( k!="*" ):
            X_test = np.matmul(class_image_vectors, rev.T)
        else:
            X_test = class_image_vectors
        #take the core matrix into consideration
        if ( k!="*" ):
            X_train = np.matmul(X_train, core)
            X_test = np.matmul(X_test, core)
                
        if ( classifier_model['class_model'] == 'SVM'):
            class_svm = SVM(X_train, Y_train)
            labels = class_svm.predict_multi_label_class(X_test)
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            df.to_csv("temp_output.csv")
            self.save_file_in_dict_in_df(images_folder, df, "SVM_task1.csv")
        
        elif ( classifier_model['class_model'] == 'Decision Tree' ):
            Y_test= np.array(self.get_X_from_image_list(class_images))
            Y_test = Y_test.reshape(Y_test.shape[0],1)
            
            X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
           
            Y_train_ = Y_train_.flatten()
            Y_test_ = Y_test_.flatten()
            
          
            class_dt = DecisionTreeClassifier(max_depth = 20)
            class_dt.fit(X_train_, Y_train_)
            labels = class_dt.predict(X_test_)
            print("Accuracy: ", accuracy_score(Y_test_, labels))
            
            
            # self.data_reults(Y_test_, labels)
            # self.perf_measure(Y_test_, labels)
            
            
            labels = class_dt.predict(X_test)
            # print("Accuracy")
            # print(accuracy_score(Y_test, labels))
            self.perf_measure(Y_test, labels)
            
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            self.save_file_in_dict_in_df(images_folder, df, "DT_task1.csv")
        
        elif ( classifier_model['class_model'] == 'PPR' ):
            class_ppr = PPR_C(X_train, Y_train, images, X_test, feature_model, images_folder)
            a = class_ppr.get_labels()
            blah = pd.DataFrame()
            Y_test = self.get_X_from_image_list(class_images)
            Y_model = self.get_X_from_image_list(a)
            
            print("accuracy on test dataset: {}".format(accuracy_score(Y_test, Y_model)))
            self.perf_measure(Y_test, Y_model)
            blah['orig'] = class_images
            blah['pred'] = a
            self.save_file_in_dict_in_df(images_folder, blah, "PPR_task1.csv")
        return
    
    """
    Drives the flow for task 2
    """
    def driver_task2(self):
        images_folder =input("Input the image folder for latent semantics: ")
        k = input("Input the value of k: ")
        feature_model = inquirer.prompt(self.feature_models)
        classifier_model = inquirer.prompt(self.classifier_models)
        class_folder = input("Input the image folder for classification: ")
        self.create_folder_in_file_dict(images_folder)
        
        lev = []
        rev = []
        core = []
        lev_name = "lev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        rev_name = "rev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        core_name = "core-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        flag = True
        if ( self.check_if_folder_exists(images_folder) and k!="*" ):
            if ( self.check_if_file_exists(lev_name, images_folder)):
                rev = self.get_file_contents(rev_name, images_folder)
                core = self.get_file_contents(core_name, images_folder)
                flag = False
                print("No Need to run the dimensionality reduction...")
        
        FM = ColorMoments(str(images_folder))
        image_vectors = []
        
        images = self.read_all_input_images_names_from_folder(images_folder)
        
        if ( feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            image_vectors = FM.compute_image_hog(images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            image_vectors = FM.compute_image_ELBP(images)
        
        if ( flag and k!="*" ):
            #create a feature model object and image vectors
            
            
            #create a decomposed object
            image_vectors = self.transform_image_vectors(image_vectors)
            decomposed = []
            #apply proper reduction technique
            print("Running PCA...")
            decomposed = self.run_PCA(image_vectors, k)
            lev = decomposed[0]
            rev = decomposed[2]
            core = decomposed[1]
            self.save_file_in_dict(images_folder, lev, lev_name)
            self.save_file_in_dict(images_folder, rev, rev_name)
            self.save_file_in_dict(images_folder, core, core_name)
        
        #create a feature model object and image vectors
        class_images = self.read_all_input_images_names_from_folder(class_folder)
        class_FM = ColorMoments(class_folder)
        class_image_vectors = []
        if ( feature_model['feature_model'] == 'Color_Moments'):
            class_image_vectors = class_FM.compute_image_color_moments(class_images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            class_image_vectors = class_FM.compute_image_hog(class_images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            class_image_vectors = class_FM.compute_image_ELBP(class_images)
        
        #apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        class_image_vectors = self.transform_image_vectors(class_image_vectors)
        X_train = []
        X_test = []
        
        if ( k != "*" ):
            X_train = np.matmul(image_vectors, rev.T)
        else:
            X_train = image_vectors
        Y_train = np.array(self.get_Y_from_image_list(images))
        Y_train = Y_train.reshape(Y_train.shape[0],1)

        if ( k!="*" ):
            X_test = np.matmul(class_image_vectors, rev.T)
        else:
            X_test = class_image_vectors
        #take the core matrix into consideration
        if ( k!="*" ):
            X_train = np.matmul(X_train, core)
            X_test = np.matmul(X_test, core)
                
        if ( classifier_model['class_model'] == 'SVM'):
            class_svm = SVM(X_train, Y_train)
            labels = class_svm.predict_multi_label_class(X_test)
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            df.to_csv("temp_output.csv")
            self.save_file_in_dict_in_df(images_folder, df, "SVM_task2.csv")
        
        elif ( classifier_model['class_model'] == 'Decision Tree' ):
            Y_test= np.array(self.get_Y_from_image_list(class_images))
            Y_test = Y_test.reshape(Y_test.shape[0],1)
            
            X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
           
            Y_train_ = Y_train_.flatten()
            Y_test_ = Y_test_.flatten()
            
          
            class_dt = DecisionTreeClassifier(max_depth = 20)
            class_dt.fit(X_train_, Y_train_)
            labels = class_dt.predict(X_test_)
            print("Accuracy: ", accuracy_score(Y_test_, labels))
            
            
            # self.data_reults(Y_test_, labels)
            # self.perf_measure(Y_test_, labels)
            
            
            labels = class_dt.predict(X_test)
            # print("Accuracy")
            # print(accuracy_score(Y_test, labels))
            self.perf_measure(Y_test, labels)
            
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            self.save_file_in_dict_in_df(images_folder, df, "DT_task2.csv")
        
        elif ( classifier_model['class_model'] == 'PPR' ):
            class_ppr = PPR_C(X_train, Y_train, images, X_test, feature_model, images_folder)
            a = class_ppr.get_labels()
            blah = pd.DataFrame()
            Y_test = self.get_Y_from_image_list(class_images)
            Y_model = self.get_Y_from_image_list(a)
            
            print("accuracy on test dataset: {}".format(accuracy_score(Y_test, Y_model)))
            self.perf_measure(Y_test, Y_model)
            blah['orig'] = class_images
            blah['pred'] = a
            self.save_file_in_dict_in_df(images_folder, blah, "PPR_task2.csv")
        return
    
    """
    Drives the flow for task 3
    """
    def driver_task3(self):
        images_folder =input("Input the image folder for latent semantics: ")
        k = input("Input the value of k: ")
        feature_model = inquirer.prompt(self.feature_models)
        classifier_model = inquirer.prompt(self.classifier_models)
        class_folder = input("Input the image folder for classification: ")
        self.create_folder_in_file_dict(images_folder)
        
        lev = []
        rev = []
        core = []
        lev_name = "lev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        rev_name = "rev-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        core_name = "core-"+str(k)+"-"+str(feature_model['feature_model'])+".npy"
        flag = True
        if ( self.check_if_folder_exists(images_folder) and k!="*" ):
            if ( self.check_if_file_exists(lev_name, images_folder)):
                rev = self.get_file_contents(rev_name, images_folder)
                core = self.get_file_contents(core_name, images_folder)
                flag = False
                print("No Need to run the dimensionality reduction...")
        
        FM = ColorMoments(str(images_folder))
        image_vectors = []
        
        images = self.read_all_input_images_names_from_folder(images_folder)
        
        if ( feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            image_vectors = FM.compute_image_hog(images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            image_vectors = FM.compute_image_ELBP(images)
        
        if ( flag and k!="*" ):
            #create a feature model object and image vectors
            
            
            #create a decomposed object
            image_vectors = self.transform_image_vectors(image_vectors)
            decomposed = []
            #apply proper reduction technique
            print("Running PCA...")
            decomposed = self.run_PCA(image_vectors, k)
            lev = decomposed[0]
            rev = decomposed[2]
            core = decomposed[1]
            self.save_file_in_dict(images_folder, lev, lev_name)
            self.save_file_in_dict(images_folder, rev, rev_name)
            self.save_file_in_dict(images_folder, core, core_name)
        
        #create a feature model object and image vectors
        class_images = self.read_all_input_images_names_from_folder(class_folder)
        class_FM = ColorMoments(class_folder)
        class_image_vectors = []
        if ( feature_model['feature_model'] == 'Color_Moments'):
            class_image_vectors = class_FM.compute_image_color_moments(class_images)
        
        elif ( feature_model['feature_model'] == 'HOG' ):
            class_image_vectors = class_FM.compute_image_hog(class_images)
        
        elif ( feature_model['feature_model'] == 'ELBP' ):
            class_image_vectors = class_FM.compute_image_ELBP(class_images)
        
        #apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        class_image_vectors = self.transform_image_vectors(class_image_vectors)
        X_train = []
        X_test = []
        
        if ( k != "*" ):
            X_train = np.matmul(image_vectors, rev.T)
        else:
            X_train = image_vectors
        Y_train = np.array(self.get_Z_from_image_list(images))
        Y_train = Y_train.reshape(Y_train.shape[0],1)

        if ( k!="*" ):
            X_test = np.matmul(class_image_vectors, rev.T)
        else:
            X_test = class_image_vectors
        #take the core matrix into consideration
        if ( k!="*" ):
            X_train = np.matmul(X_train, core)
            X_test = np.matmul(X_test, core)
                
        if ( classifier_model['class_model'] == 'SVM'):
            class_svm = SVM(X_train, Y_train)
            labels = class_svm.predict_multi_label_class(X_test)
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            df.to_csv("temp_output.csv")
            self.save_file_in_dict_in_df(images_folder, df, "SVM_task3.csv")
        
        elif ( classifier_model['class_model'] == 'Decision Tree' ):
            Y_test= np.array(self.get_Z_from_image_list(class_images))
            Y_test = Y_test.reshape(Y_test.shape[0],1)
            
            X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
           
            Y_train_ = Y_train_.flatten()
            Y_test_ = Y_test_.flatten()
            
          
            class_dt = DecisionTreeClassifier(max_depth = 20)
            class_dt.fit(X_train_, Y_train_)
            labels = class_dt.predict(X_test_)
            print("Accuracy: ", accuracy_score(Y_test_, labels))
            
            
            # self.data_reults(Y_test_, labels)
            # self.perf_measure(Y_test_, labels)
            
            
            labels = class_dt.predict(X_test)
            # print("Accuracy")
            # print(accuracy_score(Y_test, labels))
            self.perf_measure(Y_test, labels)
            
            df = pd.DataFrame()
            df['image_name'] = class_images
            df['labels'] = labels
            self.save_file_in_dict_in_df(images_folder, df, "DT_task3.csv")
        
        elif ( classifier_model['class_model'] == 'PPR' ):
            class_ppr = PPR_C(X_train, Y_train, images, X_test, feature_model, images_folder)
            a = class_ppr.get_labels()
            blah = pd.DataFrame()
            Y_test = self.get_Z_from_image_list(class_images)
            Y_model = self.get_Z_from_image_list(a)
            
            print("accuracy on test dataset: {}".format(accuracy_score(Y_test, Y_model)))
            self.perf_measure(Y_test, Y_model)
            blah['orig'] = class_images
            blah['pred'] = a
            self.save_file_in_dict_in_df(images_folder, blah, "PPR_task3.csv")
        return
    
    """
    Drives the flow for task 4
    """
    def driver_task4(self):
        feature_model = inquirer.prompt(self.feature_models)
        red_tech = inquirer.prompt(self.reduction_tech)
        k = input("Input the value of k: ")
        #create a feature model object and image vectors 
        FM = ColorMoments(self.data_path)
        subject_image_vectors = []


        images_subject_list = []
        # get type image vectors by average the image vectors for all the
        # images of same type
        for subject_id in range(1, 41):
            # get all the images of same type

            images = self.extract_all_Y_images(subject_id)

            if len(images) == 0:
                continue

            images_subject_list.append(subject_id)
            # extract image vectors for each image of same type
            image_vectors = []
            if ( feature_model['feature_model'] == 'Color_Moments'):
                image_vectors = FM.compute_image_color_moments(images)


            elif ( feature_model['feature_model'] == 'HOG' ):
                image_vectors = FM.compute_image_hog(images)
            
            elif ( feature_model['feature_model'] == 'ELBP' ):
                image_vectors = FM.compute_image_ELBP(images)
            
            image_vectors = self.transform_image_vectors(image_vectors)
            # get the average of all the image vectors of same type
            image_vectors = image_vectors.mean(0)
            
            subject_image_vectors.append(image_vectors)
            
        subject_image_vectors = np.asarray(subject_image_vectors)
        
        subject_subject_similarity_matrix = np.matmul(subject_image_vectors, subject_image_vectors.T)

        #create a decomposed object
        decomposed = []

        image_vectors = subject_subject_similarity_matrix
        #apply proper reduction technique
        if ( red_tech['reduction_technique'] == 'SVD' ):
            decomposed = self.run_SVD(image_vectors, k)
            self.save_numpy_array(image_vectors, 
                                  os.path.join(self.out_path , "task4-SVD-"+str(feature_model['feature_model'])+"-" +str(k)+'-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T, 
                                  os.path.join(self.out_path , "task4-SVD-"+str(feature_model['feature_model'])+"-" +str(k)+'-transform-1.npy'))
            self.save_numpy_array(decomposed[0], 
                                  os.path.join(self.out_path , "task4-SVD-"+str(feature_model['feature_model'])+"-" +str(k)+'-lev.npy'))
            self.save_numpy_array(decomposed[1], 
                                  os.path.join(self.out_path , "task4-SVD-"+str(feature_model['feature_model']) + "-" +str(k)+'-ll.npy'))
            self.save_numpy_array(decomposed[2], 
                                  os.path.join(self.out_path , "task4-SVD-"+str(feature_model['feature_model']) + "-" + str(k)+'-rev.npy'))
        elif ( red_tech['reduction_technique'] == 'PCA' ):
            decomposed = self.run_PCA(image_vectors, k)
            self.save_numpy_array(image_vectors, 
                                  os.path.join(self.out_path , "task4-PCA-"+str(feature_model['feature_model'])+"-" +str(k)+'-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T, 
                                  os.path.join(self.out_path , "task4-PCA-"+str(feature_model['feature_model'])+"-" +str(k)+'-transform-1.npy'))
            self.save_numpy_array(decomposed[0], 
                                  os.path.join(self.out_path , "task4-PCA-"+str(feature_model['feature_model'])+"-"+str(k)+'-lev.npy'))
            self.save_numpy_array(decomposed[1], 
                                  os.path.join(self.out_path , "task4-PCA-"+str(feature_model['feature_model']) +"-"+str(k)+'-ll.npy'))
            self.save_numpy_array(decomposed[2], 
                                  os.path.join(self.out_path , "task4-PCA-"+str(feature_model['feature_model'])+ "-" + str(k)+'-rev.npy'))
        elif ( red_tech['reduction_technique'] == 'LDA' ):
            decomposed = self.run_LDA(image_vectors, k)
            self.save_numpy_array(image_vectors, 
                                  os.path.join(self.out_path , "task4-LDA-"+str(feature_model['feature_model'])+"-" +str(k)+'-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T, 
                                  os.path.join(self.out_path , "task4-LDA-"+str(feature_model['feature_model'])+"-" +str(k)+'-transform-1.npy'))
            self.save_numpy_array(decomposed[0], 
                                  os.path.join(self.out_path , "task4-LDA-"+str(feature_model['feature_model'])+ "-" +str(k)+'-lev.npy'))
            self.save_numpy_array(decomposed[1], 
                                  os.path.join(self.out_path , "task4-LDA-"+str(feature_model['feature_model'])+"-"+str(k)+'-rev.npy'))
        elif( red_tech['reduction_technique'] == 'k-means'):
            decomposed = self.run_k_means(image_vectors, k)
            self.save_numpy_array(image_vectors, 
                                  os.path.join(self.out_path , "task4-k-means-"+str(feature_model['feature_model'])+'-'+str(k)+'-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T, 
                                  os.path.join(self.out_path , "task4-k-means-"+str(feature_model['feature_model'])+'-'+str(k)+'-transform-1.npy'))
            self.save_k_means_output(images, image_vectors, k, decomposed, "task4", str(feature_model['feature_model']), "subject")
        
        
        # TODO get subject weight pairs for each latent semantics
        self.get_subject_weight_pairs_task_3(images_subject_list, decomposed[0])
        return
    
    """
    Extracts and returns the file names in the input folder
    """
    def extract_all_images(self):
        files = os.listdir(self.data_path)
        files = [i for i in files]
        return files
    
    """
    read the transformation matrix
    """
    def load_transformation_matrix(self, filename):
        if ( os.path.exists(os.path.join(self.out_path , filename+"-transform-1.npy"))):
            transform_file = self.load_numpy_array(os.path.join(self.out_path , filename+"-transform-1.npy"))
            return True, transform_file
        return False, None
    
    def inputFeatureDictionary(self, Xtrain):
        Xinput = []
        for k in Xtrain:
            Xinput.append(Xtrain[k])
        # length = len(Xtrain)
        length = np.size(Xinput,1)
        x = pd.DataFrame(data=Xinput)
        x.columns = [str(i) for i in range(1,length+1)]
        x.index = [k for k in Xtrain]
        return x
    """
    Drives the flow for task 5
    """
    def driver_task5(self):
        #task5.Task5Subtask1()
        #print("###### Phase 5 Subtask 2 #####")
        b=input("Enter the number of bits")
        b=int(b)
        data_path=input("enter the absolute path of the images directory")
        fileList = task5.getFiles(data_path)
        typeOfData=input("Enter 1 for CM, 2 for ELBP, 3 for HOG")
        images_dictionary={}
        typeOfData=int(typeOfData)
        targetImagePath=input("Enter the absolute path of the target image")
        model=""
        if(typeOfData==1):
            Images_dictionary=task5.compute_image_color_moments(data_path,fileList)
            model="Color_Moments"
        elif(typeOfData==2):
            Images_dictionary=task5.compute_image_ELBP(data_path,fileList)
            model="HOG"
        elif(typeOfData==3):
            Images_dictionary=task5.compute_image_hog(data_path,fileList)
            model="ELBP"
        candidateList = task5.Task5Init(Images_dictionary, b, typeOfData,targetImagePath,data_path,model)

        return candidateList
    
    """
    loads the latent semantics and returns a list of all numpy arrays of decomposed matrix
    """
    def load_core_matrix(self, filename, k):
        if ( os.path.exists(os.path.join(self.out_path , filename+"-ll.npy"))):
            core_file = os.path.join(self.out_path , filename+"-ll.npy")
            core_matrix = self.load_numpy_array(core_file)
            return core_matrix
        else:
            core_matrix = np.diag(np.ones(k))
            return core_matrix
    
    """
    Drives the flow for task 6
    """
    def driver_task6(self):

        indexing_file = input("Input the indexing filename e.g. phase3-task4: ")
        # indexing_file = '2021-12-02-22-42-29/phase3_task4_Color_Moments_0'
    
        similarityDistances,nearest_images,query_feature,query_name = self.load_indexing_file(indexing_file)

        # k = int(input('Input k: '))
        k = 10
        query_image = self.read_given_single_image(query_name[0])

        X = nearest_images
        numpy_array = similarityDistances
        ysim ={}
        for x, y in numpy_array:
                ysim[x]= float(y)

        # while True:        
        i=0
        simil=[] 
        for x in ysim:
            i=i+1
            simil.append(float(ysim[x]))

        query_name = query_name[0]
        
        print("Query_name is :", query_name)
        
        query_image = self.read_given_single_image(query_name)

        query_name_list = re.split(r'\\|/', query_name)
        s = "/"
        query_path = s.join(query_name_list[:len(query_name_list) -1])
        query_name = query_name_list[len(query_name_list) - 1]

        print("query_name", query_name, "query_path", query_path)

        print("Top "+str(k)+" images are: ")
        print()
        topK = []
        for entry in sorted(ysim.items(), key=lambda item: item[1], reverse=False)[:k]:
            print(entry[0])
            topK.append(entry[0])
        
        self.visualize_k_images(topK,query_image)
        
        i = input("\n\n Press 1 to give feedback else press 0 : \n\n ")

        if i == '0':
            return
        

        # query_image = self.read_given_single_image(query_name[0])
        

        topK = self.build_And_predict(similarityDistances,nearest_images,query_image,k, X,ysim,topK,query_path)
        
        
        # i = input("\n\n Press 1 to give feedback else press 0 : \n\n ")

        # if i == '0':
        #     return
        # else:
        #     self.driver_task6()

        return

    def build_And_predict(self,similarityDistances,nearest_images,query_image,k, X, ysim,topK,query_path):

        print("\n Please give for the images : \n ",   topK)
        feedback_mcq = MCQ(self.data_path, topK, query_path)
        relevant_images, irrelevant_images = feedback_mcq.give_options()
        
        # relevant_images = [ 'image-stipple-8-1.png', 'image-stipple-16-3.png']
        # irrelevant_images =[]

        print(relevant_images)
        print(irrelevant_images)

        R = relevant_images
        IR = irrelevant_images

        for key in R:
            self.Y_feedback[key] = 1
        
        for key in IR:
            self.Y_feedback[key] = 0
        print()
        
        XId ={}
        i=0
        # print(ysim[0])
        for x in ysim:
            XId[x] = X[i]
            i=i+1
        # Ytrain = {}
        Xtrain = {}
        counter =0
        
        print()
        Xtest = XId
        ytrain = []
        
        if(not bool(self.Y_feedback)):
            print("No feedback detected, Exiting")
            return
        
        for key in self.Y_feedback:
            Xtrain[key]= XId[key]
            Xtest.pop(key)
            ytrain.append(self.Y_feedback[key])
        average = 0
        
        for val in ysim:
            average = float(ysim[val]) + average
        
        average = average /len(ysim)
        xtrain = self.inputFeatureDictionary(Xtrain)
        xtest = self.inputFeatureDictionary(Xtest)
        
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(xtrain.to_numpy(),np.array(ytrain))
        predictedY = dt.predict(np.array(xtest))
        
        j =0
        labelled ={}
        for key in Xtest:
            labelled[key] = predictedY[j]
            j = j+1

        newSimilarity = {}
        
        for val in labelled:
            if labelled[val] == 0:
                newSimilarity[val] = float(ysim[val]) + average/5
            if labelled[val] == 1:
                newSimilarity[val] = float(ysim[val]) - average/5
        
        for key in self.Y_feedback:
            
            if self.Y_feedback[key] == 0:
                newSimilarity[key] = float(ysim[key]) + average
            if self.Y_feedback[key] == 1:
                newSimilarity[key] = float(ysim[key]) - average
    

        ysim = dict(sorted(newSimilarity.items(), key=lambda item: item[1]))

        print()
        print("New Top "+str(k)+" images are: ")
        print()
        
        topK =[]
        for entry in sorted(newSimilarity.items(), key=lambda item: item[1])[:k]:
            print(entry[0])
            topK.append(entry[0])
        self.visualize_k_images(topK,query_image)

        i = input("\n\n Press 1 to give feedback else press 0 : \n\n ")
        if i == '0':
            return
        else:
            self.build_And_predict(similarityDistances, nearest_images, query_image, k, X, ysim, topK, query_path)

    """
    Drives the flow for task 7
    """
    def driver_task7(self):
        indexing_file = input("Input the indexing filename e.g. phase3-task4: ")
        #indexing_file = 'phase3_task4'
        
        candidate_name_id,candidate_feature,query_feature,query_name = self.load_indexing_file(indexing_file)
        
        if len(candidate_feature) == 0:
            print("No nearest neighbor")
            return
        
        candidate_dict = {}
        
        for i in range(len(candidate_name_id)):
            name = candidate_name_id[i][0]
            feature = candidate_feature[i]
            candidate_dict[name] = feature
        
        n = input("Number of nearest neighbor")
        #n = 10
        n = int(n)
        
        top_k = []
        i = 0
        for keys in candidate_dict:
            if i < n:
                top_k.append(keys)
                i = i+1
            else:
                break
        
        
        query_name = query_name[0]
        #query_name = '/Users/shivanipriya/Documents/GitHub/Multimedia-and-Web-Databases copy 6/Phase3/2000/image-cc-4-10.png'
        print("Query_name is :", query_name)
        
        query_image = self.read_given_single_image(query_name)

        query_name_list = re.split(r'\\|/', query_name)
        s = "/"
        query_path = s.join(query_name_list[:len(query_name_list) -1])
        query_name = query_name_list[len(query_name_list) - 1]

        print("query_name", query_name, "query_path", query_path)
        
        Task7_FB = Task7_Feedback(candidate_dict,self.data_path,query_image,query_name,query_feature,query_path,n)
        Task7_FB.visualize_k_images(top_k)
        Task7_FB.getFeedback(top_k)
        return
    """
    Drives the flow for task 8
    """
    def driver_task8(self):
        n = input("Input the value of n")
        m = input("Input the value of m")

        S_S_WeightMatrix_file = input("Input the similarity matrix filename in form (task)-(reduction technique)-(feature model)-(k)e.g. task1-PCA-HOG-5: ")
        S_S_WeightMatrix = self.load_numpy_array(os.path.join(self.out_path, S_S_WeightMatrix_file+"-similarity.npy"))
        Task8_instance = Task8()
        S_S_SimilarityMatrix = utils.convert_similarity_similarity_matrix_to_dict(S_S_WeightMatrix)
        formedGraph = Task8_instance.task8_Subtask1(S_S_SimilarityMatrix,n,m)
        s=""
        fname = constants.output_path+"/Task8-Graph.txt"
        with open(fname,'w') as ofile:
            for i in formedGraph:
                s+="\n\n"
                s+="Node:"
                s+=str(i)
                s+="\n"
                s+="Neighbors->"
                for j in formedGraph[i]:
                    s+="{Node: "
                    s+=str(j)
                    s+=" Strength: "
                    s+=str(formedGraph[i][j])
                    s+="}  "
            ofile.write(s)
            ofile.close()
        topNodes = Task8_instance.AscosPlus(formedGraph,m)
        st=""
        fname = constants.output_path+"/Task8-top-m-nodes.txt"
        with open(fname,'w') as ofile:
            for i in topNodes:
                st+="\n\n"
                st+="Node: "
                st+=str(i)
                st+=" Strength: "
                st+=str(topNodes[i])
            ofile.write(st)
            ofile.close()
        return

    
    """
    Drives the flow for task 9
    """
    def driver_task9(self):
        sim_file = input("Input the similarity matrix filename in form (task)-(reduction technique)-(feature model)-(k)e.g. task1-PCA-HOG-5: ")
        n = input("input value of n: ")
        m = input("input value of m: ")
        subject_id = num_list = list(int(num) for num in input("Enter the subject IDs separated by a space: ").strip().split())[:3]
        sim_file = self.load_numpy_array(os.path.join(self.out_path, sim_file+"-similarity.npy"))
        object = personalizePageRank.ppr()
        similarityGraph = object.convertToGraph(sim_file,n,m)
        result = object.findPersonalizedRank(subject_id[0],subject_id[1], subject_id[2], similarityGraph, 0.75)
        print("page rank vector - ")
        print(result)
        idx = np.argsort(result)[::-1]
        for i in range(0, int(m)):
            print("Subject id :", idx[i] + 1)
        self.save_numpy_array(similarityGraph, os.path.join(self.out_path , "task9-similarity-graph.npy"))
        return
    
    def get_type_weight_pairs_task_3(self, image_types, decomposed):
        latent_dict = {}
        print ("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        for i in range(0, (decomposed.shape[1])):
            type_weights = []
            for j in range(0, decomposed.shape[0]):
                type_weights.append(decomposed[j][i])
            type_weight_pairs = zip(type_weights, image_types)
            type_weight_pairs = sorted(type_weight_pairs, reverse=True)
            tuples = zip(*type_weight_pairs)
            type_weight, image_types_temp = [ list(tuple) for tuple in  tuples]
            concated = []
            for indx,  image_type in enumerate(image_types_temp):
                concated.append(str("Type:" ) + image_type + " W:" + str(round(type_weight[indx], 4)))
                # print(image_type)
                # print(type_weight[indx])
            latent_dict["Latent-"+str(i)] = np.array(concated)
        self.print_table_from_dict(latent_dict)

    def get_subject_weight_pairs_task_3(self, subjects, decomposed):

        latent_dict = {}
        print ("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        for i in range(0, (decomposed.shape[1])):
            subject_weight = []
            for j in range(0, decomposed.shape[0]):
                subject_weight.append(decomposed[j][i])
            subject_weight_pairs = zip(subject_weight, subjects)
            subject_weight_pairs = sorted(subject_weight_pairs, reverse=True)
            tuples = zip(*subject_weight_pairs)
            subject_weight, subject_temp = [ list(tuple) for tuple in  tuples]
            concated = []
            for indx,  subject in enumerate(subject_temp):
                concated.append(str("S: ")  + str(subject)  + " W:" + str(round(subject_weight[indx], 4)))
                # print(subject)
                # print(subject_weight[indx])
            latent_dict["Latent-"+str(i)] = np.array(concated)
        self.print_table_from_dict(latent_dict)
        return
    
    """
    Starts the driver code
    """
    def start_driver(self):
        while (True):
            run_task = inquirer.prompt(self.questions)
            task = run_task["task"]
            if ( task == "exit" ):
                self.save_file_dict()
                break
            elif ( task == "change data path" ):
                self.change_source_path()
            elif ( task == "change output path" ):
                self.change_output_path()
            elif ( task == self.task1 ):
                self.driver_task1()
            elif ( task == self.task2 ):
                self.driver_task2()
            elif ( task == self.task3 ):
                self.driver_task3()
            elif ( task == self.task4 ):
                self.driver_phase3_task4()
            elif ( task == self.task5 ):
                self.driver_task5()
            elif ( task == self.task6 ):
                self.driver_task6()
            elif ( task == self.task7 ):
                self.driver_task7()
            else:
                print("Invalid choice")
            print("\n\n")
    
    """
    loads the latent semantics and returns a list of all numpy arrays of decomposed matrix
    """
    def load_latent_semantics(self, filename):
        if ( os.path.exists(os.path.join(self.out_path , filename+"-rev.npy"))):
            rev_file = os.path.join(self.out_path , filename+"-rev.npy")
            lev_file = os.path.join(self.out_path , filename+"-lev.npy")
            rev = self.load_numpy_array(rev_file)
            lev = self.load_numpy_array(lev_file)
            return lev, rev
        print ("*****ERROR: FILE DOES NOT EXIST*****")
        return
    
    def load_indexing_file(self,filename):
        folder_name = input("Please enter folder path for the indexing files : \n")
            
        if (os.path.exists(os.path.join(folder_name , filename+"_nearest_images_index_and_distance.npy"))):
            candidate_name_file = os.path.join(folder_name , filename+"_nearest_images_index_and_distance.npy")
            candidate_feature_file = os.path.join(folder_name , filename+"_nearest_images.npy")
            query_feature_file = os.path.join(folder_name , filename+"_query_vector.npy")
            query_name_file = os.path.join(folder_name , filename+"_query_image_name.npy")
        
            candidate_name_id = self.load_numpy_array(candidate_name_file)
            candidate_feature = self.load_numpy_array(candidate_feature_file)
            query_feature = self.load_numpy_array(query_feature_file)
            query_name = self.load_numpy_array(query_name_file)
            return candidate_name_id,candidate_feature,query_feature,query_name
        else:
            print("File not there")
        return
        
    """
    Calculates the subject id from given image name
    """
    def get_subject_from_image_id(self, image_id):
        Y = (image_id.split("-"))[2]
        return int(Y)
    
    """
    Calculates the type from given image name
    """
    def get_type_from_image_id(self, image_id):
        X = (image_id.split("-"))[1]
        return str(X)
    
    """
    Calculates the type-weight pairs
    """
    def get_type_weight_pairs(self, image_ids, object_latent_matrix):
        subject_weight = {}
        subject_weight_pair = np.array([])
        visited = np.zeros(len(constants.list_image_types))
        for i in range(0, len(image_ids)):
            Y = self.get_type_from_image_id(image_ids[i])
            if ( visited[constants.list_image_types.index(Y)] == 0 ):
                image_Y_args = []
                for j in range(0, len(image_ids)):
                    Y_inner = self.get_type_from_image_id(image_ids[j])
                    if ( Y == Y_inner ):
                        image_Y_args.append(j)
                
                n = len(image_Y_args)
                subject_weight_pair_inner = []
                for j in range(0, n):
                    subject_weight_pair_inner.append(object_latent_matrix[image_Y_args[j]])
                
                subject_weight_pair_inner = np.array(subject_weight_pair_inner)
                subject_weight_pair_inner = np.mean(subject_weight_pair_inner, axis = 0)
                subject_weight[str(Y)] = subject_weight_pair_inner
                visited[constants.list_image_types.index(Y)] = 1
        
        
        latent_dict = {}
        for i in range(0, object_latent_matrix.shape[1]):
            df = pd.DataFrame(columns = [constants.Y, constants.weight])
            for key in subject_weight.keys():
                df2 = {constants.Y: str(key), constants.weight: subject_weight[key][i]}
                df = df.append(df2, ignore_index = True)
            df = df.sort_values(by = constants.weight, ascending=False)
            subjects = df[constants.Y].to_numpy()
            weights = df[constants.weight].to_numpy()
            concated = []
            for j in range(0, len(subjects)):
                concated.append(str("T:"+subjects[j]) + " W:" + str(round(weights[j], 4)))
            latent_dict["Latent-"+str(i)] = np.array(concated)
        print ("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        self.print_table_from_dict(latent_dict)
        return
    
    """
    Calculates the subject-weight pairs
    """
    def get_subject_weight_pairs(self, image_ids, object_latent_matrix):
        subject_weight = {}
        subject_weight_pair = np.array([])
        visited = np.zeros(constants.Y_range+1)
        for i in range(0, len(image_ids)):
            Y = self.get_subject_from_image_id(image_ids[i])
            if ( visited[Y] == 0 ):
                image_Y_args = []
                for j in range(0, len(image_ids)):
                    Y_inner = self.get_subject_from_image_id(image_ids[j])
                    if ( Y == Y_inner ):
                        image_Y_args.append(j)
                
                n = len(image_Y_args)
                subject_weight_pair_inner = []
                for j in range(0, n):
                    subject_weight_pair_inner.append(object_latent_matrix[image_Y_args[j]])
                
                subject_weight_pair_inner = np.array(subject_weight_pair_inner)
                subject_weight_pair_inner = np.mean(subject_weight_pair_inner, axis = 0)
                subject_weight[str(Y)] = subject_weight_pair_inner
                visited[Y] = 1
        
        
        latent_dict = {}
        for i in range(0, object_latent_matrix.shape[1]):
            df = pd.DataFrame(columns = [constants.Y, constants.weight])
            for key in subject_weight.keys():
                df2 = {constants.Y: str(key), constants.weight: subject_weight[key][i]}
                df = df.append(df2, ignore_index = True)
            df = df.sort_values(by = constants.weight, ascending=False)
            subjects = df[constants.Y].to_numpy()
            weights = df[constants.weight].to_numpy()
            concated = []
            for j in range(0, len(subjects)):
                concated.append(str("S:"+subjects[j]) + " W:" + str(round(weights[j], 4)))
            latent_dict["Latent-"+str(i)] = np.array(concated)
        print ("The following table shows the subject-weight pairs for each latent semantic in decreasing order -->")
        self.print_table_from_dict(latent_dict)
        return
    
    """
    Prints the table for similarity matrix in decreasing order of weights
    """
    def print_table_from_dict(self, latent_dict):
        table = []
        keys = list(latent_dict.keys())
        n = latent_dict[keys[0]].shape[0]
        counter = 0
        for i in range (0,n):
            row_latent = []
            for key in keys:
                row_latent.append(latent_dict[key][i])
            table.append(row_latent)
            counter+=1
        print(tabulate(table, keys, tablefmt="grid"))
        f = open(os.path.join(self.out_path,"output_dictionary.csv"), "w")
        f.seek(0)
        f.write(tabulate(table, keys, tablefmt="grid"))
        f.truncate()
        f.close()
        return
    
    """
    saves the output off k means algorithm in a proper structure
    """
    def save_k_means_output(self, image_ids, features, k, decomposed, task, model_type, extraction_type):
        self.save_numpy_array(decomposed[0], os.path.join(self.out_path , task+"-k-means-"+model_type+'-'+extraction_type+'-'+str(k)+'-lev.npy'))
        self.save_numpy_array(decomposed[1], os.path.join(self.out_path , task+"-k-means-"+model_type+'-'+extraction_type+'-'+str(k)+'-rev.npy'))
        return
    
    def save_output(self, fname):
        return
    
    """
    runs the SVD algorithm and return the 3 decomposed matrices in form of a numpy array
    Output: list of 3 numpy arrays
    """
    def run_SVD(self, features, k):
        svd_instance = SVD()
        decomposed = svd_instance.get_svd_from_scratch(features, int(k))
        return decomposed
    
    """
    runs the PCA algorithm and return the 3 decomposed matrices in form of a numpy array
    Output: list of 3 numpy arrays
    """
    def run_PCA(self, features, k):
        #TODO run SVD and save the output file in proper format
        pca_instance = PCA()
        decomposed = pca_instance.get_PCA(features, int(k))
        return decomposed
    
    """
    runs the LDA algorithm and return the 2 decomposed matrices in form of a numpy array
    Output: list of 2 numpy arrays
    """
    def run_LDA(self, features, k):
        lda_instance = LDA()
        decomposed = lda_instance.get_LDA_components(features, k)
        return decomposed
    
    """
    runs the k means algorithm and return the 2 decomposed matrices in form of a numpy array
    Output: list of 2 numpy arrays
    """
    def run_k_means(self, features, k):
        k_means_instance = KMeans()
        decomposed = k_means_instance.get_k_clusters(features, int(k))
        return decomposed
    
    """
    Saves the numpy array to te given file in the outputs folder
    """
    def save_numpy_array(self, matrix, filename):
        if ( os.path.exists(filename) ):
            os.remove(filename)
        # print ("Saving file "+filename)
        f = open(os.path.join(self.out_path,filename+".csv"), "w")
        f.seek(0)
        f.write(tabulate(matrix, [], tablefmt="grid"))
        f.truncate()
        f.close()
        np.save(filename, matrix)
        return
    
    """
    Loads the file to a numpy array
    Output: Numpy arary of the file
    """
    def load_numpy_array(self, filename):
        if(os.path.exists(filename)):
            matrix = np.load(filename)
        else:
            print ("***********ERROR: PATH DOES NOT EXIST************")
            sys.exit()
        print ("Loading file ",filename)
        return matrix

    def driver_phase3_task4(self):
        obj = task4_phase3.LSH()

        k = int(input("Enter Number of hash functions K in each Layer : "))
        l = int(input("Enter number of layers l : "))
        # vector_file = str(input("Enter the vector file name generated by other task : "))
        images_folder_path = str(input("Enter the images folder path : "))
        self.create_folder_in_file_dict(images_folder_path)
        # folder_list = images_folder_path.split("/")
        # folder_name = folder_list[len(folder_list) -1]
        feature_model = inquirer.prompt(self.feature_models)
        reduction = inquirer.prompt(self.use_reduction)
        reduction_val = 0
        if reduction['use_reduction'] == "yes" or reduction['use_reduction'] == "Yes":
            reduction_val = int(input("Enter the val of k for reduction : "))
        query_image_path = str(input("Input the query image path: "))
        # query_list = query_image_path.split('\\')
        query_list = re.split(r'\\|/', query_image_path)
        query_name = query_list[len(query_list) - 1]
        # print("query name ", query_name)
        q_img = self.read_given_single_image(query_image_path)
        q_image_vector = []
        t = int(input("number of nearest neighbors want to find t :"))


        images = self.read_all_input_images_names_from_folder(images_folder_path)
        if (len(images) == 0):
            print("No images in the given folder.")
            return
        else:
            print("folder found, Extracting features for number of images", len(images))

        image_vectors = []
        decomposed = []
        FM = ColorMoments(images_folder_path)
        if feature_model['feature_model'] == 'Color_Moments':
            q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)
        elif feature_model['feature_model'] == 'HOG':
            q_image_vector = FM.compute_single_image_hog_from_image(q_img)
            # print("Query vector ", q_image_vector)
        elif feature_model['feature_model'] == 'ELBP':
            q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)

        vector_file_name = "Phase_3_task_4_" + str(reduction_val) + "-" + str(feature_model['feature_model'])+".npy"
        reduction_vector_file_name = "Phase_3_task_4_reduction_vector" + str(reduction_val) + "-" + str(feature_model['feature_model']) + ".npy"
        # print("Vector file name is ", vector_file_name)
        if (self.check_if_folder_exists(images_folder_path)) and (self.check_if_file_exists(vector_file_name, images_folder_path)):
            # print("Loading image vector from stored file")
            if reduction_val == 0:
                image_vectors = self.get_file_contents(vector_file_name, images_folder_path)
            else:
                image_vectors = self.get_file_contents(vector_file_name, images_folder_path)
                decomposed = self.get_file_contents(reduction_vector_file_name, images_folder_path)

        else:
            # print("coming inside else part of the python dict")
            if feature_model['feature_model'] == 'Color_Moments':
                image_vectors = FM.compute_image_color_moments(images)
                # q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)
            elif feature_model['feature_model'] == 'HOG':
                image_vectors = FM.compute_image_hog(images)
                # q_image_vector = FM.compute_single_image_hog_from_image(q_img)
            elif feature_model['feature_model'] == 'ELBP':
                image_vectors = FM.compute_image_ELBP(images)
                # q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)

            if reduction['use_reduction'] == "yes" or reduction['use_reduction'] == "Yes":
                temp = self.run_PCA(image_vectors, reduction_val)
                decomposed = temp[2]
                image_vectors = np.matmul(image_vectors, decomposed.T)
                # todo: Save the decomposed matrix[2].
                self.save_file_in_dict(images_folder_path, image_vectors, vector_file_name)
                self.save_file_in_dict(images_folder_path, decomposed, reduction_vector_file_name)
            else:
                # todo: save the image vector
                self.save_file_in_dict(images_folder_path, image_vectors, vector_file_name)

        if reduction['use_reduction'] == "yes" or reduction['use_reduction'] == "Yes":
            # image_vectors = np.matmul(image_vectors, decomposed.T)
            q_image_vector = np.matmul(q_image_vector, decomposed.T)
        # create a decomposed object
        image_vectors = self.transform_image_vectors(image_vectors)
        q_image_vector = self.transform_image_vectors(q_image_vector)

        hash_layers = obj.create_layers(l, k, image_vectors)
        nearest_object_index_list = obj.find_nearest_neighbor(q_image_vector, hash_layers, t, 2)
        distance_dict = obj.find_topK_nearest(image_vectors, q_image_vector, nearest_object_index_list)
        topK_image_index_set = set()
        count = 0
        result = []
        dict_item =[]
        query_image_name = []
        query_image_name.append(str(query_image_path))
        query_image_name_file_name ="phase3_task4_" + str(feature_model['feature_model'])+ "_" + str(reduction_val) +"_query_image_name"+".npy"
        self.save_file_in_dict(images_folder_path, query_image_name, query_image_name_file_name)
        # self.save_numpy_array(query_image_name, os.path.join(self.out_path,
        #                                                      "phase3_task4_query_image_name.npy"))
        for key in distance_dict:
            temp = []
            # temp.append(key)
            temp.append(self.get_image_name(images_folder_path, key))
            temp.append(distance_dict[key])
            dict_item.append(temp)
        if (len(q_image_vector.shape) == 1):
            q_image_vector = q_image_vector.reshape(1, q_image_vector.shape[0])
        query_vector_file_name = "phase3_task4_" + str(feature_model['feature_model'])+ "_" + str(reduction_val) +"_query_vector"+".npy"
        self.save_file_in_dict(images_folder_path, q_image_vector, query_vector_file_name)
        # self.save_numpy_array(q_image_vector, os.path.join(self.out_path,
        #                                                    "phase3_task4_query_vector.npy"))

        nearest_images_index_and_distance_file_name ="phase3_task4_" + str(feature_model['feature_model'])+ "_" + str(reduction_val) +"_nearest_images_index_and_distance"+".npy"
        self.save_file_in_dict(images_folder_path, dict_item, nearest_images_index_and_distance_file_name)
        # test = self.get_file_contents(nearest_images_index_and_distance_file_name, images_folder_path)
        # print("loading nearest image index and distance ", test)
        # self.save_numpy_array(dict_item,
        #                       os.path.join(self.out_path,
        #                                    "phase3_task4_nearest_images_index_and_distance.npy"))
        for key in distance_dict:
            result.append(image_vectors[key])
        result = np.array(result)
        # print("results dimensions: ", result.shape)
        nearest_images_file_name ="phase3_task4_" + str(feature_model['feature_model'])+ "_" + str(reduction_val) +"_nearest_images"+".npy"
        self.save_file_in_dict(images_folder_path, result, nearest_images_file_name)
        # test1 = self.get_file_contents(nearest_images_file_name, images_folder_path)
        # print("loading nearest image ", test1)
        # self.save_numpy_array(result,
        #                       os.path.join(self.out_path,
        #                                    "phase3_task4_nearest_images.npy"))
        for key in distance_dict:
            topK_image_index_set.add(key)
            count +=1
            if(count == t):
                break

        false_positive_rate = self.calculate_false_positive_rate(t, distance_dict)
        miss_rate = self.calculate_miss_rate(q_image_vector, image_vectors, topK_image_index_set)
        # dict_items = distance_dict.items()
        # topK_images = distance_dict[:t]
        # topK_images = list(topK_images.keys())
        # print("top k results are : ", topK_images)
        print("False positive rate is : ", false_positive_rate)
        print("Miss rate is : ", miss_rate)
        print("sized of index structure", obj.get_total_index_size())
        self.print_image_name_from_id(images_folder_path, topK_image_index_set)




    def print_image_name_from_id(self, folder_path, topK_images):
        image_ids = os.listdir(folder_path)
        topK_images = set(topK_images)
        # key_set = set(topK_images.keys())
        # print("key set values : ", topK_images)
        for idx, image_id in enumerate(image_ids):
            if idx in topK_images:
                print("image id ", image_id)

    def get_image_name(self, folder_path, imageid):
        image_ids = os.listdir(folder_path)
        # key_set = set(topK_images.keys())
        for idx, image_id in enumerate(image_ids):
            if idx == imageid:
                return image_id

    
    """
    Transform the image vectors to positive values
    """
    def transform_image_vectors(self, image_vectors):
        image_vectors = np.array(image_vectors)
        min_vectors = image_vectors.min(axis=0)
        min_value = np.min(min_vectors)
        if ( min_value < 0 ):
            image_vectors = image_vectors + (-1*min_value)
        return image_vectors
    
    def load_numpyfile(self, filename):
        if ( os.path.exists(os.path.join(constants.output_file_path, filename+".npy"))):
            numpyarr_file=os.path.join(constants.output_file_path, filename+".npy")
            
            numpy_arr = self.load_numpy_array(numpyarr_file)
            return numpy_arr
        print ("*****ERROR: FILE DOES NOT EXIST*****")
        sys.exit()
        return

    def calculate_false_positive_rate(self, t, distance_dict):
        total = len(distance_dict)
        if total == 0 or ((total - t)/total)*100 < 0:
            return 0
        return ((total - t)/total)*100

    def calculate_miss_rate(self, q_image_vector, image_vectors, topK_image_index_set):
        distance_dict = {}
        t = len(topK_image_index_set)
        if t == 0:
            return 100
        for idx, vector in enumerate(image_vectors):
            dis = distance.euclidean(q_image_vector, vector)
            distance_dict[idx] = dis
        # print("before the closest objects are : ", distance_dict)
        distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        miss_count = 0
        count = 0
        for key in distance_dict:
            if key not in topK_image_index_set:
                miss_count += 1
            count += 1
            if count == t:
                break
        miss_count = (miss_count/t)*100
        return miss_count




"""
Main function to drive the execution of the entire code
"""
def main():
    phase3 = TaskDriverPhase3()
    phase3.start_driver()
    return

if __name__=="__main__":
    main()




