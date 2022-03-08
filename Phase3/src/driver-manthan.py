#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 09:06:19 2021

@author: khushalmodi
"""
import os
import re
import constants
from color_moments import ColorMoments
from FM_QueryImage import Q_Img_FM
from Top_K_Images import Top_K_Img
import inquirer
from svd import SVD
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
import task6, task7
import task4_phase3


class TaskDriver:

    def __init__(self):
        self.ph3_task4 = "phase 3 task 4"
        self.data_path = constants.input_path
        self.out_path = constants.output_path
        self.task1 = "Task1: extract the subject-weight latent semantic pairs."
        self.task2 = "Task2: extract the type-weight latent semantic pairs."
        self.task3 = "Task3: create type-type similarity matrix, perform the dimensionality reduction and get top k latent semantics"
        self.task4 = "Task4: reate subject-subject similarity matrix, perform the dimensionality reduction and get top k latent semantics"
        self.task5 = "Task5: identify and visualize most similar n images under selected latent semantics."
        self.task6 = "Task6: associate a type label (X) to the image under the selected latent semantics."
        self.task7 = "Task7: associates a subject ID (Y ) to the image under the selected latent semantics."
        self.task8 = "Task8: create a similarity graph, identify the most significant m subjects in the collection using ASCOS++ measure."
        self.task9 = "Task9: create a similarity graph, identifies the most significant m subjects (relative to the input subjects) using personalized PageRank measure."
        self.questions = [
            inquirer.List('task',
                          message="What task do you need?",
                          choices=[self.ph3_task4, self.task1, self.task2,
                                   self.task3, self.task4,
                                   self.task5, self.task6,
                                   self.task7, self.task8,
                                   self.task9,
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
        self.use_reduction = [
            inquirer.List('use_reduction',
                          message="Do you want to use reduction method? ",
                          choices=constants.yes_no_choice,
                          ),
        ]

    def find_index_of_query_image(self, folder_path, query_name):
        image_ids = os.listdir(folder_path)
        for idx, image_id in enumerate(image_ids):
            print(image_id, "\n")
            if image_id == query_name:
                print("image id for image ", query_name, "is ", idx)
                return idx

        return -1

    """
    extract all images of type X from the database
    Output: List of all the images of type X
    """

    def extract_all_X_images(self, X):
        files = os.listdir(self.data_path)
        X_files = [i for i in files if re.search(r"\w*(" + str(X) + ")\w*", i)]
        return X_files

    """
    extract all images of subject ID Y from the database
    Output: List of all the images of subject ID Y
    """

    def extract_all_Y_images(self, Y):
        files = os.listdir(self.data_path)
        Y_files = [i for i in files if re.search(r"(image)-\w*-(" + str(Y) + ")-\w*", i)]
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

    """
    Drives the flow for task 1
    """

    def driver_task1(self):
        feature_model = inquirer.prompt(self.feature_models)
        image_X = inquirer.prompt(self.image_type)
        red_tech = inquirer.prompt(self.reduction_tech)
        k = input("Input the value of k: ")

        # get all the X images
        images = self.extract_all_X_images(image_X['image_type'])
        if (len(images) == 0):
            print("No such images present in the DB")
            return

        # create a feature model object and image vectors
        FM = ColorMoments(self.data_path)
        image_vectors = []

        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)

        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)

        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)

        # create a decomposed object
        image_vectors = self.transform_image_vectors(image_vectors)
        decomposed = []
        # apply proper reduction technique
        if (red_tech['reduction_technique'] == 'SVD'):
            decomposed = self.run_SVD(image_vectors, k)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task1-SVD-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task1-SVD-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task1-SVD-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'PCA'):
            decomposed = self.run_PCA(image_vectors, k)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task1-PCA-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task1-PCA-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task1-PCA-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'LDA'):
            decomposed = self.run_LDA(image_vectors, k)
            print(decomposed[0].shape, decomposed[1].shape)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task1-LDA-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task1-LDA-" + str(feature_model['feature_model']) + '-' + image_X[
                                                   'image_type'] + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'k-means'):
            decomposed = self.run_k_means(image_vectors, k)
            self.save_k_means_output(images, image_vectors, k, decomposed, "task1", str(feature_model['feature_model']),
                                     image_X['image_type'])

        self.get_subject_weight_pairs(images, decomposed[0])
        return

    """
    Drives the flow for task 2
    """

    def driver_task2(self):
        feature_model = inquirer.prompt(self.feature_models)
        image_Y = input("Input the subject ID: ")
        red_tech = inquirer.prompt(self.reduction_tech)
        k = input("Input the value of k: ")

        # get all the X images
        images = self.extract_all_Y_images(str(image_Y))
        if (len(images) == 0):
            print("No such images present in the DB")
            return

        # create a feature model object and image vectors
        FM = ColorMoments(self.data_path)
        image_vectors = []

        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)

        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)

        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)

        # create a decomposed object
        image_vectors = self.transform_image_vectors(image_vectors)
        decomposed = []
        # apply proper reduction technique
        if (red_tech['reduction_technique'] == 'SVD'):
            decomposed = self.run_SVD(image_vectors, k)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task2-SVD-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task2-SVD-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task2-SVD-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'PCA'):
            decomposed = self.run_PCA(image_vectors, k)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task2-PCA-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task2-PCA-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task2-PCA-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'LDA'):
            decomposed = self.run_LDA(image_vectors, k)
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task2-LDA-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task2-LDA-" + str(feature_model['feature_model']) + '-' + str(
                                                   image_Y) + "-" + str(k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'k-means'):
            decomposed = self.run_k_means(image_vectors, k)
            self.save_k_means_output(images, image_vectors, k, decomposed, "task2", str(feature_model['feature_model']),
                                     str(image_Y))
        self.get_type_weight_pairs(images, decomposed[0])
        return

    """
    Drives the flow for task 3
    """

    def driver_task3(self):
        feature_model = inquirer.prompt(self.feature_models)
        red_tech = inquirer.prompt(self.reduction_tech)
        k = input("Input the value of k: ")

        # create a feature model object and image vectors
        FM = ColorMoments(self.data_path)
        type_image_vectors = []

        # get type image vectors by average the image vectors for all the
        # images of same type
        image_type_list = []
        for image_type in constants.list_image_types:
            # get all the images of same type

            images = self.extract_all_X_images(image_type)

            if len(images) == 0:
                continue
            image_type_list.append(image_type)
            # extract image vectors for each image of same type
            image_vectors = []
            if (feature_model['feature_model'] == 'Color_Moments'):
                image_vectors = FM.compute_image_color_moments(images)


            elif (feature_model['feature_model'] == 'HOG'):
                image_vectors = FM.compute_image_hog(images)

            elif (feature_model['feature_model'] == 'ELBP'):
                image_vectors = FM.compute_image_ELBP(images)

            image_vectors = self.transform_image_vectors(image_vectors)
            # print("images type")
            # print(image_type)
            # print(image_vectors.shape)
            # get the average of all the image vectors of same type
            image_vectors = image_vectors.mean(0)

            type_image_vectors.append(image_vectors)

        type_image_vectors = np.asarray(type_image_vectors)

        # print(type_image_vectors.shape)

        type_type_similarity_matrix = np.matmul(type_image_vectors, type_image_vectors.T)

        # create a decomposed object
        decomposed = []

        image_vectors = type_type_similarity_matrix

        # apply proper reduction technique
        if (red_tech['reduction_technique'] == 'SVD'):
            decomposed = self.run_SVD(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task3-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(type_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task3-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task3-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task3-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task3-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'PCA'):
            decomposed = self.run_PCA(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task3-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(type_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task3-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task3-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task3-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task3-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'LDA'):
            decomposed = self.run_LDA(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task3-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(type_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task3-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task3-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task3-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'k-means'):
            decomposed = self.run_k_means(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task3-k-means-" + str(feature_model['feature_model']) + '-type-' + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(type_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task3-k-means-" + str(feature_model['feature_model']) + '-type-' + str(
                                                   k) + '-transform-1.npy'))
            self.save_k_means_output(images, image_vectors, k, decomposed, "task3", str(feature_model['feature_model']),
                                     "type")

        # TODO get type weight pairs for each latent semantics
        self.get_type_weight_pairs_task_3(image_type_list, decomposed[0])
        return

    """
    Drives the flow for task 4
    """

    def driver_task4(self):
        feature_model = inquirer.prompt(self.feature_models)
        red_tech = inquirer.prompt(self.reduction_tech)
        k = input("Input the value of k: ")
        # create a feature model object and image vectors
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
            if (feature_model['feature_model'] == 'Color_Moments'):
                image_vectors = FM.compute_image_color_moments(images)


            elif (feature_model['feature_model'] == 'HOG'):
                image_vectors = FM.compute_image_hog(images)

            elif (feature_model['feature_model'] == 'ELBP'):
                image_vectors = FM.compute_image_ELBP(images)

            image_vectors = self.transform_image_vectors(image_vectors)
            # get the average of all the image vectors of same type
            image_vectors = image_vectors.mean(0)

            subject_image_vectors.append(image_vectors)

        subject_image_vectors = np.asarray(subject_image_vectors)

        subject_subject_similarity_matrix = np.matmul(subject_image_vectors, subject_image_vectors.T)

        # create a decomposed object
        decomposed = []

        image_vectors = subject_subject_similarity_matrix
        # apply proper reduction technique
        if (red_tech['reduction_technique'] == 'SVD'):
            decomposed = self.run_SVD(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task4-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task4-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task4-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task4-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task4-SVD-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'PCA'):
            decomposed = self.run_PCA(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task4-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task4-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task4-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task4-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-ll.npy'))
            self.save_numpy_array(decomposed[2],
                                  os.path.join(self.out_path,
                                               "task4-PCA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'LDA'):
            decomposed = self.run_LDA(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path,
                                               "task4-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T,
                                  os.path.join(self.out_path,
                                               "task4-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-transform-1.npy'))
            self.save_numpy_array(decomposed[0],
                                  os.path.join(self.out_path,
                                               "task4-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-lev.npy'))
            self.save_numpy_array(decomposed[1],
                                  os.path.join(self.out_path,
                                               "task4-LDA-" + str(feature_model['feature_model']) + "-" + str(
                                                   k) + '-rev.npy'))
        elif (red_tech['reduction_technique'] == 'k-means'):
            decomposed = self.run_k_means(image_vectors, k)
            self.save_numpy_array(image_vectors,
                                  os.path.join(self.out_path, "task4-k-means-" + str(
                                      feature_model['feature_model']) + '-subject-' + str(k) + '-similarity.npy'))
            self.save_numpy_array(subject_image_vectors.T,
                                  os.path.join(self.out_path, "task4-k-means-" + str(
                                      feature_model['feature_model']) + '-subject-' + str(k) + '-transform-1.npy'))
            self.save_k_means_output(images, image_vectors, k, decomposed, "task4", str(feature_model['feature_model']),
                                     "subject")

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
        if (os.path.exists(os.path.join(self.out_path, filename + "-transform-1.npy"))):
            transform_file = self.load_numpy_array(os.path.join(self.out_path, filename + "-transform-1.npy"))
            return True, transform_file
        return False, None

    """
    Drives the flow for task 5
    """

    def driver_task5(self):
        query_image = input("Input the query image path: ")
        latent_sem_file = input(
            "Input the latent semantics filename in the form of (task)-(reduction technique)-(feature model)-(type/ID)-(k)e.g. task1-PCA-HOG-emboss-5: ")
        n = input("Give the value of n to get n similar images: ")
        decomposed = self.load_latent_semantics(latent_sem_file)
        core_matrix = self.load_core_matrix(latent_sem_file, decomposed[0].shape[1])
        transformation_matrix = self.load_transformation_matrix(latent_sem_file)
        feature_model = inquirer.prompt(self.feature_models)

        # TODO call into the task 5 functions properly
        images = self.read_all_input_images_names_from_folder(self.data_path)
        # transform the query image
        q_img = self.read_given_single_image(query_image)
        FM = ColorMoments(self.data_path)
        image_vectors = []
        q_image_vector = []

        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
            q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)

        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)
            q_image_vector = FM.compute_single_image_hog_from_image(q_img)

        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)
            q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)

        # apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        q_image_vector = self.transform_image_vectors(q_image_vector)
        # transform the images to original space first
        if (transformation_matrix[0]):
            q_image_vector = np.matmul(q_image_vector, transformation_matrix[1])
            image_vectors = np.matmul(image_vectors, transformation_matrix[1])

        # transform the images
        transformed_images = np.matmul(image_vectors, decomposed[1].T)
        q_transformed_image = np.matmul(q_image_vector, decomposed[1].T)

        # take the core matrix into consideration
        transformed_images = np.matmul(transformed_images, core_matrix)
        q_transformed_image = np.matmul(q_transformed_image, core_matrix)

        image_np_arr = Image.open(query_image)
        image_np_arr = np.array(image_np_arr)

        Task5 = Top_K_Img(self.data_path, image_np_arr.flatten())
        Task5.compute_score(q_transformed_image, transformed_images, images, feature_model, n)

        return

    """
    loads the latent semantics and returns a list of all numpy arrays of decomposed matrix
    """

    def load_core_matrix(self, filename, k):
        if (os.path.exists(os.path.join(self.out_path, filename + "-ll.npy"))):
            core_file = os.path.join(self.out_path, filename + "-ll.npy")
            core_matrix = self.load_numpy_array(core_file)
            return core_matrix
        else:
            core_matrix = np.diag(np.ones(k))
            return core_matrix

    """
    Drives the flow for task 6
    """

    def driver_task6(self):
        query_image = input("Input the query image path: ")
        latent_sem_file = input(
            "Input the latent semantics filename in the form of (task)-(reduction technique)-(feature model)-(type/ID)-(k)e.g. task1-PCA-HOG-emboss-5: ")
        decomposed = self.load_latent_semantics(latent_sem_file)
        core_matrix = self.load_core_matrix(latent_sem_file, decomposed[0].shape[1])
        transformation_matrix = self.load_transformation_matrix(latent_sem_file)
        feature_model = inquirer.prompt(self.feature_models)
        # print(decomposed[0].shape, decomposed[1].shape)

        # TODO call into the task 5 functions properly
        images = self.read_all_input_images_names_from_folder(self.data_path)
        # transform the query image
        q_img = self.read_given_single_image(query_image)
        FM = ColorMoments(self.data_path)
        image_vectors = []
        q_image_vector = []

        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
            q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)

        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)
            q_image_vector = FM.compute_single_image_hog_from_image(q_img)
            q_image_vector = q_image_vector.reshape(1, q_image_vector.shape[0])

        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)
            q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)
            q_image_vector = q_image_vector.reshape(1, q_image_vector.shape[0])

        # apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        q_image_vector = self.transform_image_vectors(q_image_vector)

        # transform the images to original space first
        if (transformation_matrix[0]):
            q_image_vector = np.matmul(q_image_vector, transformation_matrix[1])
            image_vectors = np.matmul(image_vectors, transformation_matrix[1])

        # transform the images
        transformed_images = np.matmul(image_vectors, decomposed[1].T)
        q_transformed_image = np.matmul(q_image_vector, decomposed[1].T)

        # take the core matrix into consideration
        transformed_images = np.matmul(transformed_images, core_matrix)
        q_transformed_image = np.matmul(q_transformed_image, core_matrix)

        # task6.task_to_get_Type(transformed_images, images, q_transformed_image, feature_model['feature_model'])
        image_type = task6.task_to_get_Type(transformed_images, images, q_transformed_image, feature_model)
        print("Type of the image is most likely: ", image_type)
        return

    """
    Drives the flow for task 7
    """

    def driver_task7(self):
        query_image = input("Input the query image path: ")
        latent_sem_file = input(
            "Input the latent semantics filename in the form of (task)-(reduction technique)-(feature model)-(type/ID)-(k)e.g. task1-PCA-HOG-emboss-5: ")
        decomposed = self.load_latent_semantics(latent_sem_file)
        core_matrix = self.load_core_matrix(latent_sem_file, decomposed[0].shape[1])
        transformation_matrix = self.load_transformation_matrix(latent_sem_file)
        feature_model = inquirer.prompt(self.feature_models)
        # print(decomposed[0].shape, decomposed[1].shape)

        # TODO call into the task 5 functions properly
        images = self.read_all_input_images_names_from_folder(self.data_path)
        # transform the query image
        q_img = self.read_given_single_image(query_image)
        FM = ColorMoments(self.data_path)
        image_vectors = []
        q_image_vector = []

        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
            q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)

        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)
            q_image_vector = FM.compute_single_image_hog_from_image(q_img)
            q_image_vector = q_image_vector.reshape(1, q_image_vector.shape[0])

        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)
            q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)
            q_image_vector = q_image_vector.reshape(1, q_image_vector.shape[0])

        # apply min value conversion transformation
        image_vectors = self.transform_image_vectors(image_vectors)
        q_image_vector = self.transform_image_vectors(q_image_vector)

        # transform the images to original space first
        if (transformation_matrix[0]):
            q_image_vector = np.matmul(q_image_vector, transformation_matrix[1])
            image_vectors = np.matmul(image_vectors, transformation_matrix[1])

        # transform the images
        transformed_images = np.matmul(image_vectors, decomposed[1].T)
        q_transformed_image = np.matmul(q_image_vector, decomposed[1].T)

        # take the core matrix into consideration
        transformed_images = np.matmul(transformed_images, core_matrix)
        q_transformed_image = np.matmul(q_transformed_image, core_matrix)

        task7.task_to_get_subject(transformed_images, images, q_transformed_image, feature_model)
        return

    """
    Drives the flow for task 8
    """

    def driver_task8(self):
        n = input("Input the value of n")
        m = input("Input the value of m")

        S_S_WeightMatrix_file = input(
            "Input the similarity matrix filename in form (task)-(reduction technique)-(feature model)-(k)e.g. task1-PCA-HOG-5: ")
        S_S_WeightMatrix = self.load_numpy_array(os.path.join(self.out_path, S_S_WeightMatrix_file + "-similarity.npy"))
        Task8_instance = Task8()
        S_S_SimilarityMatrix = utils.convert_similarity_similarity_matrix_to_dict(S_S_WeightMatrix)
        formedGraph = Task8_instance.task8_Subtask1(S_S_SimilarityMatrix, n, m)
        s = ""
        fname = constants.output_path + "/Task8-Graph.txt"
        with open(fname, 'w') as ofile:
            for i in formedGraph:
                s += "\n\n"
                s += "Node:"
                s += str(i)
                s += "\n"
                s += "Neighbors->"
                for j in formedGraph[i]:
                    s += "{Node: "
                    s += str(j)
                    s += " Strength: "
                    s += str(formedGraph[i][j])
                    s += "}  "
            ofile.write(s)
            ofile.close()
        topNodes = Task8_instance.AscosPlus(formedGraph, m)
        st = ""
        fname = constants.output_path + "/Task8-top-m-nodes.txt"
        with open(fname, 'w') as ofile:
            for i in topNodes:
                st += "\n\n"
                st += "Node: "
                st += str(i)
                st += " Strength: "
                st += str(topNodes[i])
            ofile.write(st)
            ofile.close()
        return

    """
    Drives the flow for task 9
    """

    def driver_task9(self):
        sim_file = input(
            "Input the similarity matrix filename in form (task)-(reduction technique)-(feature model)-(k)e.g. task1-PCA-HOG-5: ")
        n = input("input value of n: ")
        m = input("input value of m: ")
        subject_id = num_list = list(
            int(num) for num in input("Enter the subject IDs separated by a space: ").strip().split())[:3]
        sim_file = self.load_numpy_array(os.path.join(self.out_path, sim_file + "-similarity.npy"))
        object = personalizePageRank.ppr()
        similarityGraph = object.convertToGraph(sim_file, n, m)
        result = object.findPersonalizedRank(subject_id[0], subject_id[1], subject_id[2], similarityGraph, 0.75)
        print("page rank vector - ")
        print(result)
        idx = np.argsort(result)[::-1]
        for i in range(0, int(m)):
            print("Subject id :", idx[i] + 1)
        self.save_numpy_array(similarityGraph, os.path.join(self.out_path, "task9-similarity-graph.npy"))
        return

    def get_type_weight_pairs_task_3(self, image_types, decomposed):
        latent_dict = {}
        print("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        for i in range(0, (decomposed.shape[1])):
            type_weights = []
            for j in range(0, decomposed.shape[0]):
                type_weights.append(decomposed[j][i])
            type_weight_pairs = zip(type_weights, image_types)
            type_weight_pairs = sorted(type_weight_pairs, reverse=True)
            tuples = zip(*type_weight_pairs)
            type_weight, image_types_temp = [list(tuple) for tuple in tuples]
            concated = []
            for indx, image_type in enumerate(image_types_temp):
                concated.append(str("Type:") + image_type + " W:" + str(round(type_weight[indx], 4)))
                # print(image_type)
                # print(type_weight[indx])
            latent_dict["Latent-" + str(i)] = np.array(concated)
        self.print_table_from_dict(latent_dict)

    def get_subject_weight_pairs_task_3(self, subjects, decomposed):

        latent_dict = {}
        print("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        for i in range(0, (decomposed.shape[1])):
            subject_weight = []
            for j in range(0, decomposed.shape[0]):
                subject_weight.append(decomposed[j][i])
            subject_weight_pairs = zip(subject_weight, subjects)
            subject_weight_pairs = sorted(subject_weight_pairs, reverse=True)
            tuples = zip(*subject_weight_pairs)
            subject_weight, subject_temp = [list(tuple) for tuple in tuples]
            concated = []
            for indx, subject in enumerate(subject_temp):
                concated.append(str("S: ") + str(subject) + " W:" + str(round(subject_weight[indx], 4)))
                # print(subject)
                # print(subject_weight[indx])
            latent_dict["Latent-" + str(i)] = np.array(concated)
        self.print_table_from_dict(latent_dict)
        return

    """
    Starts the driver code
    """

    def start_driver(self):
        while (True):
            run_task = inquirer.prompt(self.questions)
            task = run_task["task"]
            if (task == "exit"):
                break
            elif (task == "change data path"):
                self.change_source_path()
            elif (task == "change output path"):
                self.change_output_path();
            elif (task == self.task1):
                self.driver_task1()
            elif (task == self.task2):
                self.driver_task2()
            elif (task == self.task3):
                self.driver_task3()
            elif (task == self.task4):
                self.driver_task4();
            elif (task == self.task5):
                self.driver_task5()
            elif (task == self.task6):
                self.driver_task6()
            elif (task == self.task7):
                self.driver_task7()
            elif (task == self.task8):
                self.driver_task8();
            elif (task == self.task9):
                self.driver_task9()
            elif task == self.ph3_task4:
                self.driver_phase3_task4()
            else:
                print("Invalid choice")
            print("\n\n")

    """
    loads the latent semantics and returns a list of all numpy arrays of decomposed matrix
    """

    def load_latent_semantics(self, filename):
        if (os.path.exists(os.path.join(self.out_path, filename + "-rev.npy"))):
            rev_file = os.path.join(self.out_path, filename + "-rev.npy")
            lev_file = os.path.join(self.out_path, filename + "-lev.npy")
            rev = self.load_numpy_array(rev_file)
            lev = self.load_numpy_array(lev_file)
            return lev, rev
        print("*****ERROR: FILE DOES NOT EXIST*****")
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
            if (visited[constants.list_image_types.index(Y)] == 0):
                image_Y_args = []
                for j in range(0, len(image_ids)):
                    Y_inner = self.get_type_from_image_id(image_ids[j])
                    if (Y == Y_inner):
                        image_Y_args.append(j)

                n = len(image_Y_args)
                subject_weight_pair_inner = []
                for j in range(0, n):
                    subject_weight_pair_inner.append(object_latent_matrix[image_Y_args[j]])

                subject_weight_pair_inner = np.array(subject_weight_pair_inner)
                subject_weight_pair_inner = np.mean(subject_weight_pair_inner, axis=0)
                subject_weight[str(Y)] = subject_weight_pair_inner
                visited[constants.list_image_types.index(Y)] = 1

        latent_dict = {}
        for i in range(0, object_latent_matrix.shape[1]):
            df = pd.DataFrame(columns=[constants.Y, constants.weight])
            for key in subject_weight.keys():
                df2 = {constants.Y: str(key), constants.weight: subject_weight[key][i]}
                df = df.append(df2, ignore_index=True)
            df = df.sort_values(by=constants.weight, ascending=False)
            subjects = df[constants.Y].to_numpy()
            weights = df[constants.weight].to_numpy()
            concated = []
            for j in range(0, len(subjects)):
                concated.append(str("T:" + subjects[j]) + " W:" + str(round(weights[j], 4)))
            latent_dict["Latent-" + str(i)] = np.array(concated)
        print("The following table shows the type-weight pairs for each latent semantic in decreasing order -->")
        self.print_table_from_dict(latent_dict)
        return

    """
    Calculates the subject-weight pairs
    """

    def get_subject_weight_pairs(self, image_ids, object_latent_matrix):
        subject_weight = {}
        subject_weight_pair = np.array([])
        visited = np.zeros(constants.Y_range + 1)
        for i in range(0, len(image_ids)):
            Y = self.get_subject_from_image_id(image_ids[i])
            if (visited[Y] == 0):
                image_Y_args = []
                for j in range(0, len(image_ids)):
                    Y_inner = self.get_subject_from_image_id(image_ids[j])
                    if (Y == Y_inner):
                        image_Y_args.append(j)

                n = len(image_Y_args)
                subject_weight_pair_inner = []
                for j in range(0, n):
                    subject_weight_pair_inner.append(object_latent_matrix[image_Y_args[j]])

                subject_weight_pair_inner = np.array(subject_weight_pair_inner)
                subject_weight_pair_inner = np.mean(subject_weight_pair_inner, axis=0)
                subject_weight[str(Y)] = subject_weight_pair_inner
                visited[Y] = 1

        latent_dict = {}
        for i in range(0, object_latent_matrix.shape[1]):
            df = pd.DataFrame(columns=[constants.Y, constants.weight])
            for key in subject_weight.keys():
                df2 = {constants.Y: str(key), constants.weight: subject_weight[key][i]}
                df = df.append(df2, ignore_index=True)
            df = df.sort_values(by=constants.weight, ascending=False)
            subjects = df[constants.Y].to_numpy()
            weights = df[constants.weight].to_numpy()
            concated = []
            for j in range(0, len(subjects)):
                concated.append(str("S:" + subjects[j]) + " W:" + str(round(weights[j], 4)))
            latent_dict["Latent-" + str(i)] = np.array(concated)
        print("The following table shows the subject-weight pairs for each latent semantic in decreasing order -->")
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
        for i in range(0, n):
            row_latent = []
            for key in keys:
                row_latent.append(latent_dict[key][i])
            table.append(row_latent)
            counter += 1
        print(tabulate(table, keys, tablefmt="grid"))
        f = open(os.path.join(self.out_path, "output_dictionary.csv"), "w")
        f.seek(0)
        f.write(tabulate(table, keys, tablefmt="grid"))
        f.truncate()
        f.close()
        return

    """
    saves the output off k means algorithm in a proper structure
    """

    def save_k_means_output(self, image_ids, features, k, decomposed, task, model_type, extraction_type):
        self.save_numpy_array(decomposed[0], os.path.join(self.out_path,
                                                          task + "-k-means-" + model_type + '-' + extraction_type + '-' + str(
                                                              k) + '-lev.npy'))
        self.save_numpy_array(decomposed[1], os.path.join(self.out_path,
                                                          task + "-k-means-" + model_type + '-' + extraction_type + '-' + str(
                                                              k) + '-rev.npy'))
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
        # TODO run SVD and save the output file in proper format
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
        if (os.path.exists(filename)):
            os.remove(filename)
        print("Saving file " + filename)
        f = open(os.path.join(self.out_path, filename + ".csv"), "w")
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
        if (os.path.exists(filename)):
            matrix = np.load(filename)
        else:
            print("***********ERROR: PATH DOES NOT EXIST************")
            sys.exit()
        print("Loading file ", filename)
        return matrix

    """
    Transform the image vectors to positive values
    """

    def transform_image_vectors(self, image_vectors):
        image_vectors = np.array(image_vectors)
        min_vectors = image_vectors.min(axis=0)
        min_value = np.min(min_vectors)
        if (min_value < 0):
            image_vectors = image_vectors + (-1 * min_value)
        return image_vectors

    def load_numpyfile(self, filename):
        if (os.path.exists(os.path.join(constants.output_file_path, filename + ".npy"))):
            numpyarr_file = os.path.join(constants.output_file_path, filename + ".npy")

            numpy_arr = self.load_numpy_array(numpyarr_file)
            return numpy_arr
        print("*****ERROR: FILE DOES NOT EXIST*****")
        sys.exit()
        return

    def driver_phase3_task4(self):
        obj = task4_phase3.LSH()
        k = int(input("Enter Number of hash functions K in each Layer : "))
        l = int(input("Enter number of layers l : "))
        vector_file = str(input("Enter the vector file name generted by other task : "))
        images_folder_path = str(input("Enter the images folder path : "))
        folder_list = images_folder_path.split("/")
        folder_name = folder_list[len(folder_list) -1]
        feature_model = inquirer.prompt(self.feature_models)
        reduction = inquirer.prompt(self.use_reduction)
        reduction_val = 0
        if reduction['use_reduction'] == "yes" or reduction['use_reduction'] == "Yes":
            reduction_val = int(input("Enter the val of k for reduction : "))
        query_image_path = str(input("Input the query image path: "))
        query_list = query_image_path.split('\\')
        query_name = query_list[len(query_list) - 1]
        print("query name ", query_name)
        q_img = self.read_given_single_image(query_image_path)
        q_image_vector = []
        t = int(input("number of nearest neighbors want to find t :"))


        images = self.read_all_input_images_names_from_folder(images_folder_path)
        if (len(images) == 0):
            print("No images in the given folder.")
            return
        else:
            print("folder found, Extracting features for number of images", len(images))

        FM = ColorMoments(images_folder_path)
        image_vectors = []
        if (feature_model['feature_model'] == 'Color_Moments'):
            image_vectors = FM.compute_image_color_moments(images)
            q_image_vector = FM.compute_single_image_color_moments_from_image(q_img)
        elif (feature_model['feature_model'] == 'HOG'):
            image_vectors = FM.compute_image_hog(images)
            q_image_vector = FM.compute_single_image_hog_from_image(q_img)
        elif (feature_model['feature_model'] == 'ELBP'):
            image_vectors = FM.compute_image_ELBP(images)
            q_image_vector = FM.compute_single_image_ELBP_from_image(q_img)


        # create a decomposed object
        image_vectors = self.transform_image_vectors(image_vectors)
        q_image_vector = self.transform_image_vectors(q_image_vector)

        # apply proper reduction technique
        if reduction['use_reduction'] == "yes" or reduction['use_reduction'] == "Yes":
                decomposed = self.run_PCA(image_vectors, reduction_val)
                image_vectors = np.matmul(image_vectors, decomposed[2].T)
                q_image_vector = np.matmul(q_image_vector, decomposed[2].T)
        else:
            # Todo: write code to store features in file.
            # print("coming inside the orginal feature space")
            self.save_numpy_array(image_vectors, os.path.join(self.out_path, "phase 3 task 4 " + str(
                feature_model['feature_model']) + '.npy'))
            # print("dimentions of the input image vector.", np.array(image_vectors).shape)

        hash_layers = obj.create_layers(l, k, image_vectors)
        nearest_object_index_list = obj.find_nearest_neighbor(q_image_vector, hash_layers, t, 2)
        distance_dict = obj.find_topK_nearest(image_vectors, q_image_vector, nearest_object_index_list)
        topK_image_index = set()
        count = 0
        result = []
        dict_item =[]
        query_image_name = []
        query_image_name.append(str(query_name))
        self.save_numpy_array(query_image_name, os.path.join(self.out_path,
                                                           "phase3_task4_query_image_name.npy"))
        for key in distance_dict:
            temp = []
            temp.append(key)
            temp.append(self.get_image_name(images_folder_path, key))
            temp.append(distance_dict[key])
            dict_item.append(temp)
        self.save_numpy_array(q_image_vector, os.path.join(self.out_path,
                                           "phase3_task4_query_vector.npy"))
        self.save_numpy_array(dict_item,
                              os.path.join(self.out_path,
                                           "phase3_task4_nearest_images_index_and_distance.npy"))
        for key in distance_dict:
            result.append(image_vectors[key])
        result = np.array(result)
        print("results dimentions: ", result.shape)
        self.save_numpy_array(result,
                              os.path.join(self.out_path,
                                           "phase3_task4_nearest_images.npy"))
        for key in distance_dict:
            topK_image_index.add(key)
            count +=1
            if(count == t):
                break
        # dict_items = distance_dict.items()
        # topK_images = distance_dict[:t]
        # topK_images = list(topK_images.keys())
        # print("top k results are : ", topK_images)
        self.print_image_name_from_id(images_folder_path, topK_image_index)

    def print_image_name_from_id(self, folder_path, topK_images):
        image_ids = os.listdir(folder_path)
        topK_images = set(topK_images)
        # key_set = set(topK_images.keys())
        print("key set values : ", topK_images)
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
Main function to drive the execution of the entire code
"""


def main():
    phase2 = TaskDriver()
    phase2.start_driver()
    return


if __name__ == "__main__":
    main()
