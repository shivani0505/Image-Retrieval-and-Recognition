# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 00:04:40 2021
@author: rchityal
"""
import os
from PIL import Image
import numpy as np
import pickle
import numpy as np
import constants

def get_type_images(input_folder):

    image_types = ["cc", "con", "detail", "emboss", "jitter", "neg", 
                   "noise1", "noise2", "original", "poster", "rot", 
                   "smooth", "stipple"]

    all_type_images = []
    for image_type in image_types:
        type_images = get_images_specific_type(input_folder, image_type)
        type_images = np.asarray(type_images)
        average_type_images = type_images.mean(0)
        all_type_images.append(average_type_images)

    return all_type_images


def get_images_all(input_folder):

    input_images = {}

    for index, filename in enumerate(os.listdir(input_folder)):
        filename = input_folder + "\\" + filename
        img = Image.open(filename)
        img = img.convert("L")
        img = np.asarray(img)
        img = img.tolist()
        input_images[filename] = img

    return input_images

# def aggregate_type_images(images):

#     for key in images:
#         file_name = key.split("\\")
#         file_name = file_name[-1]
#         image_type




def get_images_specific_type(input_folder, image_type):
    input_images = {}

    for index, filename in enumerate(os.listdir(input_folder)):
        check_image_type = filename.split("-")[1]
        if (check_image_type == image_type) :
            filename = input_folder + "\\" + filename
            img = Image.open(filename)
            img = img.convert("L")
            img = np.asarray(img)
            img = img.tolist()
            input_images[filename] = img
    # input_images = np.array(input_images)
    return input_images


def pickle_dump(file_name,images, eigen_values, eigen_vectors):
    with open(file_name, 'wb') as f:
        pickle.dump(images, f)
        pickle.dump(eigen_values, f)
        pickle.dump(eigen_vectors, f)


def convert_similarity_similarity_matrix_to_dict(numpy_arr):
    similarity_matrix={}
    print('shape of input matrix is')
    print(str(numpy_arr.shape[0])+","+str(numpy_arr.shape[1]))
    for i in range(0,numpy_arr.shape[0]):
        file_name=str(i+1)
        edge={}
        for j in range(0,numpy_arr.shape[1]):
            target_file_name=str(j+1)
            edge.update({target_file_name:numpy_arr[i][j]})
        similarity_matrix.update({file_name:edge})

    return similarity_matrix









