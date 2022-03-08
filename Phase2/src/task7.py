from numpy.lib.type_check import imag
from color_moments import ColorMoments
import numpy as np
from scipy.spatial import distance
import numpy
import scipy
from skimage.feature import hog
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from scipy.stats import wasserstein_distance

# Define color moments and hog function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import task6
import sys

def task_to_get_subject(transformed_images, images, q_transformed_image, model):  

    score ={}
    representative={}
    representativeData={}
    representativeOrient={}
    representativeDataObject ={}
    counter = 0
    
    
    queryType = task6.task_to_get_Type(transformed_images, images, q_transformed_image, model)
    model = model['feature_model']
    for i in range(len(images)):
        current_image = images[i]
        current_data = transformed_images[i]
        _, type, subject, orient = current_image.split("-")
        if type == queryType:

            if subject in representative.keys():
                representative[subject].append(current_image)
                representativeData[subject].append(current_data)
                
            else:
                representative[subject] = [current_image]
                representativeData[subject]=[ current_data ]
        counter = counter+1
    
    for key in representativeData:
        
        # image = representativeData[key]
        for image in representativeData[key]:
                inter = get_similarity(image, q_transformed_image.flatten(), model)
                if key in score.keys():
                    if score[key] > inter :
                        score[key] = inter
                        
                else:
                    score[key] = inter
        
    
    dictScore = dict(sorted(score.items(), key = itemgetter(1), reverse = False))
    print('Subject: '+list(dictScore.keys())[0])
                  


# define similarity functions
def earthmover(a,b):
  return wasserstein_distance(a,b)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cmTest(img1,img2):
  return mse(img1,img2)

def l1(img1,img2):
  img1 = img1.reshape(1,img1.shape[0])
  return numpy.linalg.norm((img1 - img2), ord=1)

def mse(x,y):
    return mean_squared_error(x,y)

def hogTest(img1,img2):
    return earthmover( img1, img2 ) 

def elbpTest(img1,img2):
    hamming_dist = distance.hamming(img1, img2)
    return hamming_dist

def get_similarity(image, q_transformed_image, model):

    if model == 'Color_Moments':
        return cmTest(image,q_transformed_image)
    
    elif model == 'HOG':
        return hogTest(image,q_transformed_image)

    elif model == 'ELBP':
        return elbpTest(image,q_transformed_image)