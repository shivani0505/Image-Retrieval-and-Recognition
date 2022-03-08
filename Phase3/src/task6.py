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

def task_to_get_Type(transformed_images, images, q_transformed_image, model):  


    left_matrix = transformed_images
    q_latent = q_transformed_image
    feature_model = model
    images_matrix = images
    score_dict = {}
    for i in range(len(left_matrix)):
        s = get_similarities_task6(left_matrix[i],q_latent.flatten(),feature_model)
        score_dict[i] = s
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))
    nearest_images = n_nearest_images_task6(score_dict,images_matrix,10)
    
    freq_image_type = {}
    
    for image in nearest_images:
        image_type = image.split('-')[1]
        if image_type not in freq_image_type:
            freq_image_type[image_type] = 1
        else:
            freq_image_type[image_type] = freq_image_type[image_type] + 1
    max_freq_val = -1e9
    result_image_type = ""
    for image_type in freq_image_type:
        if (freq_image_type[image_type] > max_freq_val):
            max_freq_val = freq_image_type[image_type]
            result_image_type = image_type
    return result_image_type            
     
	
def get_similarities_task6(database_latent, q_latent, feature_model):
    if ( feature_model['feature_model'] == 'Color_Moments'):
        return cmTest(database_latent,q_latent)
    elif ( feature_model['feature_model'] == 'HOG'):
        return hogTest(database_latent,q_latent)
    elif ( feature_model['feature_model'] == 'ELBP'):
        return elbpTest(database_latent,q_latent)
		
def n_nearest_images_task6(dis, images_matrix,n):
    n = int(n)
    res = dict(list(dis.items())[0: n])
    images = []
    for keys in res:
        images.append(images_matrix[keys])
    return images    

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

def mse(x,y):
    return mean_squared_error(x,y)

def hogTest(img1,img2):
    return np.linalg.norm(img1-img2,ord = 2)
    #return earthmover( img1, img2 ) 

def elbpTest(img1,img2):
    return np.linalg.norm(img1-img2,ord = 2)

def get_similarity(image, q_transformed_image, model):

    if model == 'Color_Moments':
        return cmTest(image,q_transformed_image)
    
    elif model == 'HOG':
        return hogTest(image,q_transformed_image)

    elif model == 'ELBP':
        return elbpTest(image,q_transformed_image)