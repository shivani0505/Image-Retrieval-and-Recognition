import numpy as np
import constants
from Distance_Similarity_Measure import Dis_Measure
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import scipy
from scipy.spatial import distance
import constants as c
from matplotlib import pyplot as plt
from PIL import Image
import os

class Top_K_Img:
    def __init__(self, data_path, q_img):
        self.q_img = q_img
        self.data_path = data_path
        return

    def compute_score(self, q_latent, left_matrix, images_matrix,feature_model,n):
        score_dict = {}
        for i in range(len(left_matrix)):
            s = self.get_similarities(left_matrix[i],q_latent.flatten(),feature_model)
            score_dict[i] = s
        score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]))
        return self.n_nearest_images(score_dict,images_matrix,n)
    
    def get_similarities(self,database_latent, q_latent, feature_model):
        if ( feature_model['feature_model'] == 'Color_Moments'):
            return self.cmTest(database_latent,q_latent)
        elif ( feature_model['feature_model'] == 'HOG'):
            return self.hogTest(database_latent,q_latent)
        elif ( feature_model['feature_model'] == 'ELBP'):
            return self.elbpTest(database_latent,q_latent)
        
    def n_nearest_images(self, dis, images_matrix,n):
        n = int(n)
        res = dict(list(dis.items())[0: n])
        images = []
        for keys in res:
            images.append(images_matrix[keys])
            #print(images_matrix[keys])
        #self.visualize_k_images(images)
        #self.visualize_result(res,images,n)
        return images

    def earthmover(self,a,b):
        return wasserstein_distance(a,b)
        
    def cmTest(self,img1,img2):
        return mean_squared_error(img1,img2)

    def hogTest(self,img1,img2):
        return np.linalg.norm(img1-img2,ord = 2)
        #return self.earthmover( img1, img2 )
        
    def elbpTest(self,img1,img2):
        return np.linalg.norm(img1-img2,ord = 2)
        #hamming_dist = distance.hamming(img1, img2)
        #return hamming_dist
        
    def visualize_result(self,distances,images,n):
        counter = 1
        for keys in distances:
            filename = c.input_path + "/" + images[keys]
            img = plt.imread(filename)
            plt.subplot(1,n,counter)
            plt.imshow(img)
            plt.axis('off')
            plt.title(images[keys])
            counter = counter + 1
        plt.show()
    
    def visualize_k_images(self, images):        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(len(images)+1,
                                    figsize=(20,20),
                                    subplot_kw={'xticks': [], 'yticks': []})
        
        axis[0].imshow(np.array(self.q_img.reshape(64,64)), cmap='gray')
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
        plt.show()