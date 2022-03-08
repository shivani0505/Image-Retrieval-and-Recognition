import os
import numpy as np
from Ph3_Task7 import SVM_Extended
from svm import SVM
import constants as c
from matplotlib import pyplot as plt
from PIL import Image
from os import name
import operator
from image_mcq import MCQ


class Task7_Feedback:
    def __init__(self,candidate_dict= None, data_path = None,query_image = None, query_name = None, query_feature = None, query_path = None, n = None):
        self.x_train=[]
        self.x_train_name = []
        self.y_train=[]
        self.candidates = candidate_dict
        self.x_test = []
        self.test_name = []
        for keys in candidate_dict:
            self.x_test.append(candidate_dict[keys])
            self.test_name.append(keys)
        self.y_test=[]
        self.final_result = None
        self.data_path = data_path
        self.query_image = query_image
        self.query_name = query_name
        self.query_feature = []
        self.query_feature = query_feature
        self.n = n
        self.x_train.append(query_feature.flatten())
        self.x_train_name.append(query_name)
        self.y_train.append(1.)
        self.query_path = query_path
        return
    
    
    
    def visualize_k_images(self, images):        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(len(images)+1,
                                    figsize=(20,20),
                                    subplot_kw={'xticks': [], 'yticks': []})
        
        
        axis[0].imshow(np.array(self.query_image.reshape(64,64)), cmap='gray')
        counter = 1
        
        image_arr = []
        for image_id in images:
            image_np_arr = []
            if os.path.exists(os.path.join(self.data_path, image_id)):
                image_np_arr = Image.open(os.path.join(self.data_path, image_id))
            else:
                image_np_arr = Image.open(os.path.join(self.query_path, image_id))
            # image_np_arr = Image.open(os.path.join(self.data_path, image_id))
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
        plt.savefig(os.path.join(c.output_path, my_file))  
        
        plt.show()
        
    def getFeedback(self,closest_image):
        
        i = input("\n\n Press 1 to give feedback else press 0 : \n\n ")
        if i == '0':
            return
        print("\n Please give for the images : \n ", closest_image)
        feedback_mcq = MCQ(self.data_path, closest_image, self.query_path)
        relevant_images, irrelevant_images = feedback_mcq.give_options()
        
        print(relevant_images)
        print(irrelevant_images)
        
        # R = relevant_images.split(',')
        # IR = irrelevant_images.split(',')
        R = relevant_images
        IR = irrelevant_images
        
        self.learnFromFeedback(R,IR)
        return
    
    def learnFromFeedback(self,R,IR):
        for i in range(len(R)):
            name = R[i]
            if name not in self.x_train_name and name:
                self.x_train.append(self.candidates[name])
                self.y_train.append(1.)
                self.x_train_name.append(name)
                if name in self.test_name:
                    index = self.test_name.index(name)
                    del self.test_name[index]
                    del self.x_test[index]
                        
        for i in range(len(IR)):
            name = IR[i]
            if name not in self.x_train_name and name:
                self.x_train.append(self.candidates[name])
                self.y_train.append(-1.)
                self.x_train_name.append(name)
                if name in self.test_name:
                    index = self.test_name.index(name)
                    del self.test_name[index]
                    del self.x_test[index]
        
        svm = SVM_Extended(C=0.05)
        svm.fit(np.array(self.x_train),np.array(self.y_train))
        y_test = svm.predict(np.array(self.x_test))
        
        top_k_dict = svm.get_nearest(self.x_test, y_test, self.test_name,self.x_train,self.y_train,self.x_train_name,self.n)
        top_k = []
        for keys in top_k_dict:
            top_k.append(keys)
        print("top k images are :", top_k)
        self.final_result = top_k
        self.visualize_k_images(top_k)
        self.getFeedback(top_k)
        
        return
        