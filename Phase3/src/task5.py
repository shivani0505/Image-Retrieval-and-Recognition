import os
import numpy as np
import sys
import math
from os import listdir
from os.path import isfile, join
from scipy.stats import skew
from PIL import Image
from skimage import feature
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from driver_phase2 import TaskDriver
from tabulate import tabulate
import constants
from driver import TaskDriverPhase3


    
def makeDir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def Task5Subtask1():
     #DBImageVectors = findImageVectors(imagesFolder, model)
    b=input("Enter the number of bits")
    b=int(b)
    latent_sem_file=input("enter the latent semantics file")
    decomposed = load_numpy(latent_sem_file)
    #decomposed=decom
    #core_matrix = TaskDriver.load_core_matrix(latent_sem_file, decomposed[0].shape[1])
    decomposed=np.array(decomposed)
    dictionaryVal=convertArrayToDict(decomposed)
    lengths = [len(v) for v in dictionaryVal.values()]
    numberOfDimensions = lengths[0]
    bitsPerDimension = findNumberOfBitsPerDimension(numberOfDimensions, b)
    partitionPoints = findPartitionPoints(bitsPerDimension, dictionaryVal, numberOfDimensions)
    approximationDictionary = findApproximationForEachImage(partitionPoints, dictionaryVal, numberOfDimensions)
    VA_files_dictionary=getVAFilesStringsUniquely(approximationDictionary)
    print('##########----index size is----#########')
    print('ParititonsDictionary is: '+str(sys.getsizeof(partitionPoints))+' bytes')
    print("Approximations dictionary is {} bytes".format(sys.getsizeof(approximationDictionary)))
    print("Therefore total index size is {} bytes".format(sys.getsizeof(partitionPoints)+sys.getsizeof(approximationDictionary)))
    print(VA_files_dictionary)
    data = list(VA_files_dictionary.items())
    an_array=np.array(data)
    filename="outputs/task5/task5_subtask1_VA_Files.npy"
    #save_numpy_array(matrix=an_array,filename=filename)


def convertArrayToDict(arr):
    dictionaryVal={}
    #arr=np.transpose(arr)
    for i in range(0,arr.shape[0]):
        dc={}
        for j in range(0,arr.shape[1]):
            dc[j]=arr[i][j]
        dictionaryVal[i]=dc
    return dictionaryVal


def Task5Initialize(): #self, imagesFolder, model, b
    #DBImageVectors = findImageVectors(imagesFolder, model)
    b=input("Enter the number of bits")
    b=int(b)
    data_path=input("enter the absolute path of the images directory")
    fileList = getFiles(data_path)
    typeOfData=input("Enter 1 for CM, 2 for ELBP, 3 for HOG")
    images_dictionary={}
    typeOfData=int(typeOfData)
    if(typeOfData==1):
        Images_dictionary=compute_image_color_moments(data_path,fileList)
    elif(typeOfData==2):
        Images_dictionary=compute_image_ELBP(data_path,fileList)
    elif(typeOfData==3):
        Images_dictionary=compute_image_hog(data_path,fileList)
    approximationDictionary = Task5Init(Images_dictionary, b, typeOfData)
    return
    #imagePath = input('enter the path of the image')

    # get the image from the image path and convert to the model and then use that to approximate

def Task5Init(DBImageVectors, b, model, targetImagePath,data_path,models):
    #print('the image vectors are')
    t=input("enter the value of t")
    t=int(t)
    path=constants.output_path+"outputs/task5"
    makeDir(path)
    #print(len(DBImageVectors))
    lengths = [len(v) for v in DBImageVectors.values()]
    numberOfDimensions = lengths[0]
    #print('the number of dimensions are')
    #print(numberOfDimensions)
    bitsPerDimension = findNumberOfBitsPerDimension(numberOfDimensions, b)
    #print('The bits per dimension are')
    #print(bitsPerDimension)
    partitionPoints = findPartitionPoints(bitsPerDimension, DBImageVectors, numberOfDimensions)
    #print('the partition points are')
    #print(partitionPoints)
    approximationDictionary = findApproximationForEachImage(partitionPoints, DBImageVectors, numberOfDimensions)
    VA_files_dictionary=getVAFilesStringsUniquely(approximationDictionary)
    #print('the approximationDictionary is')
    #print(approximationDictionary)
    indexSize = outputSizeOfIndexStructure(approximationDictionary)
    print('##########----index size is----#########')
    print('ParititonsDictionary size is: '+str(sys.getsizeof(partitionPoints))+' bytes')
    print("Approximations dictionary is {} bytes".format(sys.getsizeof(approximationDictionary)))
    print("Therefore total index size is {} bytes".format(sys.getsizeof(partitionPoints)+sys.getsizeof(approximationDictionary)))
    targetImage={}
    if(model==1):
        targetImage=compute_single_image_color_moments(targetImagePath)
        targetImage=targetImage[0]
    elif(model == 2):
        targetImage=compute_single_image_ELBP(targetImagePath)
    elif (model==3):
        targetImage=compute_single_image_hog(targetImagePath)
    # if cm then targetImage=targetImage[0] else just targetImage
    #targetImage=compute_single_image_ELBP(target_image_directory)
    #targetImage=targetImage[0]
    candidateList = VA_SSA(approximationDictionary, targetImage, DBImageVectors,t,partitionPoints,model,VA_files_dictionary, targetImagePath,data_path,models,b)
    return candidateList
    
def outputSizeOfIndexStructure(approximationDictionary):
    ans=0
    for i in approximationDictionary.keys():
        for j in approximationDictionary[i].keys():
            #print('ith key has binary as')
            #print(approximationDictionary[i][j])
            ans=ans+len(approximationDictionary[i][j])
            #print('len is '+str(len(approximationDictionary[i][j])))
    return ans


def findApproximationForEachImage(partitionPoints, DBImageVectors, numberOfDimensions):
    # check the partition points properly
    approximations={} #key is dimension, value is string
    for i in DBImageVectors.keys():
        approximations[i]={}
    for j in range(0,numberOfDimensions):
        if(len(partitionPoints[j])==0):
            continue
        for i in DBImageVectors:
            val = DBImageVectors[i][j]
            bucket = findBucket(partitionPoints[j], j, val) #iterate over the array and find where the val is 
            approximateString = findApproximate(bucket)
            approximations[i][j]=approximateString
    return approximations

def findBucket(partitionPoints, dimension, value):
    partitionFound=False
    #print(partitionPoints)
    #print(value)
    if(len(partitionPoints)==1):
        return 0
    for i in range(0,len(partitionPoints)):
        if(value<partitionPoints[i]):
            return i-1
    return len(partitionPoints)-1
    

def findApproximate(value):#find binary representation of this number
    get_bin = lambda x: format(x, 'b')
    binaryRepresentation = str(get_bin(value))
    return binaryRepresentation


def findNumberOfBitsPerDimension(numberOfDimensions, b):
    bitsPerDimension = {}
    d = numberOfDimensions
    for j in range(1, numberOfDimensions+1):
        k = 0
        if(j<=(b%d)):
            k=1
        bitsPerDimension[j-1]=int((b/numberOfDimensions)+ k)
    return bitsPerDimension


def findJthPartitions(jthValues, numberofBits):
    numberOfPartitionsInThisDimension = pow(2,numberofBits)+1
    
    partitions=[]
    totalPoints=len(jthValues)
    iteration=int(totalPoints/(numberOfPartitionsInThisDimension-1))
    if(numberOfPartitionsInThisDimension==2):
        return partitions
    interval=iteration
    partitions.append(0)
    for i in range(0,numberOfPartitionsInThisDimension-2):#while(iteration<=totalPoints-1):
        if(iteration>=len(jthValues)-1):
            break
        lowerVal=jthValues[iteration]
        upperVal=jthValues[iteration+1]
        partitions.append(lowerVal+(upperVal-lowerVal)/2)
        iteration=iteration+1+interval
    partitions.sort()
    return partitions


def findPartitionPoints(bitsPerDimension, DBImageVectors, numberOfDimensions):
    partitionPoints = {} #stores partition indexes in jth dimension
    #sort the DBImageVectors and then partition equally for every dimension
    sortedValues={}
    for j in range(0,numberOfDimensions):
        jthValues=[]
        for i in DBImageVectors.keys():
            jthValues.append(DBImageVectors[i][j])
        jthValues.sort()# sort based on values of jth dimension and retain i
        partitionPoints[j]=findJthPartitions(jthValues,bitsPerDimension[j])
    
    return partitionPoints

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
 


def VA_SSA(approximationDictionary, targetImage, DBImageVectors,t,partitionPoints,model,VA_files_dictionary, targetImagePath,data_path,models,b):
    print(data_path,", data_path")
    print(models,", model")
    print(b,", bits")
    obj=TaskDriverPhase3()
    obj.create_folder_in_file_dict(data_path)
    d = sys.maxsize
    numberOfUniqueImages=0
    candidateList = {}
    bucketsSearched={}
    dst = initCandidate(t)
    l={}
    candidatesFound=0
    ans = initAnsList(t)
    for i in approximationDictionary.keys():
        l[i]=GetBounds(approximationDictionary[i],targetImage,partitionPoints)
        #print('l[i] is '+str(l[i]))
        if l[i]<d:
            bucketsSearched[VA_files_dictionary[i]]=1
            numberOfUniqueImages=numberOfUniqueImages+1
            dist,d,candidatesFound,ans,dst=Candidate(findDist(DBImageVectors[i],targetImage,model),i,dst,t,ans,candidatesFound)
            candidateList[i]=dist
    #print(VA_files_dictionary)
    print('The final ans vector is')
    print(ans)
    print('the final dst vector is')
    print(dst)
    print('######## number of unique images considered are ###')
    print(numberOfUniqueImages)
    print('#### Number of buckets searched are ####')
    print(len(bucketsSearched.keys()))
    actualAnswer = findActualDistances(DBImageVectors,targetImage,t,model)
    intersect_ans=intersection(actualAnswer, ans)
    number_of_correct_images=len(intersect_ans)
    miss_rate=((t-number_of_correct_images)/t)
    print("###### Miss rate is #####")
    print(miss_rate)
    false_positive_rate=(numberOfUniqueImages-number_of_correct_images)/numberOfUniqueImages
    print('######## False Positive Rate is ####')
    print(false_positive_rate)
    candidateList = dict(sorted(candidateList.items(), key=lambda item:item[1]))
    consideredList={}
    for i in candidateList.keys():
        consideredList[i]=DBImageVectors[i]

    data = list(VA_files_dictionary.items())
    an_array = np.array(data)
    filename= "task5_VA_files.npy"
    save_numpy_array(matrix=an_array,filename=filename)
    candidateListFeatureVectors=[]

    for i in consideredList:
        # ls=[]
        # ls.append(i)
        # ls.append(consideredList[i])
        candidateListFeatureVectors.append(consideredList[i])
    an_array=np.array(candidateListFeatureVectors)
    print("phase3_task5_nearest_images")
    print(an_array)
    filename= "phase3_task5_nearest_images.npy"
    nearest_images_filename = "phase3_task5_" + str(models)+ "_" + str(b) +"_nearest_images"+".npy"
    obj.save_file_in_dict(data_path, an_array, nearest_images_filename)
    filenameCSV= "phase3_task5_nearest_images.csv"
    #np.savetxt(filenameCSV, an_array, delimiter=",")
    save_numpy_array(matrix=an_array,filename=filename)
    save_numpy_arrays(matrix=an_array, filename=filenameCSV)
    nearest_images_index=[]
    for i in candidateList:
        ls=[]
        ls.append(i)
        ls.append(candidateList[i])
        nearest_images_index.append(ls)
    an_array=np.array(nearest_images_index)
    print("phase3_task5_nearest_images_index_and_distance")
    print(an_array)
    filename= "phase3_task5_nearest_images_index_and_distance.npy"
    filenameCSV="phase3_task5_nearest_images_index_and_distance.csv"
    nearest_images_index_and_distance_filename = "phase3_task5_" + str(models)+ "_" + str(b) +"_nearest_images_index_and_distance"+".npy"
    obj.save_file_in_dict(data_path, an_array, nearest_images_index_and_distance_filename)
    #np.savetxt(filenameCSV, an_array, delimiter=",")
    save_numpy_array(matrix=an_array,filename=filename)
    save_numpy_arrays(matrix=an_array, filename=filenameCSV)
    data = targetImage
    an_array=targetImage
    print("task5_query_image_vector")
    print(an_array)
    filename="phase3_task5_query_vector.npy"
    filenameCSV="phase3_task5_query_vector.csv"
    targetImageName=[]
    targetImageName.append(targetImage)
    query_vector_filename = "phase3_task5_" + str(models)+ "_" + str(b) +"_query_vector"+".npy"
    obj.save_file_in_dict(data_path, targetImageName, query_vector_filename)
    save_targetImage(matrix=targetImageName, filename=filename)
    save_numpy_arrays(matrix=targetImageName, filename=filenameCSV)
    #np.savetxt(filenameCSV, targetImageName, delimiter=",")
    #print(targetImage)
    ansDict = getAnsDict(ans,dst)
    data = list(ansDict.items())
    an_array=np.array(data)
    filename="phase3_task5_top_t_images.npy"
    save_numpy_array(matrix=an_array,filename=filename)
    #save_numpy_arrays(matrix=an_array, filename=filenameCSV)
    filename="phase3_task5_query_image_name.npy"
    filenameCSV="phase3_task5_query_image_name.csv"
    print("query image name")
    #print(imageName)
    imageName = targetImagePath#.rsplit('/', 1)
    print(imageName)
    targetImageName=[]
    targetImageName.append(targetImagePath)
    query_image_name_filename = "phase3_task5_" + str(models)+ "_" + str(b) +"_query_image_name"+".npy"
    obj.save_file_in_dict(data_path, targetImageName, query_image_name_filename)
    save_numpy_array(matrix=targetImageName,filename=filename)
    save_numpy_array(matrix=targetImageName,filename=filenameCSV)
    return consideredList

def save_targetImage(matrix, filename):
    filename =os.path.join(constants.output_path,filename)
    if ( os.path.exists(filename) ):
        os.remove(filename)
    with open(filename, 'wb') as f:
        np.save(f,matrix)


def getAnsDict(ans,dst):
    ansDict={}
    for i in range(0,len(ans)):
        ansDict[ans[i]]=dst[i]
    print(ansDict)
    return ansDict

def findActualDistances(DBImageVectors, targetImage, t, model):
    ansList={}
    for i in DBImageVectors.keys():
        ansList[i]=findDist(DBImageVectors[i],targetImage,model)
    ansList = dict(sorted(ansList.items(), key=lambda item: item[1]))
    ans=[]
    num=0
    for i in ansList.keys():
        ans.append(i)
        num=num+1
        if(num==t):
            break
    return ans

def earth_mover(Image1, Image2):
    cost = wasserstein_distance(Image1,Image2)
    return cost
    

def findDistance(Image1,Image2):
    ans=0
    for i in range(0,len(Image1)):
        ans=ans+(Image1[i]-Image2[i])*(Image1[i]-Image2[i])
    math.sqrt(ans)
    return ans

def getVAFilesStringsUniquely(approximationDictionary):
    VA_files_dictionary={}
    st=str(0)
    for i in approximationDictionary.keys():
        approxString=""
        for j in approximationDictionary[i].keys():
            if(int(approximationDictionary[i][j])==-1):
                approxString=approxString+st
                continue
            approxString=approxString+str(approximationDictionary[i][j])
        VA_files_dictionary[i]=approxString
        #print('approx string is')
        #print(approxString)
    #print('number of keys in VA files are '+str(len(VA_files_dictionary.keys())))
    return VA_files_dictionary

def findDist(Image1,Image2,model):
    return findDistance(Image1, Image2)
#     if(model==1):
#          return findMSE(Image1,Image2)
#     elif model==2:
#         return earth_mover(Image1,Image2)
#     elif model==3:
#         return distance.hamming(Image1,Image2)
    
def findMSE(Image1, Image2):
    return mean_squared_error(Image1,Image2)

def initAnsList(t):
    ans=[]
    for i in range(0,t):
        ans.append(-1)
    return ans



def initCandidate(t):
    dst = []
    maxSize = sys.maxsize
    for k in range(0,t):
        dst.append(maxSize)

    return dst

def Candidate(d, i,dst,t,ans,candidatesFound):
    #print('distance is '+str(d))
    #print('candidates found is')
    #print(candidatesFound)
    #print('dst[t-1] is ' +str(dst[t-1]))
    #print(dst)
    
    if(candidatesFound<t-1):
        dst[candidatesFound]=d
        ans[candidatesFound]=i
        candidatesFound=candidatesFound+1
        dst,ans = sortOnDST(ans,dst,t)
        #print('ans vector is ')
        #print(ans)
        #print('dst vector is')
        #print(dst)
        return d,dst[t-1],candidatesFound,ans,dst
    elif(d<dst[t-1]):
        dst[t-1]=d
        ans[t-1]=i
        dst,ans = sortOnDST(ans,dst,t)
    return d,dst[t-1],candidatesFound,ans,dst

def sortOnDST(ans,dst,t):
    zipped_lists = zip(dst, ans)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    dst, ans = [ list(tuple) for tuple in  tuples]
    return dst,ans

def GetBounds(approximates, targetImage, partitionPoints):
    d=0
#     if(str(typeOfCheck)=="CM"):
#         print('hello')
#         targetImageValues=targetImage[0]
#     else:
#         targetImageValues=targetImage
    #print(targetImage)
    targetImageValues=targetImage
    for j in approximates.keys():#j represents keys which are dimensions
        region = convertBinaryToNumber(approximates[j])
        #print(partitionPoints[j])
        #print(targetImageValues[j])
        #print(int(j))
        qjthRegion = findRegion(partitionPoints[j],targetImageValues[j],int(j))
        #print('partitionpoints for dimension '+str(j) +' and region '+str(region)+' is')
        #print(partitionPoints[j][region])
        if(region<qjthRegion):
            k = targetImageValues[j]
            #if(len(partitionPoints[j])<(region+1)):
            k= k-partitionPoints[j][region+1]
            k=k*k
            d=d+k
        elif(region>qjthRegion):
            #if(len(partitionPoints[j])<(region)):            
            k=partitionPoints[j][region]
            k= k-targetImageValues[j]
            k=k*k
            d=d+k

    return d



def findRegion(jthPartitions, targetImagesJthValue,dimension):
    if(len(jthPartitions)==1 or len(jthPartitions)==0):
        return 0
    for i in range(0,len(jthPartitions)):
        if(targetImagesJthValue<=jthPartitions[i]):
            return i-1
    return len(jthPartitions)-2

def convertBinaryToNumber(binaryString):
    return int(binaryString, 2)

def compute_image_color_moments(data_path,image_ids):
        images_CM = []
        images_vector_CM={}
        for image_id in image_ids:
            features =  []
            image_np_arr = Image.open(data_path+"/"+image_id)
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            M = 8
            N = 8
            tiles = [image_np_arr[x:x+M,y:y+N] for x in range(0,image_np_arr.shape[0],M) for y in range(0,image_np_arr.shape[1],N)]
            nptiles = np.array(tiles)
            CM_mean = nptiles.mean(axis=(1,2))
            CM_SD = nptiles.std(axis=(1,2))
            CM_skew = [skew(tile.flatten()) for tile in nptiles]
            #print(len(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()])))
            images_CM.append(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()]))
            images_vector_CM[image_id]=np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()])
        ##TODO how to combine the mean, std and skew to make a feature vector
        #print(len(images_CM[0]))
        return images_vector_CM
    
def compute_image_ELBP(data_path,image_ids):
        ELBP = []
        images_vector_ELBP={}
        for image_id in image_ids:
            image_np_arr = Image.open(os.path.join(data_path,image_id))
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            i_min = np.min(image_np_arr)
            i_max = np.max(image_np_arr)
            if ( i_max - i_min != 0 ):
                image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
            
            lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
            ELBP.append(lbp.flatten())
            images_vector_ELBP[image_id]=lbp.flatten()
        return images_vector_ELBP
    
def compute_image_hog(data_path,image_ids):
        image_vector_HOG={}
        for image_id in image_ids:
            image_np_arr = Image.open(data_path+"/"+image_id) #(os.path.join(self.data_path, image_id))
            image_np_arr = image_np_arr.convert("L")
            image_np_arr = np.array(image_np_arr)
            fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
            fd = np.array(fd)
            image_vector_HOG[image_id]=fd.flatten()
        return image_vector_HOG
    
    
def compute_single_image_color_moments(target_image_path):
        images_CM = []
        image_np_arr = Image.open(target_image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        M = 8
        N = 8
        tiles = [image_np_arr[x:x+M,y:y+N] for x in range(0,image_np_arr.shape[0],M) for y in range(0,image_np_arr.shape[1],N)]
        nptiles = np.array(tiles)
        CM_mean = nptiles.mean(axis=(1,2))
        CM_SD = nptiles.std(axis=(1,2))
        CM_skew = [skew(tile.flatten()) for tile in nptiles]
        ##TODO how to combine the mean, std and skew to make a feature vector
        images_CM.append(np.concatenate([CM_mean.flatten(), CM_SD.flatten(), np.array(CM_skew).flatten()]))
        return images_CM
    
def compute_single_image_ELBP(target_image_path):
        ELBP = []
        image_np_arr = Image.open(target_image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        i_min = np.min(image_np_arr)
        i_max = np.max(image_np_arr)
        if ( i_max - i_min != 0 ):
            image_np_arr = (image_np_arr-i_min)/(i_max - i_min)
        
        lbp = feature.local_binary_pattern(image_np_arr, 8, 1, method = 'nri_uniform')
        return lbp.flatten()
    
def compute_single_image_hog(target_image_path):
        HOG = []
        image_np_arr = Image.open(target_image_path)
        image_np_arr = image_np_arr.convert("L")
        image_np_arr = np.array(image_np_arr)
        fd, hog_image = feature.hog(image_np_arr, orientations=9, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        fd = np.array(fd)
        return fd.flatten()
    
    
def getFiles(data_path):
    filesList = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    return filesList

def getTargetImage(target_image_path):
    return

def save_numpy_array(matrix, filename):
    filename =os.path.join(constants.output_path,filename)
    if ( os.path.exists(filename) ):
        os.remove(filename)
    print ("Saving file "+filename)
    f = open((filename), "w")
    f.seek(0)
    f.write(tabulate(matrix, [], tablefmt="grid"))
    f.truncate()
    f.close()
    np.save(filename, matrix)
    return

def save_numpy_arrays(matrix, filename):
    filename=constants.output_path+"/"+filename
    if ( os.path.exists(filename) ):
        os.remove(filename)
    # print ("Saving file "+filename)
    f = open(filename, "w")
    f.seek(0)
    f.write(tabulate(matrix, [], tablefmt="grid"))
    f.truncate()
    f.close()
    np.save(filename, matrix)
    return


def load_numpy(filename):
    if(os.path.exists(filename)):
        matrix = np.load(filename,allow_pickle=True)
        return matrix

def load_latent_semantics(filename):
    if ( os.path.exists(filename)):
        lev_file = os.path.join(self.out_path , filename)
        lev = self.load_numpy_array(lev_file)
        return lev
    print ("*****ERROR: FILE DOES NOT EXIST*****")
    return
        
