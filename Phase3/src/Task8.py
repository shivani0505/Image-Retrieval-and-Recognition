#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: glen-dsouza
"""

import pickle
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand



class Task8:

	def getRandomSimilarityMatrixFromWeightMatrix(self,weight_matrix):
	    similarity_matrix={}
	    seed(1)
	    lengths = [len(v) for v in weight_matrix.values()]
	    mx=np.random.rand(len(weight_matrix.keys()),lengths[0]) #len(weight_matrix.keys())
	    row=0
	    col=0
	    for i in weight_matrix.keys():
	        rowDict={}        
	        for j in weight_matrix[i].keys():
	            val=mx[row][col]
	            rowDict.update({str(j):val})
	            col=col+1
	        col=0
	        row=row+1
	        similarity_matrix.update({i:rowDict})
	    return similarity_matrix


	def task8_Subtask1(self,S_S_WeightMatrix,n,m):
		NGraph={}
		for i in S_S_WeightMatrix:
			topKEdges=dict(sorted(S_S_WeightMatrix[i].items(), key=lambda item: item[1], reverse=True))
			N_similar_neighbors={}
			for j in topKEdges:
				N_similar_neighbors.update({j:topKEdges[j]})
			NGraph[i]=N_similar_neighbors
		TopNNodes={}
		nodeNum=0
		for i in NGraph.keys():
			nodeNum=0
			edges={}
			for j in NGraph[i].keys():
				if(nodeNum==0):
					nodeNum=nodeNum+1
					continue
				if(int(nodeNum)<=int(n)):
					edges.update({j:NGraph[i][j]})
					nodeNum=nodeNum+1
				else:
					break
				TopNNodes.update({i:edges})
		return TopNNodes



	def AscosPlus(self,weight_matrix,m):
	    #check if graph converges
	    #Optimization function for similarity_matrix
	    #similarity_matrix=np.random.rand(len(weight_matrix.keys()),len(weight_matrix.keys()))
	    similarity_matrix=self.getRandomSimilarityMatrixFromWeightMatrix(weight_matrix)
	    while(True): #To converge
	        #print('similarity score is')
	        #print(similarity_matrix)
	        current_matrix=self.copydict(similarity_matrix)
	        for i in similarity_matrix.keys():
	            for j in similarity_matrix[i].keys():
	                similarity_matrix[i][j]=self.calculateSimilarity(similarity_matrix,i,j,weight_matrix)
	        diff=self.calculateError(current_matrix,similarity_matrix)
	        if(diff==0.0):
	            break
	    # print('Ascos++')
	    # print(similarity_matrix) 
	    topNodes = self.findTopMNodes(m,similarity_matrix)
	    return topNodes


	def calculateWStar(self,i,weight_matrix):
	    weight=0;
	    for k in weight_matrix[i].keys():
	        if((i in weight_matrix[k].keys())):
	            weight=weight+weight_matrix[i][k]
	    return weight

	def calculateError(self,intialMatrix, CurrentMatrix):
	    diff=0;
	    for i in intialMatrix.keys():
	        for j in intialMatrix[i].keys():
	            diff=diff+abs(intialMatrix[i][j]-CurrentMatrix[i][j])
	    return diff


	def findTopMNodes(self,m, similarity_matrix):
	    nodes={}
	    for i in similarity_matrix.keys():
	        val=0
	        N=0
	        for j in similarity_matrix[i].keys():
	            val=val+similarity_matrix[i][j]
	            N=len(j)
	        val=val/(N)
	        nodes.update({i:val})
	    topNodes=dict(sorted(nodes.items(), key=lambda item: item[1],reverse=True))
	    topMNodes={}
	    r=0
	    for i in topNodes:
	    	if(int(r)==int(m)):
	    		break
	    	r=r+1
	    	topMNodes.update({i:topNodes[i]})
	    return topMNodes
    	



	#weight matrix is the subject subject similarity matrix
	def calculateSimilarity(self,similarity_matrix,i,j, weight_matrix):
	    #No. of incoming nodes to i is i'th column
	    c=1
	    val=0
	    if(i==j):
	        return 1.0
	    wStar = self.calculateWStar(i,weight_matrix)
	    #k is the neighbor node in consideration
	    for k in weight_matrix[i].keys():
	        if(k==i or (i not in weight_matrix[k].keys()) or (j not in weight_matrix[k].keys())):
	            continue
	        else:
	            val=val+((float(weight_matrix[i][k])/wStar)*(1-math.exp(-weight_matrix[i][k])))*similarity_matrix[k][j]
	    ans = c*val
	    return ans
	    


	def copydict(self,similarity_matrix):
	    copy_matrix={}
	    for i in similarity_matrix.keys():
	        edges={}
	        for j in similarity_matrix[i].keys():
	            edges.update({j:similarity_matrix[i][j]})
	        copy_matrix.update({i:edges})
	    return copy_matrix
        









