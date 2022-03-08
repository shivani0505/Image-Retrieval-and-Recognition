import numpy as np


class ppr:
    def convertToGraph(self, similarity_matrix, n, m):
        n = int(n)
        m = int(m)
        similarity_matrix =  np.array(similarity_matrix)
        for i in range(0, similarity_matrix.shape[0]):
            col  = similarity_matrix[:, i]
            idx = np.argsort(col)[::-1]
            for j in range (n, np.array(idx).size):
                similarity_matrix[idx[j], i] = 0
        for i in range(0, similarity_matrix.shape[0]):
            col  = similarity_matrix[:, i]
            col = col / np.sum(col)
            similarity_matrix[:, i] = col
        return similarity_matrix

    def findPersonalizedRank(self, subject1, subject2, subject3, similarityGraph, randomWalkProb):
        similarityGraph = np.array(similarityGraph)
        similarityGraph = np.multiply(similarityGraph, randomWalkProb)
        similarityGraph = np.array(similarityGraph)
        I = np.identity(similarityGraph.shape[0], dtype=float)
        similarityGraph = np.subtract(I, similarityGraph)
        inverseOfsimilarityGraph = np.linalg.inv(similarityGraph)
        teleportProb = 1 - randomWalkProb

        seedVector = np.zeros((similarityGraph.shape[0]))
        seedVector1 = np.zeros((similarityGraph.shape[0]))
        seedVector2 = np.zeros((similarityGraph.shape[0]))
        seedVector3 = np.zeros((similarityGraph.shape[0]))
        weight = 1
        seedVector[subject1-1] = weight
        #seedVector[subject2-1] = weight
        #seedVector[subject3-1] = weight
        seedVector1[subject1-1] = 1
        seedVector2[subject2 -1] = 1
        seedVector3[subject3-1] = 1
        seedVector = np.multiply(teleportProb, seedVector)
        seedVector1 = np.multiply(teleportProb, seedVector1)
        seedVector2 = np.multiply(teleportProb, seedVector2)
        seedVector3 = np.multiply(teleportProb, seedVector3)
        pageRank1 = np.matmul(inverseOfsimilarityGraph, seedVector1)
        pageRank2 = np.matmul(inverseOfsimilarityGraph, seedVector2)
        pageRank3 = np.matmul(inverseOfsimilarityGraph, seedVector3)
        pageRank =  np.matmul(inverseOfsimilarityGraph, seedVector)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(0, np.array(pageRank3).size):
            if i == subject2-1 or i == subject1 -1 or i == subject3 -1:
                sum1 = sum1 + pageRank1[i]
                sum2 = sum2 + pageRank2[i]
                sum3 = sum3 + pageRank3[i]
        return pageRank
        if sum1 > sum2 and sum1 > sum3:
            return pageRank1
        if sum2 > sum1 and sum2 > sum3:
            return pageRank2
        if sum3 > sum2 and sum3 > sum1:
            return pageRank3
        if sum1 == sum2:
            pagerankresult = np.add(pageRank1,pageRank2)
            pagerankresult = np.multiply(pagerankresult, 0.5)
            return pagerankresult
        if sum3 == sum2:
            pagerankresult = np.add(pageRank3,pageRank2)
            pagerankresult = np.multiply(pagerankresult, 0.5)
            return pagerankresult
        if sum1 == sum3:
            pagerankresult = np.add(pageRank1,pageRank3)
            pagerankresult = np.multiply(pagerankresult, 0.5)
            return pagerankresult
        if sum1 == sum2 ==sum3:
            pagerankresult = np.add(np.add(pageRank1, pageRank3),pageRank2)
            pagerankresult = np.multiply(pagerankresult, np.divide(1,3))
            return pagerankresult

# object = ppr()
#
#
# # similarityGraph = object.convertToGraph(np.random.random( (4,4)), 2, 2)
# t = np.divide(1,3)
# t = float(t)
# similarityGraph = np.array([[0, 1, 1, 0.5, 0], [t, 0, 0, 0, 0], [t, 0, 0, 0, 0], [t, 0, 0, 0, 1], [0, 0, 0, 0.5, 0]])
# result = object.findPersonalizedRank(1,2,3,similarityGraph,0.85)
# idx = np.argsort(result)[::-1]
# for i in range(0, 3):
#     print(idx[i] + 1)
# print(idx)
# print(result)
