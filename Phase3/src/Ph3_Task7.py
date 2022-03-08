
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from _testcapi import test_dict_iteration

def polynomial_kernel(x, y, p=3):
    x = x.transpose()
    return (1 + np.dot(x, y)) ** p

class SVM_Extended(object):

    def __init__(self, kernel=polynomial_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
        self.a = None
        self.sv = None
        self.sv_y = None
        self.b = 0
        self.final_score = {}
        self.decision_boundary = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1, tc='d')
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1), tc='d')
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)), tc='d')
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-7})

        a = np.ravel(solution['x'])
        print(a)
        
        # Support vectors have non zero lagrange multipliers
        sv = a > 0
        
        ind = np.arange(len(a))[sv]
        
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        if(len(self.a) == 0):
            self.b = 0
        else:
            for n in range(len(self.a)):
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
            self.b /= len(self.a)

        self.w = None
            
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.b
    
    def predict(self, X):
        val = self.project(X)
        return np.sign(val)
    
    def kernel_nn(self,x1,x2):
        return self.kernel(x1,x1) - 2*self.kernel(x1,x2) + self.kernel(x2,x2)
    
    def get_nearest(self,x_test,y_test,names,x_train,y_train,x_train_name,k):
        distance_dict = {}
        for n in range(len(y_test)):
            if y_test[n] == 1.:
                for sv_y,sv in zip(self.sv_y, self.sv):
                    if sv_y == 1.:
                        dist = self.kernel_nn(x_test[n],sv)
                        if names[n] in distance_dict:
                            val = distance_dict[names[n]]
                            if val > dist:
                                distance_dict[names[n]] = dist
                        else:
                            distance_dict[names[n]] = dist
        for n in range(len(y_train)):
            if y_train[n] == 1.:
                for sv_y,sv in zip(self.sv_y, self.sv):
                    if sv_y == 1.:
                        dist = self.kernel_nn(x_train[n],sv)
                        if x_train_name[n] in distance_dict:
                            val = distance_dict[x_train_name[n]]
                            if val > dist:
                                distance_dict[x_train_name[n]] = dist
                        else:
                            distance_dict[x_train_name[n]] = dist
        distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        #returning top n elements
        return dict(list(distance_dict.items())[0: k])