import numpy as np 

class LogisticLoss():
    def sigmoid(self, z):
        z =np.asarray(z,dtype=np.float64)
        return 1/(1+np.exp(-z))

    
    def loss(self, y,z):
        return np.logaddexp(0,z) - y*z

    def grad(self, y,z):
        y = np.asarray(y,dtype=np.float64)
        return self.sigmoid(z)-y
    def hess(self, y,z):
        s=self.sigmoid(z)
        return s*(1-s)
    def init_score(self,y):
        p = np.clip(np.mean(y), 1e-15,1-1e-15)
        return np.log(p/(1-p))

class SquaredLoss():
    def loss(self, y,z):
        return .5*(np.asarray(y,dtype=np.float64)-np.asarray(z,dtype=np.float64))**2
    def grad(self, y,z):
        return np.asarray(z,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    def hess(self, y,z):
        return np.ones(len(np.asarray(z,dtype=np.float64)))
    def init_score(self,y,z):
        return 0



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value 

    def is_leaf(self):
        return self.value is None 

