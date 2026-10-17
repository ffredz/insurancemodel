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


class Tree:
    def __init__(self, max_depth=3, lam=1.0, gamma=0.0, min_child_weight=1.0): 
        self.max_depth = max_depth
        self.lam = lam 
        self.gamma = gamma 
        self.min_child_weight = min_child_weight
    def fit(self, X, g, h):
        X,g,h = (np.asarray(a,dtype=np.float64) for a in (X,g,h))
        self.root = self._build(X,g,h,np.arange((len(g))), depth = 0)
        return self 
    def _build(self,X,g,h,idx,depth):
        G,H = g[idx].sum(), h[idx].sum()
        w = -G/(H+self.lam)

        if depth >=self.max_depth:
            return Node(value=w)

        split = self._find_best_split(X,g,h,idx)
        if split is None:
            return Node(value=w)

        feature, threshold = split 
        left_idx = idx[X[idx,feature]<=threshold]
        right_idx = idx[X[idx,feature]>threshold]
        return Node(feature=feature, threshold=threshold,
                    left=self._build(X,g,h,left_idx,depth+1),
                    right=self._build(X,g,h,right_idx,depth+1))

    def _find_best_split(self, X,g,h,idx):
        G,H = g[idx].sum(), h[idx].sum()
        if len(idx) < 2:                    
            return None
        best_gain, best_feature, best_threshold = -np.inf, None, None
        for feature in range(X.shape[1]):
            order = np.argsort(X[idx, feature])
            rows = idx[order]
            xs, g_s, h_s = X[rows,feature], g[rows], h[rows]
            G_L, H_L = np.cumsum(g_s[:-1]), np.cumsum(h_s[:-1])
            G_R = G-G_L     
            H_R = H-H_L 
            thresholds = (xs[1:]+xs[:-1])/2
            valid = (xs[1:] > xs[:-1]) & (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
            gain = G_L**2/(H_L+self.lam) + G_R**2/(H_R+self.lam) -G**2/(H+self.lam)
            gain = np.where(valid, gain, -np.inf)
            k = np.argmax(gain)
            if gain[k]>best_gain:
                best_gain,best_feature,best_threshold = gain[k], feature, thresholds[k]
        if best_gain>self.gamma:
            return best_feature, best_threshold
        return None 

    def _predict_node(self,node, X,idx,out):
        if node.left is None:
            out[idx]=node.value
            return 
        go_left = X[idx,node.feature]<=node.threshold
        self._predict_node(node.left,X,idx[go_left],out) 
        self._predict_node(node.right,X,idx[~go_left],out)

    def predict(self, X):    
        X = np.asarray(X,dtype=float)
        out = np.zeros(len(X))
        self._predict_node(self.root, X, np.arange(len(X)),out)
        return out 