import numpy as np 
from sklearn.base import BaseEstimator
from metrics import norm_gini_coeff




#egen implementasjon
class LogisticRegression(BaseEstimator): #arver fra base estimator for å beholde litt brukbare funksjoner

    
    def __init__(self, lr=0.1, n_iterations=10000, class_weight = None):
        
        self.class_weight = class_weight
        self.lr = lr
        self.n_iterations = n_iterations
        
        

        
    
    def fit(self, X, y):
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0 
        self.loss_history = []
        self.accuracies = []
        self.NGC = []
        if self.class_weight == "balanced": 
            classes, counts = np.unique(y, return_counts=True) #finner klassebalansen. 
            n_samples = len(y)
            n_classes = len(classes)

            weights={
                cls: n_samples/(n_classes*count) for cls, count in zip(classes, counts)
            } #utregninger av vekter på samme måte som sklearn gjør det, Wc = N/(KNc), hvor N er antall samples, K er antall klasser, Nc er antall i klassen c. 
            #minoritetsklasser blir vektet høyere. 

            sample_weights = np.array([weights[label] for label in y])
            #initialiserer vektene basert på klasse.

        elif isinstance(self.class_weight, dict): #bonus case handling for å kunne passere inn egendefinerte vekter.
            sample_weights = np.array([self.class_weight[label] for label in y])

        else: 
            sample_weights = np.ones(len(y))
        #initialiser alt til enere hvis vi ikke har definert noe annet 

        for _ in range(self.n_iterations):
            if _%100 == 0:
                print(f"traing {_}/{self.n_iterations}")
            lin_model = np.dot(X,self.weights) + self.bias 
           
            y_pred = self.sigmoid(lin_model)

            error = y_pred-y #for lesbarhet

            grad_w = (X.T @ (sample_weights*error)/np.sum(sample_weights)) #vektet gjennomsnitt i gradientberegningnene. Loss også for øvrig

            grad_b = (np.sum(sample_weights*error))/np.sum(sample_weights)

            
            self.weights = self.weights- self.lr*grad_w 
            self.bias = self.bias - self.lr*grad_b 

            eps = 1e-15 #bare for sikkerhets skyld
            y_pred_safe = np.clip(y_pred, eps, 1-eps)


            loss = -np.sum(
                sample_weights*(
                    y*np.log(y_pred_safe)
                    +(1-y)*np.log(1-y_pred_safe)
                )/np.sum(sample_weights)
            )
            self.loss_history.append(loss)
            predictions = (y_pred>=.5).astype(int)
            self.accuracies.append(
                np.mean(predictions==y)
    
            )
            if _%50==0:
                self.NGC.append(norm_gini_coeff(self.predict_proba(X)[:,1],y))
        
        return self 
       
    
    def predict_proba(self, X):
        X=np.asarray(X,dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias 
        p=self.sigmoid(linear_model)
        return np.column_stack([1-p,p])
        
        
    def predict(self, X):
        X=np.asarray(X,dtype=np.float64)
        return (self.predict_proba(X)[:,1]>=.5).astype(int)
        
    
    def sigmoid(self, z):
        z=np.asarray(z,dtype=np.float64)
        return 1/(1+np.exp(-z))



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


def test_log_loss():
    L = LogisticLoss()
    rng = np.random.default_rng(0)
    z = rng.normal(0,3,1000)
    y = rng.binomial(1,0.3,1000).astype(float)
    eps = 1e-5
    num_grad = (L.loss(y, z+eps) - L.loss(y, z-eps)) / (2*eps)
    num_hess = (L.grad(y, z+eps) - L.grad(y, z-eps)) / (2*eps)
    assert np.allclose(num_grad, L.grad(y, z), atol=1e-6)
    assert np.allclose(num_hess, L.hess(y, z), atol=1e-6)
    assert abs(L.grad(y, np.full(len(y), L.init_score(y))).sum()) < 1e-8



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value 

    def is_leaf(self):
        return self.value is None 


class Tree():
    def __init__(self, max_depth = 3, lam = 1.0, gamma = 0.0, min_child_weight = 1.0):
        self.max_depth = max_depth
        self.lam = lam 
        self.gamma = gamma 
        self.min_child_weight = min_child_weight

    def _build_tree(self,X,g,h,depth):
        if depth >= self.max_depth:
            return Node(value=self._leaf_value(g,h))
        #Todo finn beste splitt 
        
    def fit(self, X,g,h):
        X=np.asarray(X,dtype=np.float64)
        g=np.asarray(g,dtype=np.float64)
        h=np.asarray(h,dtype=np.float64)

        self.root = self._build_tree(X,g,h,depth = 0)

        return self 

    def _leaf_value(self, g,h):
        G = np.sum(g)
        H = np.sum(h)
        return -G/(H+self.lam)