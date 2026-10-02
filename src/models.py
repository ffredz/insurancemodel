import numpy as np 
from sklearn.base import BaseEstimator
from src.metrics import norm_gini_coeff




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
            predictions = (y>=.5).astype(int)
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
        