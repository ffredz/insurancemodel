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
    def init_score(self,y):
        return np.mean(y)



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value 

    def is_leaf(self):
        return self.value is not None 


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
            gain = 0.5 * (G_L**2 / (H_L + self.lam)+ G_R**2 / (H_R + self.lam)- G**2 / (H + self.lam))
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



class GradientBoosting():
    def __init__(self, n_estimators, learning_rate, max_depth, lam, gamma, min_child_weight, early_stopping_rounds,objective):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lam = lam 
        self.gamma = gamma 
        self.min_child_weight = min_child_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective

    def fit(self, X,y,eval_set=None):
        X= np.asarray(X, dtype=float)  
        y= np.asarray(y, dtype=float)  

        #masse feilhåndtering først bare
        if X.ndim != 2 or len(X) != len(y) or len(y)==0:
            raise ValueError("Ugyldige dimensjoner for X og y")

        if self.n_estimators<1:
            raise ValueError("n_estimators må minst være 1")

        if self.objective == "logistic":
            self.loss = LogisticLoss()
            if not np.all(np.isin(y,[0,1])):
                raise ValueError("Logloss krever binærdata")
        elif self.objective == "squared":
            self.loss = SquaredLoss()
        else:
            raise ValueError("ukjent objective")

        #Valideringsdata 

        if eval_set is not None:
            X_val,y_val = eval_set
            X_val = np.asarray(X_val,dtype = np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)

            if (X_val.ndim != 2
                or X_val.shape[1] != X.shape[1]
                or len(X_val) != len(y_val)
                or len(y_val) == 0):
                raise ValueError("Ugyldig eval_set")

            if self.objective == "logistic":
                if not np.all(np.isin(y_val, [0, 1])):
                    raise ValueError("y_val må inneholde 0 og 1")

            z_val = None
        elif self.early_stopping_rounds is not None:
            raise ValueError("Early stopping krever eval_set")

        if (self.early_stopping_rounds is not None
            and self.early_stopping_rounds < 1):
            raise ValueError("early_stopping_rounds må være positiv")
        #--------------------------------------------------

        #faktisk initialisering
        self.z0_ = float(self.loss.init_score(y))
        z = np.full(len(y),self.z0_)

        if eval_set is not None:
            z_val = np.full(len(y_val),self.z0_)

        self.trees_ = []
        self.train_loss_ = []
        self.val_loss_ = []

        best_loss = np.inf 
        best_n_trees = 0
        rounds_without_improvement = 0

        #moroa begynner 
        for t in range(self.n_estimators):
            #steg 1 gradienter og hessianer 
            g = self.loss.grad(y,z)
            h = self.loss.hess(y,z)

            #2 trene nytt tre
            tree = Tree(max_depth=self.max_depth, lam=self.lam , gamma=self.gamma, min_child_weight=self.min_child_weight)
            tree.fit(X,g,h)
            self.trees.append(tree)

            #oppdater prediksjoner 
            z+=self.learning_rate*tree.predict(X)

            train_loss = self.loss.loss(y,z).mean()
            self.train_loss_.append(float(train_loss))

            #validering
            if eval_set is not None:
                z_val +=self.learning_rate*tree.predict(X_val)
                val_loss = float(self.loss.loss(y_val,z_val).mean())
                self.val_loss_.append(val_loss)
                #sjekke forbedringen

                if val_loss<best_loss-1e-12:
                    best_loss = val_loss
                    best_n_trees = len(self.trees_)
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement+=1

                if self.early_stopping_rounds is not None:
                    if rounds_without_improvement>=self.early_stopping_rounds:
                        break
        if eval_set is not None:
            self.best_iteration = best_n_trees-1
            self.best_score = best_loss 

            if self.early_stopping_rounds is not None:
                self.trees_ = self.trees_[:best_n_trees]
        else:
            self.best_iteration = len(self.trees_)-1
            self.best_score = self.train_loss_[-1]
        return self 
       
    def predict_raw(self,X,n_trees=None):
        if not hasattr(self, "trees_"):
            raise RuntimeError("Modell må trenes")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X må være 2D")
        if n_trees is None:
            n_trees = len(self.trees_)
        if (not isinstance(n_trees, (int, np.integer))
            or not 0 <= n_trees <= len(self.trees_)):
            raise ValueError("Ugyldig n_trees")

        z = np.full(len(X),self.z0_)

        for tree in self.trees_[:n_trees]:
            z+= self.learning_rate*tree.predict(X)
        return z 
    def predict_proba(self, X,n_trees=None):
        if self.objective != "logistic":
            raise ValueError(
                "predict_proba krever logistic objective"
            )
        z = self.predict_raw(X,n_trees)
        p = self.loss.sigmoid(z)
        return np.column_stack((1-p,p))
    def predict(self,X,n_trees=None):
        if self.objective=="logistic":
            p = self.predict_proba(X,n_trees)[:,1]
            return (p>=.5).astype(int)
        return self.predict_raw(X,n_trees)