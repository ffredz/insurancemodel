import numpy as np 
import matplotlib.pyplot as plt 
from src.data import loader, split 
from src.metrics import norm_gini_coeff, lorenz_plot
from src.models import LogisticRegression
import pandas as pd 
from src.features import build_preprocessor 
from sklearn.linear_model import LogisticRegression as LR


df =pd.read_csv('data/raw/train.csv')
df = loader(df)
df_train, df_val = split(df,.8,.2,42)
df_train_X, df_train_y = df_train.drop('target',axis=1), df_train['target']
df_val_X, df_val_y = df_val.drop('target',axis=1), df_val['target']

preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(df_train_X) #lærer og bruker
X_val = preprocessor.transform(df_val_X) #aldri fit på validering



model = LogisticRegression(lr=.1,n_iterations=100,class_weight='balanced')
model.fit(X_train, df_train_y)
n=model.n_iterations

print(norm_gini_coeff(model.predict_proba(X_val)[:,1],np.asarray(df_val_y,dtype=np.float64)))


model_unbalanced = LogisticRegression(lr=.1, n_iterations = 100)
model_unbalanced.fit(X_train, df_train_y)

print(norm_gini_coeff(model_unbalanced.predict_proba(X_val)[:,1],np.asarray(df_val_y,dtype=np.float64)))

model_theoretical = LR(class_weight='balanced',)
model_theoretical.fit(X_train, df_train_y)
print(norm_gini_coeff(model_theoretical.predict_proba(X_val)[:,1],np.asarray(df_val_y, dtype=np.float64)))