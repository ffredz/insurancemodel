import numpy as np 
import matplotlib.pyplot as plt 
from src.data import loader, split 
from src.metrics import norm_gini_coeff, lorenz_plot
from src.models import LogisticRegression
import pandas as pd 

df =pd.read_csv('data/raw/train.csv')
df = loader(df)
df_train, df_val = split(df,.8,.2,42)
df_train_X, df_train_y = df_train.drop('target',axis=1), df_train['target']
df_val_X, df_val_y = df_val.drop('target',axis=1), df_val['target']
model = LogisticRegression(lr=.1,n_iterations=1000,class_weight='balanced')
model.fit(df_train_X, df_train_y)
n=model.n_iterations
print(model.NGC[0])
print(model.NGC[-1])
print(norm_gini_coeff(model.predict_proba(df_val_X),np.asarray(df_val_y,dtype=np.float64)))
