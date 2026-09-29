import pandas as pd 
import numpy as np 


def norm_gini_coeff(predicted_probs, values): 
    order = np.argsort(predicted_probs)[::-1] 
    sorted_labels = values[order]
    n = len(predicted_probs)
    positives = 0
    x = np.arange(0,1+1/n, 1/n)
    true_positives = sum(k == 1 for k in values)
    L,L_perf = np.zeros(n+1), np.zeros(n+1)

    print(true_positives)
    for i in range(1,n+1):
        if sorted_labels[i-1]==1:
            positives+=1
        L[i] = positives/true_positives
        L_perf[i] = min(i,true_positives)/true_positives
    print(L)
    print(L_perf)
    area, area_perf = np.trapezoid(L, x=x), np.trapezoid(L_perf, x=x)

    print(area)
    print(area_perf)
    G = area-.5
    G_perf = area_perf-.5
    return G/G_perf 

    
  

df_test = {
    'val':[1,0,1,0,0],
    'pred':[.6,.5,.9,.8,.3]}
df_test = pd.DataFrame(data=df_test)

pred = df_test['pred']
val = df_test['val']

print(norm_gini_coeff(pred,val))