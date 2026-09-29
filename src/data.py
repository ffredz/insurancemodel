import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


def loader(df):
    missing_flags = (df==-1).add_suffix("_was_missing")
    df = df.replace(-1, np.nan)
    df=pd.concat([df,missing_flags],axis = 1)
    return df 



def split(df,train_prop, val_prop, random_state = 42):
    if not np.isclose(train_prop + val_prop, 1.0):
        raise ValueError("invalid split")

    df_train, df_val = train_test_split(
        df,test_size=val_prop,stratify=df['target'],random_state=random_state
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)