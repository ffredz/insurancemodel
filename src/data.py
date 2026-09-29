def loader(df):
    missing_flags = (df==-1).add_suffix("_was_missing")
    df = df.replace(-1, np.nan)
    df=pd.concat([df,missing_flags],axis = 1)
    return df 
