import numpy as np 
import pandas as pd 
N = 10000
mu = 10
std = 2
population_df = np.random.normal(mu,std,N) 
def systematic_sampling(df, step):
    id = pd.Series(np.arange(1,len(df),1)) 
    df = pd.Series(df) 
    df_pd = pd.concat([id, df], axis = 1) 
    df_pd.columns = ["id", "data"] 
    selected_index = np.arange(1,len(df),step) 
    systematic_sampling = df_pd.iloc[selected_index] 
    return(systematic_sampling) 
n = 10
step = int(N/n) 
sample = systematic_sampling(population_df, step) 
print(sample)
