import numpy as np
N=10000
mu=10
std=2
population_df=np.random.normal(mu,std,N)
def random_sampling(df,n):
    random_sample=np.random.choice(df,replace=False, size=n)
    return (random_sample)
randomsample=random_sampling(population_df,N)
print(randomsample)
