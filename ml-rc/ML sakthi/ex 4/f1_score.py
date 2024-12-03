import numpy as np 
from sklearn.metrics import f1_score 
actual = np.repeat([1, 0], repeats=[160, 240]) 
pred = np.repeat([1, 0, 1, 0], repeats=[120, 40, 70, 170]) 
print(f1_score(actual, pred)) 
