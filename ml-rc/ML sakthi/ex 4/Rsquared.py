from sklearn.metrics import r2_score 
y =[10, 20, 30] 
f =[10, 20, 30] 
r2 = r2_score(y, f) 
print('r2 score for perfect model is', r2) 
y =[10, 20, 30] 
f =[20, 20, 20] 
r2 = r2_score(y, f) 
print('r2 score for a model which predicts mean value always is', r2) 
y = [10, 20, 30] 
f = [30, 10, 20] 
r2 = r2_score(y, f) 
print('r2 score for a worse model is', r2) 

