from sklearn.metrics import mean_squared_error 
Y_true = [1,1,2,2,4] 
Y_pred = [0.6,1.29,1.99,2.69,3.4] 
MSE = mean_squared_error(Y_true,Y_pred) 
print(MSE) 

