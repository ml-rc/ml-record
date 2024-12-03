from sklearn.metrics import mean_absolute_error as mae 
actual = [2, 3, 5, 5, 9] 
calculated = [3, 3, 8, 7, 6] 
error = mae(actual, calculated) 
print("Mean absolute error : " + str(error)) 
