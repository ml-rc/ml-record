from sklearn.linear_model import LinearRegression
import pandas as pd 
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv" 
data = pd.read_csv(url) 
model = LinearRegression() 
X, y = data[["mpg", "wt", "drat", "qsec"]], data['hp']
model.fit(X,y)
print(1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)) 
