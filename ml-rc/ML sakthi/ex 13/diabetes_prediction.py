import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("diabetes.csv")
print(data.head())
print(data.dtypes)
print(data.describe())
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=7)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
model_score = model.score(X_test, Y_test)
print("Model Score:", model_score)
print("Confusion Matrix:\n", metrics.confusion_matrix(Y_test, Y_predict))
