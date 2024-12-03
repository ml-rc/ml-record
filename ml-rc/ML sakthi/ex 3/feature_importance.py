from sklearn.ensemble import RandomForestClassifier 
from sklearn import datasets 
import numpy as np 
import matplotlib.pyplot as plt 
iris = datasets.load_iris() 
X = iris.data 
y = iris.target 
clf = RandomForestClassifier(random_state=0, n_jobs=-1) 
model = clf.fit(X, y) 
importances = model.feature_importances_ 
indices = np.argsort(importances)[::-1] 
names = [iris.feature_names[i] for i in indices] 
plt.figure() 
plt.title("Feature Importance") 
plt.bar(range(X.shape[1]), importances[indices]) 
plt.xticks(range(X.shape[1]), names, rotation=90) 
plt.show()
