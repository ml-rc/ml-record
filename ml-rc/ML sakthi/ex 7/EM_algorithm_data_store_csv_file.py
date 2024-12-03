from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width', 'Class']
dataset = pd.read_csv("E:\iris.csv", names=names)

X = dataset.iloc[:, :-1]
label = {'Iris-Setosa': 0,'Iris-Versicolor': 1, 'Iris-Virginica': 2}
y = [label[c] for c in dataset.iloc[:, -1]]

plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])
plt.title('Real')
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y])
plt.show()

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
y_cluster_gmm = gmm.predict(X)

plt.title('GMM Classification')
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_cluster_gmm])
plt.show()

print('The adjusted rand index of EM: ', metrics.adjusted_rand_score(y, y_cluster_gmm))
print('The silhouette score of EM: ', metrics.silhouette_score(X, y_cluster_gmm))
