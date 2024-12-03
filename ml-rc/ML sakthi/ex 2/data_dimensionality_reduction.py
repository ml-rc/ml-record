from sklearn import datasets 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import seaborn as sns 
import matplotlib.pyplot as plt 
iris = datasets.load_iris() 
df = pd.DataFrame(iris['data'], columns = iris['feature_names']) 
print(df.head()) 
scalar = StandardScaler() 
scaled_data = pd.DataFrame(scalar.fit_transform(df)) 
print(scaled_data) 
sns.heatmap(scaled_data.corr())
plt.show() 
pca = PCA(n_components = 3) 
pca.fit(scaled_data) 
data_pca = pca.transform(scaled_data) 
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3']) 
print(data_pca.head()) 
sns.heatmap(data_pca.corr()) 
plt.show()
