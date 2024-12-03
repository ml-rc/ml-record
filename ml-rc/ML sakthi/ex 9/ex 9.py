import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import tree 
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
X,y=load_breast_cancer(return_X_y=True) 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0) 
clf=DecisionTreeClassifier(random_state=0) 
clf.fit(X_train,y_train) 
y_train_predicted=clf.predict(X_train) 
y_test_predicted=clf.predict(X_test) 
train_acc=accuracy_score(y_train,y_train_predicted) 
test_acc=accuracy_score(y_test,y_test_predicted) 
print("accuracy of training dataset:",train_acc) 
print("accuracy of test dataset:",test_acc) 
plt.figure(figsize=(16,8)) 
tree.plot_tree(clf) 
plt.show() 
path=clf.cost_complexity_pruning_path(X_train,y_train) 
ccp_alphas,impurities=path.ccp_alphas,path.impurities 
print("ccp alpha wil give list of values :",ccp_alphas) 
print("***********************************************************") 
print("Impurities in Decision Tree :",impurities) 
clfs=[] 
for ccp_alpha in ccp_alphas:
 clf=DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
 clf.fit(X_train,y_train)
 clfs.append(clf) 
print("Last node in Decision tree is {} and ccp_alpha for last node is {}".format(clfs[
1].tree_.node_count,ccp_alphas[-1])) 
train_scores = [clf.score(X_train, y_train) for clf in clfs] 
test_scores = [clf.score(X_test, y_test) for clf in clfs] 
fig, ax = plt.subplots() 
ax.set_xlabel("alpha") 
ax.set_ylabel("accuracy") 
ax.set_title("Accuracy vs alpha for training and testing sets") 
ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post") 
ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post") 
ax.legend() 
plt.show() 
clf=DecisionTreeClassifier(random_state=0,ccp_alpha=0.02) 
clf.fit(X_train,y_train) 
plt.figure(figsize=(12,8)) 
tree.plot_tree(clf,rounded=True,filled=True) 
plt.show() 
acc=accuracy_score(y_test,clf.predict(X_test)) 
print("accuracy of post-pruning operation:",acc) 
clf=DecisionTreeClassifier(criterion= 'gini',max_depth= 
17,min_samples_leaf=3,min_samples_split= 12,splitter= 'random') 
clf.fit(X_train,y_train) 
plt.figure(figsize=(20,12)) 
tree.plot_tree(clf,rounded=True,filled=True) 
plt.show() 
y_predicted=clf.predict(X_test) 
accuracy=accuracy_score(y_test,y_predicted) 
print("accuracy of pre-pruning operation:",accuracy)
