from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
features = iris.data  
labels = iris.target

clf=KNeighborsClassifier()
clf.fit(features,labels)
preds=clf.predict([[1,1,1,1]])
print(preds)
preds=clf.predict([[3,1,5,8]])
print(preds)
