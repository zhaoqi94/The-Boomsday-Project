from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np

# Generate dataset
X, y = make_blobs(centers=3, n_samples=100000, random_state=1,n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(np.var(X, axis=0))

X_test = X_test[0:1000]
y_test = y_test[0:1000]

from sklearn.neighbors import KNeighborsClassifier
print("sklearn KNN Brute")
clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
start_time = datetime.now()
clf.fit(X_train, y_train)
mid_time = datetime.now()
score = clf.score(X_test, y_test)
end_time = datetime.now()
print("准确率",score)
print("训练时间:",(mid_time-start_time).seconds)
print("测试时间:",(end_time-mid_time).seconds)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree",leaf_size=1)

print("sklearn KNN KDTree")
start_time = datetime.now()
clf.fit(X_train, y_train)
mid_time = datetime.now()
score = clf.score(X_test, y_test)
end_time = datetime.now()
print("准确率",score)
print("训练时间:",(mid_time-start_time).seconds)
print("测试时间:",(end_time-mid_time).seconds)

from neighbors.knn import KNeighborsClassifier
print("我的 KNN 暴力方法")
clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
start_time = datetime.now()
clf.fit(X_train, y_train)
mid_time = datetime.now()
score = clf.score(X_test, y_test)
end_time = datetime.now()
print("准确率",score)
print("训练时间:",(mid_time-start_time).seconds)
print("测试时间:",(end_time-mid_time).seconds)


from neighbors.knn import KNeighborsClassifier
print("我的 KNN KDTree")
clf = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
start_time = datetime.now()
clf.fit(X_train, y_train)
mid_time = datetime.now()
score = clf.score(X_test, y_test)
end_time = datetime.now()
print("准确率",score)
print("训练时间:",(mid_time-start_time).seconds)
print("测试时间:",(end_time-mid_time).seconds)
