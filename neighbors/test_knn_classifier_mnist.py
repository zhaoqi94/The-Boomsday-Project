from sklearn.datasets import fetch_mldata
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target


X = X / 255 - np.mean(X,axis=0)

pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)

X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index],y_train[shuffle_index]
X_train = X_train
X_test = X_test
from scipy.spatial import KDTree

X_test = X_test[0:100]
y_test = y_test[0:100]

# print(X_test[0])
# print(abs(X_train[0:10]-X_test[0]))
# print(np.var(X_train,axis=0))

# from sklearn.neighbors import KNeighborsClassifier
# print("sklearn KNN Brute")
# clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
# start_time = datetime.now()
# clf.fit(X_train, y_train)
# mid_time = datetime.now()
# score = clf.score(X_test, y_test)
# end_time = datetime.now()
# print("准确率",score)
# print("训练时间:",(mid_time-start_time).seconds)
# print("测试时间:",(end_time-mid_time).seconds)
#
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
#
# print("sklearn KNN KDTree")
# start_time = datetime.now()
# clf.fit(X_train, y_train)
# mid_time = datetime.now()
# score = clf.score(X_test, y_test)
# end_time = datetime.now()
# print("准确率",score)
# print("训练时间:",(mid_time-start_time).seconds)
# print("测试时间:",(end_time-mid_time).seconds)

# from neighbors.knn import KNeighborsClassifier
# print("我的 KNN 暴力方法")
# clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
# start_time = datetime.now()
# clf.fit(X_train, y_train)
# mid_time = datetime.now()
# score = clf.score(X_test, y_test)
# end_time = datetime.now()
# print("准确率",score)
# print("训练时间:",(mid_time-start_time).seconds)
# print("测试时间:",(end_time-mid_time).seconds)
#
#
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
# print(clf.tree.depth_box_compare)
# print(clf.tree.depth_box_compare_fail)
print(clf.tree.total_count / X_test.shape[0])



