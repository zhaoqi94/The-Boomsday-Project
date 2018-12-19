from neighbors.knn import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape)
print(X_test.shape)

# KNN 暴力解法
print("KNN 暴力解法")
clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# KNN KDTree
print("KNN KDTree")
clf = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# KNN BallTree
print("KNN BallTree")