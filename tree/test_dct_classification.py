from tree.dct_classification import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape)
print(X_test.shape)

clf = DecisionTreeClassifier(max_features='sqrt')
clf.fit(X_train, y_train)
# print(clf.predict(X_test))
print(clf.score(X_test, y_test))
