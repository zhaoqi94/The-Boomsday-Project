import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from linear_model.logistic import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape)
print(X_test.shape)
clf = LogisticRegression(learning_rate=0.01, max_iter=1000 ,penalty="l2", alpha=0.0, fit_intercept=True)

# LogisticRegression in sklearn do not use SGD
# it uses other optimal solver

# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier(loss='log', n_iter=1000,learning_rate="constant", eta0=0.001,penalty="none")
# clf = LogisticRegression(C=1000000)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))