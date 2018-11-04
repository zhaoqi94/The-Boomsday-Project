from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from ensemble.adaboost import AdaBoostClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

X, y = make_hastie_10_2(n_samples=12000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500, algorithm="SAMME")
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=3000)

print(X_train.shape)
print(X_test.shape)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
