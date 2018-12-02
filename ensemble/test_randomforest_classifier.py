from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ensemble.randomforest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


iris = load_digits()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)

rf_clf = RandomForestClassifier(n_estimators=10, sample_rate=1.0, max_features='auto')
# from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(n_estimators=500)

rf_clf.fit( X_train, y_train )
score = rf_clf.score(X_test, y_test)
print(score)