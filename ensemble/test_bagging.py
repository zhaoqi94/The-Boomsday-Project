from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# 导入数据
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# base_estimator: 基学习器
# n_estimators: 基学习器的个数
# max_samples：每个基学习器中的样本数，如果是整形，则就是样本个数；如果是float，则是样本个数占所有训练集样本个数的比例
# bootstrap: 是否采用有放回抽样(bagging)，为True表示采用，否则为pasting。默认为True
bag_clf = BaggingClassifier( DecisionTreeClassifier(), n_estimators=500, sample_rate=0.2, bootstrap=True, )

bag_clf.fit( X_train, y_train )
score = bag_clf.score(X_test, y_test)
print(score)
# dcf = bag_clf.decision_function(X_test)
# print(dcf)



###############可视化的代码####################


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    print(x1.ravel().shape)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    print(X_new.shape)
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    return

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

plt.figure(figsize=(8,3))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()