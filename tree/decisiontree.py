import numpy as np
import tree.tree_utils as tree_utils


# TODO: 接下来的任务 1.模块化
# TODO: 接下来的任务 2.梯度提升树

class BaseDecisionTree:
    # @abstractmethod
    def __init__(self, criterion, max_depth, max_features,
                 min_samples_split, min_samples_leaf):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # process
        if self.max_depth == None:
            self.max_depth = np.inf
        self.tree = None

    def fit(self, X, y):
        builder = tree_utils.TreeBuilder(self.criterion, self.max_depth, self.max_features,
                                                   self.min_samples_split,self.min_samples_leaf)
        self.tree = builder.build(X, y)

        return self

    def predict(self, X):
        y_pred = []

        for i in range(X.shape[0]):
            tree = self.tree
            while(True):
                if tree.results != None:
                    y_pred.append(tree.results)
                    break

                if X[i][tree.attr_index] <= tree.split_value:
                    tree = tree.left_sub_tree
                else:
                    tree = tree.right_sub_tree

        return np.array(y_pred)

    # TODO: 输出决策路径
    def decision_path(self):
        pass


# version1.0 of DecisionTreeClassifier
# not support sample weights
class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion="gini", max_depth=None,max_features=None,
                 min_samples_split=2, min_samples_leaf=1):
        super(DecisionTreeClassifier, self).__init__(criterion,
                max_depth, max_features, min_samples_split, min_samples_leaf)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion="mse", max_depth=None,max_features=None,
                 min_samples_split=2, min_samples_leaf=1):
        super(DecisionTreeRegressor, self).__init__(criterion,
                max_depth, max_features, min_samples_split, min_samples_leaf)

    # def fit(self, X, y):
    #     pass
    #
    # def predict(self, X):
    #     pass

    # 参考sklearn
    # 并不是MSE
    # the coefficient of determination R^2 of the prediction
    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X)-np.array(y))) / np.sum(np.square(np.mean(y)-np.array(y)))
