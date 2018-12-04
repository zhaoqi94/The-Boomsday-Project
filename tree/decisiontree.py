import numpy as np
import tree.tree_utils as tree_utils
import tree.criterion as criterion
import tree.splitter as splitter
import six
import numbers

CRITERIA_CLF = {"gini": criterion.Gini}
CRITERIA_REG = {"mse": criterion.MSE}
SPLITTERS = {"best": splitter.BestSplitter}

# TODO: 接下来的任务 1.模块化
# TODO: 接下来的任务 2.梯度提升树

class BaseDecisionTree:
    # @abstractmethod
    def __init__(self, criterion, splitter, max_depth, max_features,
                 min_samples_split, min_samples_leaf, is_classifier):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.is_classifier = is_classifier
        # process
        if self.max_depth == None:
            self.max_depth = np.inf
        self.tree = None

    def fit(self, X, y):
        # 处理max_depth
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        # 处理min_samples_split
        min_samples_split = self.min_samples_split
        # 处理min_samples_leaf
        min_samples_leaf = self.min_samples_leaf

        # 处理max_features,即树中每个节点选择属性时属性集合所含属性的最大值
        # 这个功能可以方便随机森林的随机属性选择
        n_features = X.shape[1]
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * n_features))
            else:
                max_features = 0

        # 处理criterion
        if self.is_classifier:
            criterion = CRITERIA_CLF[self.criterion]()
        else:
            criterion = CRITERIA_REG[self.criterion]()
        # 构造splitter
        splitter = SPLITTERS[self.splitter](criterion, max_features, min_samples_leaf)

        builder = tree_utils.TreeBuilder(splitter,
                        max_depth, min_samples_split, min_samples_leaf)

        self.tree = builder.build(X, y)



        return self

    def predict(self, X):
        return self.tree.predict(X)

    # TODO: 输出决策路径
    def decision_path(self):
        pass


# version1.0 of DecisionTreeClassifier
# not support sample weights
class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion="gini", splitter="best", max_depth=None,max_features=None,
                 min_samples_split=2, min_samples_leaf=1):
        super(DecisionTreeClassifier, self).__init__(criterion,splitter,
                max_depth, max_features, min_samples_split, min_samples_leaf,True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion="mse", splitter="best", max_depth=None,max_features=None,
                 min_samples_split=2, min_samples_leaf=1):
        super(DecisionTreeRegressor, self).__init__(criterion,splitter,
                max_depth, max_features, min_samples_split, min_samples_leaf, False)

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
