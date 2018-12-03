import numpy as np
from abc import ABCMeta, abstractmethod
from tree.criterion import Criterion

class Splitter(metaclass=ABCMeta):
    """
    Spliter类抽象出划分属性的过程
    """
    @abstractmethod
    def __init__(self, criterion, max_features,
                  min_samples_leaf):
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.X = None
        self.y = None

    def init(self, X, y):
        self.X = X
        self.y = y

    def node_reset(self, X, y):
        self.X = X
        self.y = y

    def node_split(self):
        pass

    def node_impurity(self):
        return self.criterion.node_impurity()

    def node_value(self):
        return self.criterion.node_value(self.y)


class BestSplitter(Splitter):
    def __init__(self, criterion, max_features,
                  min_samples_leaf):
        super(BestSplitter, self).__init__(
            criterion, max_features, min_samples_leaf
        )

    def node_split(self):
        X,y = self.X,self.y
        sample_num, feature_num = X.shape

        # 选择候选特征集
        selected_features = np.random.choice(feature_num, self.max_features, replace=False)

        current_impurity = self.criterion.node_impurity(y)

        best_split = {
            "best_gain": -np.inf,
            "best_attr_index": None,
            "best_split_value": None,
            "best_left_X": None,
            "best_left_y": None,
            "best_right_X": None,
            "best_right_y": None,
        }


        for j in selected_features:
            # 找到切分点
            split_values = list(set(X[:, j]))
            split_values.extend([-np.inf, np.inf])
            split_values.sort()
            split_values = [(split_values[i] + split_values[i + 1]) / 2.0
                            for i in range(len(split_values) - 1)]

            for i in split_values:
                current_attr_index = j
                current_split_value = i
                left_X, left_y, right_X, right_y = split(X, y, current_attr_index, current_split_value)

                impurity = 0
                impurity += self.criterion.node_impurity(left_y) * left_X.shape[0] / sample_num
                impurity += self.criterion.node_impurity(right_y) * right_X.shape[0] / sample_num
                current_gain = current_impurity - impurity

                if current_gain >= best_split["best_gain"]:
                    best_split["best_gain"] = current_gain
                    best_split["best_attr_index"] = current_attr_index
                    best_split["best_split_value"] = current_split_value
                    best_split["left_X"] = left_X
                    best_split["left_y"] = left_y
                    best_split["right_X"] = right_X
                    best_split["right_y"] = right_y

        return best_split

# split the dataset (X, y) to two parts w.r.t (current_attr_index, current_split_value)
def split(X, y, current_attr_index, current_split_value):
    left_X = []
    left_y = []
    right_X = []
    right_y = []
    for i in range(X.shape[0]):
        if X[i][current_attr_index] <= current_split_value:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    left_X = np.array(left_X)
    left_y = np.array(left_y)
    right_X = np.array(right_X)
    right_y = np.array(right_y)
    return left_X, left_y, right_X, right_y
