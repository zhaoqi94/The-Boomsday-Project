import numpy as np
import six
import numbers

class Tree:
    def __init__(self, attr_index=-1, split_value=None,
                 results=None, left_sub_tree=None, right_sub_tree=None):
        self.attr_index = attr_index
        self.split_value = split_value
        # predicate: used to determine samples belong to left or right
        # support discrete and continuous variables
        # self.predicate = None
        self.results = results
        self.left_sub_tree = left_sub_tree
        self.right_sub_tree = right_sub_tree


class TreeBuilder:
    def __init__(self, max_depth, max_features, min_samples_split):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_features_ = None

    def build(self, X, y):
        # 处理max_features,即树中每个节点选择属性时属性集合所含属性的最大值
        # 这个功能可以方便随机森林的随机属性选择
        max_features = X.shape[1]
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                self.max_features_ = max(1, int(np.sqrt(max_features)))
            elif self.max_features == "sqrt":
                self.max_features_ = max(1, int(np.sqrt(max_features)))
            elif self.max_features == "log2":
                self.max_features_ = max(1, int(np.log2(max_features)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            self.max_features_ = max_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            self.max_features_ = self.max_features
        else:  # float
            if self.max_features > 0.0:
                self.max_features_ = max(1,
                                   int(self.max_features * max_features))
            else:
                self.max_features_ = 0

        # print(self.max_features_)

        return self._build(X, y, 1)

    def _build(self, X, y, depth, ):
        sample_num = X.shape[0]
        feature_num = X.shape[1]

        if (depth > self.max_depth) and (sample_num < self.min_samples_split):
            return Tree(results=majority(y))

        # 选择候选特征集
        selected_features = np.random.choice(feature_num, self.max_features_, replace=False)

        current_impurity = gini(y)

        best_split = {
            "best_gain": 0.0,
            "best_attr_index": None,
            "best_split_value": None,
            "best_left_X": None,
            "best_left_y": None,
            "best_right_X": None,
            "best_right_y": None,
        }

        # for j in range(feature_num):
        for j in selected_features:
            for i in range(sample_num):
                current_attr_index = j
                current_split_value = X[i, j]
                left_X, left_y, right_X, right_y = split(X, y, current_attr_index, current_split_value)

                impurity = 0
                impurity += gini(left_y) * left_X.shape[0] / sample_num
                impurity += gini(right_y) * right_X.shape[0] / sample_num
                current_gain = current_impurity - impurity

                if current_gain > best_split["best_gain"]:
                    best_split["best_gain"] = current_gain
                    best_split["best_attr_index"] = current_attr_index
                    best_split["best_split_value"] = current_split_value
                    best_split["left_X"] = left_X
                    best_split["left_y"] = left_y
                    best_split["right_X"] = right_X
                    best_split["right_y"] = right_y

        if best_split["best_gain"] == 0:
            return Tree(results=majority(y))


        depth += 1
        left_sub_tree = self._build(
            best_split["left_X"],
            best_split["left_y"],
            depth)
        right_sub_tree = self._build(
            best_split["right_X"],
            best_split["right_y"],
            depth)

        return Tree(
            attr_index=best_split["best_attr_index"],
            split_value=best_split["best_split_value"],
            results=None,
            left_sub_tree=left_sub_tree,
            right_sub_tree=right_sub_tree,
        )


def gini(y):
    if y.shape[0] == 0:
        return 0.0

    gini = 1.0
    unique_item, unique_counts = np.unique(y, return_counts=True)
    N = y.shape[0]
    gini -= np.sum(np.square(unique_counts)) / (N*N)
    return gini

# choose the most frequent label as the result
def majority(y):
    unique_item, unique_counts = np.unique(y, return_counts=True)
    return unique_item[np.argmax(unique_item)]


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
