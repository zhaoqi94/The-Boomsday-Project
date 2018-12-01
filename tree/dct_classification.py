import numpy as np
import tree.tree_utils as tree_utils

# version1.0 of DecisionTreeClassifier
# not support sample weights


class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None,max_features=None,
                 min_samples_split=1, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # process
        if self.max_depth == None:
            self.max_depth = 100000
        self.tree_builder = tree_utils.TreeBuilder(self.max_depth, self.max_features, self.min_samples_split)
        self.tree = None

    def fit(self, X, y):
        self.tree = self.tree_builder.build(X, y)

    def predict(self, X):
        y_pred = []

        for i in range(X.shape[0]):
            tree = self.tree
            count = 0
            while(True):
                if tree.results != None:
                    y_pred.append(tree.results)
                    # print(count)
                    break

                if X[i][tree.attr_index] <= tree.split_value:
                    tree = tree.left_sub_tree
                else:
                    tree = tree.right_sub_tree
                count += 1

        return y_pred

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def decision_path(self):
        pass
