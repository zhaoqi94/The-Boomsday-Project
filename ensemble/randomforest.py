from tree.decisiontree import DecisionTreeClassifier,DecisionTreeRegressor
# 用自己的决策树真的时蜗牛般的速度 效果还差！
# from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from ensemble.bagging import BaggingClassifier,BaggingRegressor

class RandomForestClassifier(BaggingClassifier):
    def __init__(self, n_estimators=10,
                 sample_rate=1.0, bootstrap=True,
                 criterion='gini', max_depth=None,
                 max_features='auto', min_samples_split=2,
                 min_samples_leaf=1
            ):
        super(RandomForestClassifier, self).__init__(
            DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            ),
            n_estimators,sample_rate, bootstrap=bootstrap)


class RandomForestRegressor(BaggingRegressor):
    def __init__(self, n_estimators=10,
                 sample_rate=1.0, bootstrap=True,
                 criterion='mse', max_depth=None,
                 max_features='auto', min_samples_split=2,
                 min_samples_leaf=1
            ):
        super(RandomForestRegressor, self).__init__(
            DecisionTreeRegressor(
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            ),
            n_estimators,sample_rate, bootstrap=bootstrap)