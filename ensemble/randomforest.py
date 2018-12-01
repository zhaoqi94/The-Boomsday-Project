from tree.dct_classification import DecisionTreeClassifier
# 用自己的决策树真的时蜗牛般的速度 效果还差！
# from sklearn.tree import DecisionTreeClassifier
from ensemble.bagging import BaggingClassifier

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

