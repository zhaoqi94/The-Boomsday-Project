from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from ensemble.randomforest import RandomForestRegressor
from tree.decisiontree import DecisionTreeRegressor
# from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)

start_time = datetime.now()
dct_rg = DecisionTreeRegressor(max_features=None)
dct_rg.fit(X_train, y_train)
score = dct_rg.score(X_test, y_test)
print(score)
end_time = datetime.now()
print((end_time-start_time).seconds)

# 需要注意的是，在sklearn的RandomForestRegressor中max  _features="auto"就是全部的特征
# 和随机森林分类不一样！
# 但是目前我的实现max_features="auto"仍然和分类是一样的！
# 所以效果会比sklearn差一点
# 巨慢无比！！！
res = 0
for i in range(30):
    start_time = datetime.now()
    print("My RandomForestRegressor:")
    rf_rg = RandomForestRegressor(n_estimators=10, sample_rate=1.0, max_features=None)
    rf_rg.fit(X_train, y_train)
    score = rf_rg.score(X_test, y_test)
    print(score)
    res += score
    end_time = datetime.now()
    print((end_time-start_time).seconds)
    # print(y_test[0:10])
    # print(rf_rg.predict(X_test)[0:10])
print(res/30)

res = 0
for i in range(30):
    print("sklearn RandomForestRegressor:")
    from sklearn.ensemble import RandomForestRegressor
    rf_rg = RandomForestRegressor(n_estimators=10, max_features=None)
    rf_rg.fit(X_train, y_train)
    score = rf_rg.score(X_test, y_test)
    print(score)
    res += score
    # print(y_test[0:10])
    # print(rf_rg.predict(X_test)[0:10])
print(res/30)