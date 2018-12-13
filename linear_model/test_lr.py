from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from linear_model.lr import LinearRegression
from linear_model.lr_ridge import RidgeRegression
from linear_model.lr_lasso import LassoRegression
from sklearn.preprocessing import StandardScaler

boston = load_boston()
X = boston.data
y = boston.target
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)

print("线性回归：")
rg = LinearRegression()
rg.fit(X_train, y_train)
print(rg.score(X_test, y_test))
print(rg.W)

# 在这个问题上，Ridge回归反而变差了，这是因为线性模型
# 在Boston房价数据上本来就是欠拟合的！！！
print("岭回归：")
rg = RidgeRegression(alpha=1.0)
rg.fit(X_train, y_train)
print(rg.score(X_test, y_test))
print(rg.W)

# Lasso
# alpha设置为100.0才会稀疏啊
print("Lasso回归：")
rg = LassoRegression(alpha=1.0, max_iter=1000)
rg.fit(X_train, y_train)
print(rg.score(X_test, y_test))
print(rg.W)

# sklearn的Lasso
print("sklearn的Lasso回归：")
from sklearn.linear_model import Lasso
rg = Lasso(alpha=1.0)
rg.fit(X_train, y_train)
print(rg.score(X_test, y_test))
print(rg.coef_)