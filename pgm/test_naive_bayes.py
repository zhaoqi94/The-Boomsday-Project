from pgm.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape)
print(X_test.shape)


clf = GaussianNB()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
print(y_test)
print(clf.predict(X_test))



X = np.random.randint(2, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])
clf = BernoulliNB()
clf.fit(X, y)
print(clf.predict(X))


# 新闻分类
from sklearn.datasets import fetch_20newsgroups   #fetch_20newsgroups是新闻数据抓取器
news=fetch_20newsgroups(subset='all')   #fetch_20newsgroups即时从互联网下载数据
print(len(news.data))

X_train,X_test,y_train,y_test = train_test_split(news.data, news.target, test_size=0.4, random_state=33)

# print(type(X_train))

from sklearn.feature_extraction.text import CountVectorizer
# 特征抽取,将文本特征向量化
# 转换过后为什么是sparse matrix
# <class 'scipy.sparse.csr.csr_matrix'>
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

# print(type(X_train))

# from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)

from sklearn.metrics import classification_report
print('The accuracy of Navie Bayes Classifier is',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))