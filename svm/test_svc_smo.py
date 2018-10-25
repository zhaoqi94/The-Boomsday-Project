from sklearn.datasets import load_breast_cancer
from svm.svc_smo import *
from datetime import datetime
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
data = dataset.data[:]
dataset.target[dataset.target == 0] = -1
label = dataset.target[:]

scaler = StandardScaler()
data = scaler.fit_transform(data)

# 对参数太敏感了！！！
# 线性的为啥还不如感知机！！！
#
clf = SVC(C=1.0, kernel="linear",max_iter=10000, tol=0.001, eps=0.01)
# clf = SVC(C=10.0, kernel="gaussian",max_iter=10000, tol=0.01, eps=0.01)
# clf = SVC(C=1.0, kernel="poly", max_iter=10000, tol=0.01, eps=0.01)
# clf = SVC(C=1.0, kernel="sigmoid", max_iter=10000, tol=0.01, eps=0.01)

print("Start training svm......")
starttime = datetime.now()
clf.fit(data, label)
endtime = datetime.now()
print("Training Time:%.2f" % (endtime - starttime).seconds)
print("End training svm")

accuracy = clf.score(data, label)
print("Training set accuracy=%2.2f%%" % (accuracy*100))