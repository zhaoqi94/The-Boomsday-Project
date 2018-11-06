import numpy as np
from utils.activation import softmax
from sklearn import preprocessing


class LogisticRegression:
    def __init__(self, learning_rate=1, max_iter=100 ,penalty="l2", alpha=0.1, fit_intercept=True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty
        self.alpha = alpha
        self.W = None
        self.b = None
        self.classes = None
        # whether to fit bias
        self.fit_intercept = fit_intercept


    def fit(self, X, y):
        # np.unique(): has benn sorted
        self.classes = np.unique(y)
        y = preprocessing.OneHotEncoder().fit_transform(y[:, np.newaxis]).toarray()
        sample_num = X.shape[0]
        input_dim = X.shape[1]
        class_num = self.classes.shape[0]
        self.W = np.random.rand(input_dim, class_num ) * 0.0001
        self.b = np.zeros(class_num)
        # Gradient Descent
        for i in range(self.max_iter):
            logits = np.dot(X, self.W) + self.b
            probs = softmax(logits)
            dW = np.dot(X.T, probs - y) / sample_num
            self.W -= self.learning_rate * dW
            if self.fit_intercept:
                db = np.sum(probs - y, axis=0) / sample_num
                self.b -= self.learning_rate * db
            # regularization
            if self.penalty == "l2":
                self.W -= self.learning_rate * self.alpha * self.W


    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        return softmax(np.dot(X, self.W) + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def decision_function(self, X):
        return np.dot(X, self.W) + self.b