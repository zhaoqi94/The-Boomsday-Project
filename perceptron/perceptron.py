import numpy as np
import utils

# Binary Perceptron
class Perceptron:
    def __init__(self, max_iter=100, eta=0.1, shuffle=True, tol=None):
        self.max_iter = max_iter
        self.eta = eta
        self.shuffle = shuffle
        self.tol = tol
        self.W = None
        self.b = None

    def fit(self, X, y):
        # X: [n_samples, input_dim]
        n_samples,input_dim = X.shape
        self.W = np.random.randn(input_dim)
        self.b = 0.0
        # pass dataset for max_iter runs
        for i in range(self.max_iter):
            # suffle
            if self.shuffle:
                utils.shuffle(X, y)
            for j in range(n_samples):
                score = self.predict(X[j])
                self.W -= self.eta * (score - y[j]) * X[j]
                self.b -= self.eta * (score - y[j]) * 1

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.int32)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / X.shape[0]

    def decision_function(self, X):
        return np.dot(X,self.W) + self.b


if __name__=='__main__':
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,1])
    pp = Perceptron(max_iter=100, eta=0.1)
    pp.fit(X, y)
    print("0 and 0:", pp.predict([0,0]))
    print("0 and 1:", pp.predict([0,1]))
    print("1 and 0:", pp.predict([1,0]))
    print("1 and 1:", pp.predict([1,1]))
    print("score:", pp.score(X, y))

    ###############################################

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    dataset = load_breast_cancer()
    X = dataset.data[:]
    y = dataset.target[:]
    print(X.shape,y.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pp = Perceptron(max_iter=100, eta=0.1)
    pp.fit(X, y)
    print("score:", pp.score(X, y))
