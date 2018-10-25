import numpy as np

def shuffle(X, y):
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    X[:] = X[indices]
    y[:] = y[indices]
