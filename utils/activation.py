import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x,axis=-1,keepdims=True)
    x = np.exp(x)
    x /= np.sum(x,axis=-1,keepdims=True)
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
