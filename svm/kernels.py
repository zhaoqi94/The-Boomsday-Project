import numpy as np


def linear_kernel(x, y, b=0):
    # return (x * y).sum(-1)
    return np.dot(x, y.T) + b

def gaussian_kernel(x, y, sigma=2.0):
    # return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- np.linalg.norm(x - y, axis=1) ** 2 / (2 * (sigma ** 2)))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :] , axis=2) ** 2 / (2 * (sigma ** 2)))
    return result

# def gaussian_kernel(x, y, eta=1):
#     if np.ndim(x) == 1 and np.ndim(y) == 1:
#         result = np.exp(- eta * np.linalg.norm(x - y))
#     elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
#         result = np.exp(- eta * np.linalg.norm(x - y, axis=1))
#     elif np.ndim(x) > 1 and np.ndim(y) > 1:
#         result = np.exp(- eta * np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2))
#     return result


def poly_kernel(x, y, eta=1.0, c=1.0, degree=3):
    return np.power(eta * np.dot(x, y.T) + c, degree)

def sigmoid_kernel(x, y, eta=1.0, c=1.0):
    return np.tanh(eta * np.dot(x, y.T) + c)

KERNEL_TYPES = {
    'linear': linear_kernel,
    'gaussian': gaussian_kernel,
    'poly': poly_kernel,
    'sigmoid': sigmoid_kernel
}