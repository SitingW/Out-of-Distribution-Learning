import numpy as np
from scipy.linalg import sqrtm


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def weighted_norm(x, A):
    return np.sqrt(x.T @ A @ x)


def gaussian_sample_ellipsoid(center, design, radius):
    dim = len(center)
    sample = np.random.normal(0, 1, (dim,))
    res = np.real_if_close(center + np.linalg.solve(sqrtm(design), sample) * radius)
    return res