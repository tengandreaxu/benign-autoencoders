import numpy as np


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))
