import os
import torch
import random
import sys
import numpy as np
import time


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"

    # Mac M1 GPU Acceleration
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def accuracy(y_pred, y_true):
    # Get the class predictions (indices with the highest probability)
    _, predicted = torch.max(y_pred, 1)

    # Calculate the number of correct predictions
    correct = (predicted == y_true).sum().item()

    # Calculate accuracy as the ratio of correct predictions to total predictions
    acc = correct / y_true.size(0)

    return acc



