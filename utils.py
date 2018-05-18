import numpy as np
import math

ACTIVATION_NAMES = ['sigmoid', 'relu', 'tanh']



def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# xavier initialization
def calc_init_fcnn_values(in_dim, out_dim):
    return math.sqrt(6) / math.sqrt(in_dim + out_dim)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    """expecting a to be tanh(x)"""
    return a * (1 - a)


def tanh_derivative(a):
    """expecting a to be tanh(x)"""
    return 1 - np.power(a, 2)


ACTIVATION_NAME_TO_FUNC = {'sigmoid': sigmoid, 'tanh': np.tanh}
ACTIVATION_NAME_TO_DERIVATIVE = {'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative}
