import numpy as np
import math

ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'tanh']

def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# xavier initialization
def calc_init_fcnn_values(in_dim, out_dim):
    return math.sqrt(6) / math.sqrt(in_dim + out_dim)
