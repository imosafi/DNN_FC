import numpy as np
import math
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


ACTIVATION_NAMES = ['sigmoid', 'tanh', 'relu']


def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# xavier initialization
def calc_init_fcnn_values(in_dim, out_dim):
    return math.sqrt(6) / math.sqrt(in_dim + out_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0, x)


def relu_derivative(a):
    a[a <= 0] = 0
    a[a > 0] = 1
    return a


def sigmoid_derivative(a):
    """expecting a to be sigmoid(x)"""
    return a * (1 - a)


def tanh_derivative(a):
    """expecting a to be tanh(x)"""
    return 1 - np.power(a, 2)


ACTIVATION_NAME_TO_FUNC = {'sigmoid': sigmoid, 'tanh': np.tanh, 'relu': relu}
ACTIVATION_NAME_TO_DERIVATIVE = {'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative, 'relu': relu_derivative}


def create_confusion_matrix(model, testset, categories_num):
    confusion_martix = []
    for i in range(categories_num):
        confusion_martix.append([0] * categories_num)

    for x, y in testset:
        confusion_martix[y][model.predict(x)] += 1

    print('Confustion matrix:')
    for l in range(categories_num):
        print(confusion_martix[l])


def ensure_directory_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def save_test_val_acc_loss_plots(train_acc, val_acc, train_loss, val_loss):
        output_dir = os.path.join(os.curdir, 'output')
        ensure_directory_exists(output_dir)

        plt.title('Net Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        blue_patch = mpatches.Patch(color='blue', label='training')
        red_patch = mpatches.Patch(color='red', label='validation')
        plt.plot([i for i in range(len(train_acc))], train_acc, color='tab:blue')
        plt.plot([i for i in range(len(val_acc))], val_acc, color='tab:red')
        plt.legend(handles=[red_patch, blue_patch], loc=4)

        plt.savefig(os.path.join(output_dir, 'Accuracies.png'), dpi=100)

        plt.clf()

        plt.title('Net Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        blue_patch = mpatches.Patch(color='blue', label='training')
        red_patch = mpatches.Patch(color='red', label='validation')
        plt.plot([i for i in range(len(train_loss))], train_loss, color='tab:blue')
        plt.plot([i for i in range(len(val_loss))], val_loss, color='tab:red')
        plt.legend(handles=[red_patch, blue_patch], loc=1)

        plt.savefig(os.path.join(output_dir, 'Losses.png'), dpi=100)