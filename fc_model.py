import numpy as np
from utils import calc_init_fcnn_values, softmax, ACTIVATION_NAMES
from gradients_calculator import GradientsCalculator
import math
import random


class FCClassifierModel(object):
    """We'll use the same activation function for the entire network sigmoid/tanh"""
    def __init__(self, dim_sizes, activation='tanh'):
        if activation not in ACTIVATION_NAMES:
            raise Exception('Unknown activation function')
        self.__params = self.__init_params(dim_sizes)
        self.__activation_name = activation
        self.__gradient_calculator = GradientsCalculator(activation)


        c = 2

    def __init_params(self, dims):
        params = []
        init_value = calc_init_fcnn_values(dims[0], dims[-1])
        for index, val in enumerate(dims):
            if (index != len(dims) - 1):
                params.append([np.random.uniform(-init_value, init_value, (dims[index], dims[index + 1])),
                               np.zeros(dims[index + 1])])
        return params

    def __classifier_output(self, x):
        i = 0
        value = x
        while i < len(self.__params) - 1:
            value = np.tanh(np.dot(value, self.__params[i][0]) + self.__params[i][1])
            i += 1
        return softmax(np.dot(value, self.__params[-1][0]) + self.__params[-1][1])

    def __predict(self, x):
        return np.argmax(self.__classifier_output(x))

    def calc_loss(self, y, y_pred):
        return -np.log(y_pred[y])

    def train(self, train_set, epochs, val=None, use_minibatch=True, minibatch_size=32, LR=0.003):
        for epoch in range(epochs):
            output = []
            output.append('Epoch {}:'.format(epoch + 1))
            train_loss = 0.0
            random.shuffle(train_set)

            for x, y in train_set:
                y_pred = self.__classifier_output(x)
                loss = self.calc_loss(y, y_pred)
                train_loss += loss

                grads = self.__gradient_calculator.calc_gradients(x, y, y_pred, self.__params)

                i = 0
                while i < len(self.__params):
                    self.__params[i][0] -= LR * grads[i][0]
                    self.__params[i][1] -= LR * grads[i][1]
                    i += 1

            train_loss /= len(train_set)
            train_accuracy, _ = self.__calc_accuracy_and_loss(train_set)

            output.append('Train accuracy {}'.format(train_accuracy))
            output.append('Train loss {}'.format(train_loss))

            if val is not None:
                val_accuracy, val_loss = self.__calc_accuracy_and_loss(val)
                output.append('Validation accuracy {}'.format(val_accuracy))
                output.append('Validation loss {}'.format(val_loss))

            for t in output:
                print(t)

    def test(self, test_set):
        acc, loss = self.__calc_accuracy_and_loss(test_set)
        print('Test accuracy: {}, test loss: {}'.format(acc, loss))

    def __calc_accuracy_and_loss(self, data_set):
        good_preds, bad_preds, loss = 0.0, 0.0, 0.0
        for x, y in data_set:
            y_pred = self.__classifier_output(x)
            loss += self.calc_loss(y, y_pred)
            if np.argmax(y_pred) == y:
                good_preds += 1
            else:
                bad_preds += 1
        return good_preds / (good_preds + bad_preds), loss / len(data_set)
