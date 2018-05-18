import os
import sys
from data_loader import DataLoader
from fc_model import FCClassifierModel


def main():
    data_loader = DataLoader()
    # train_x, train_y, test_x, test_y = data_loader.load_dataset('mnist')
    train_set, test_set = data_loader.load_dataset('mnist')

    model = FCClassifierModel([784, 300, 100, 10])

    model.test(test_set)
    model.train(train_set, 10, test_set)

    # print(model.__classifier_output(train_x[0]))
    # print(model.predict(train_x[0]))


if __name__ == '__main__':
    main()

