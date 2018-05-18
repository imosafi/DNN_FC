import os
import sys
from data_loader import DataLoader
from fc_model import FCClassifierModel


def main():
    data_loader = DataLoader()
    train_set, test_set = data_loader.load_dataset('mnist')

    model = FCClassifierModel([784, 300, 100, 10], activation='tanh')

    model.test(test_set)
    model.train(train_set, 10, test_set)



if __name__ == '__main__':
    main()

