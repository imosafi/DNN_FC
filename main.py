import os
import sys
from data_loader import DataLoader
from fc_model import FCClassifierModel
from utils import create_confusion_matrix


def main():
    data_loader = DataLoader()
    train_set, test_set = data_loader.load_dataset('mnist')

    model = FCClassifierModel([784, 200, 100, 10], activation='tanh')

    model.train(train_set, epochs=20, val=test_set)

    create_confusion_matrix(model, test_set, 10)


if __name__ == '__main__':
    main()

