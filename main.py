import os
import sys
from data_loader import DataLoader
from fc_model import FCClassifierModel
from utils import create_confusion_matrix


def main():
    data_loader = DataLoader()
    train_set, test_set = data_loader.load_dataset('mnist', input_normalization=False)

    model = FCClassifierModel([784, 200, 100, 10], activation='relu')

    model.train(train_set, epochs=30, val=test_set, LR=0.001, use_minibatch=False, use_decreasing_lr=False)

    create_confusion_matrix(model, test_set, categories_num=10)


if __name__ == '__main__':
    main()

