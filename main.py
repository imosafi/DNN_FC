import os
import sys
from data_loader import DataLoader
# from fc_model import FCModel


def main():
    data_loader = DataLoader()
    train_x, train_y, test_x, test_y = data_loader.load_dataset('mnist')

    # model = FCModel()



if __name__ == '__main__':
    main()

