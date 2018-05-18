import os
import numpy as np
import gzip

# mnist dataset was manually downloaded from http://yann.lecun.com/exdb/mnist/
class DataLoader(object):

    def __init__(self):
        self.__path_data_dict = {}
        self.__data_dim_dict = {}
        self.__dataset_size_dict = {}

        self.__path_data_dict['mnist_train_x'] = 'data/mnist/train-images-idx3-ubyte.gz'
        self.__path_data_dict['mnist_train_y'] = 'data/mnist/train-labels-idx1-ubyte.gz'
        self.__path_data_dict['mnist_test_x'] = 'data/mnist/t10k-images-idx3-ubyte.gz'
        self.__path_data_dict['mnist_test_y'] = 'data/mnist/t10k-labels-idx1-ubyte.gz'

        self.__data_dim_dict['mnist'] = 28

        self.__dataset_size_dict['mnist_train'] = 60000
        self.__dataset_size_dict['mnist_test'] = 10000

    def extract_data(self, filename, num_images, dataset_name):
        img_size = self.__data_dim_dict[dataset_name]
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(img_size * img_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            data = data.reshape(num_images, img_size * img_size)
            return data

    @staticmethod
    def extract_labels(filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

    def load_dataset(self, name, normalize=True):
        data_dir = os.path.join('data', name)
        if not os.path.isdir(data_dir):
            raise Exception('dataset wasn\'t found')

        try:
            train_x = self.extract_data(self.__path_data_dict[name + '_train_x'],
                                        self.__dataset_size_dict[name + '_train'], name)
            train_y = self.extract_labels(self.__path_data_dict[name + '_train_y'],
                                          self.__dataset_size_dict[name + '_train'])
            test_x = self.extract_data(self.__path_data_dict[name + '_test_x'],
                                       self.__dataset_size_dict[name + '_test'], name)
            test_y = self.extract_labels(self.__path_data_dict[name + '_test_y'],
                                         self.__dataset_size_dict[name + '_test'])
            if normalize:
                train_x /= 255
                test_x /= 255
            return list(zip(train_x, train_y)), list(zip(test_x, test_y))

        except:
            raise Exception('Failed to load {} data'.format(name))
