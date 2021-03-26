from scipy.io import loadmat
import numpy as np


def load_data(data_set):
    data = loadmat(f'Data/{data_set}.mat')
    train_x = data['Yt']
    test_x = data['Yv']
    train_y = data['Ct']
    test_y = data['Cv']

    return train_x, test_x, train_y, test_y


def shuffle_data(train_x, test_x, train_y, test_y):
    train_data = np.concatenate([train_x, train_y])
    test_data = np.concatenate([test_x, test_y])
    np.random.shuffle(np.transpose(train_data))
    np.random.shuffle(np.transpose(test_data))
    return train_data, test_data


def pre_processing(train_x, test_x, train_y, test_y, batch_size):
    train_data_shf, test_data_shf = shuffle_data(train_x, test_x, train_y, test_y)
    train_data_batches = np.array_split(train_data_shf, indices_or_sections=batch_size, axis=1)
    test_data_batches = np.array_split(test_data_shf, indices_or_sections=batch_size, axis=1)
    # train_x_shf, train_y_shf = tuple(train_data_split)


def sgd_test():
    data_sets = ['SwissRollData', 'GMMData', 'PeaksData.mat']
    train_x, test_x, train_y, test_y = load_data(data_sets[0])  # loading dataset

    # Hyper-Params
    batches_sizes = [2 ** i for i in range(0, 7)]
    # lrs = [1 * (1/10)**i for i in range(2, 6)]
    lr = 1e-5
    pre_processing(train_x, test_x, train_y, test_y, 256)


sgd_test()
