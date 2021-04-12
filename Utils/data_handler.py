from scipy.io import loadmat
import numpy as np


def load_data(data_set):
    """
    loading the datasets
    :param data_set:
    :return:
    """
    data = loadmat(f'../Data/{data_set}.mat')
    train_x = data['Yt']
    test_x = data['Yv']
    train_y = data['Ct']
    test_y = data['Cv']

    return train_x, test_x, train_y, test_y


def shuffle_data(train_x, train_y):
    """
    shuffling the training data
    :param train_x:
    :param train_y:
    :return:
    """
    train_data = np.concatenate([train_x, train_y])
    np.random.shuffle(np.transpose(train_data))
    return train_data


def initiate_batch(all_batches, batch_num, num_features):
    """
    splitting the batch data to X and Y
    :param all_batches:
    :param batch_num:
    :param num_features:
    :return:
    """
    x_batch = all_batches[batch_num][0:num_features, :]  # all the features rows including bias
    y_batch = all_batches[batch_num][num_features:, :]  # the label rows
    return x_batch, y_batch


def add_bias_neuron(train_x, test_x):
    """
    Adding bias neuron (vector of ones)
    :param train_x:
    :param test_x:
    :return:
    """
    train_ones_v = np.ones((1, train_x.shape[1]))
    test_one_v = np.ones((1, test_x.shape[1]))
    train_x = np.concatenate([train_x, train_ones_v], axis=0)
    test_x = np.concatenate([test_x, test_one_v], axis=0)
    return train_x, test_x


def pre_processing(train_x, train_y, batch_size):
    """
    Perform pre-processing to the data, including shuffling the data and split it to batches
    :param train_x:
    :param train_y:
    :param batch_size:
    :return:
    """
    m = train_x.shape[1]
    train_data_shf = shuffle_data(train_x, train_y)
    train_data_batches = np.array_split(train_data_shf, indices_or_sections=m // batch_size, axis=1)
    return train_data_batches
